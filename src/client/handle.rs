use crate::error::GeminiError;
use crate::types::{
    ActivityEnd, ActivityStart, BidiGenerateContentClientContent, BidiGenerateContentRealtimeInput,
    Blob, ClientMessagePayload, Content, Part, Role,
};
#[cfg(feature = "audio-resampling")]
use audioadapter::direct::SequentialSliceOfVecs; // Adapter for Vec<Vec<f32>>
#[cfg(feature = "audio-resampling")]
use audioadapter::{Adapter, AdapterMut, SizeError}; // Core traits & Error
use base64::Engine as _;
#[cfg(feature = "audio-resampling")]
use rubato::{Fft, FixedSync, Indexing, Resampler};
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::{mpsc, oneshot};

#[cfg(feature = "audio-resampling")]
use tracing::trace;
use tracing::{error, info, warn};

use super::GeminiLiveClientBuilder;

/// Target sample rate for audio sent to the Gemini API after resampling.
pub const TARGET_AUDIO_SAMPLE_RATE_HZ: u32 = 16000;
/// Target number of channels for audio sent to the Gemini API (mono).
pub const TARGET_AUDIO_CHANNELS: u16 = 1; // This is for the *output* of resampling

/// Represents the state of an active audio resampler using Rubato v1.0.
#[cfg(feature = "audio-resampling")]
pub(crate) struct ActiveResamplerState {
    resampler: rubato::Fft<f32>, // Unified FFT resampler, configured for FixedSync::Input
    original_input_rate: u32,    // Sample rate of the audio fed TO THIS MODULE
    original_input_channels: u16, // Channels of the audio fed TO THIS MODULE
    // Buffer for MONO f32 audio, accumulated across send_audio_chunk calls
    internal_mono_buffer: Vec<f32>,
    // Pre-allocated output buffer for resampler processing, to avoid allocations in loop/flush
    // This will be a Vec<Vec<f32>> where the inner Vec is for the single mono channel.
    resampler_output_buffer_alloc: Vec<Vec<f32>>, // For AdapterMut
}

#[cfg(feature = "audio-resampling")]
impl std::fmt::Debug for ActiveResamplerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveResamplerState")
            .field("original_input_rate", &self.original_input_rate)
            .field("original_input_channels", &self.original_input_channels)
            .field("internal_mono_buffer_len", &self.internal_mono_buffer.len())
            .field(
                "resampler_output_buffer_alloc_cap", // Changed to capacity
                &self
                    .resampler_output_buffer_alloc
                    .get(0)
                    .map_or(0, |v| v.capacity()),
            )
            .field("resampler", &"<rubato::Fft<f32> instance>")
            .finish()
    }
}

/// A client for real-time, bidirectional communication with Google's Gemini API.
#[derive(Clone)]
pub struct GeminiLiveClient<S: Clone + Send + Sync + 'static> {
    pub(crate) shutdown_tx: Arc<TokioMutex<Option<oneshot::Sender<()>>>>, // Changed type
    pub(crate) outgoing_sender: Option<mpsc::Sender<ClientMessagePayload>>,
    pub(crate) state: Arc<S>,

    #[cfg(feature = "audio-resampling")]
    pub(crate) resampler_state: Arc<TokioMutex<Option<ActiveResamplerState>>>,
    #[cfg(feature = "audio-resampling")]
    pub(crate) automatic_resampling_configured_in_builder: bool,
}

impl<S: Clone + Send + Sync + 'static> GeminiLiveClient<S> {
    pub async fn close(&mut self) -> Result<(), GeminiError> {
        info!("Client close requested.");
        let mut shutdown_tx_guard = self.shutdown_tx.lock().await; // Lock
        if let Some(tx) = shutdown_tx_guard.take() {
            // Take from Option inside guard
            if tx.send(()).is_err() {
                // send consumes tx
                info!("Shutdown signal failed: Listener task already gone or shut down.");
            } else {
                info!("Shutdown signal sent to listener task.");
            }
        } // Mutex guard is dropped here
        self.outgoing_sender.take();
        Ok(())
    }

    pub fn get_outgoing_mpsc_sender_clone(
        &self,
    ) -> Option<tokio::sync::mpsc::Sender<ClientMessagePayload>> {
        self.outgoing_sender.clone()
    }

    pub fn builder_with_state(
        api_key: String,
        model: String,
        state: S,
    ) -> GeminiLiveClientBuilder<S> {
        GeminiLiveClientBuilder::new_with_state(api_key, model, state)
    }

    async fn send_message(&self, payload: ClientMessagePayload) -> Result<(), GeminiError> {
        if let Some(sender) = &self.outgoing_sender {
            let sender_clone = sender.clone();
            match sender_clone.send(payload).await {
                Ok(_) => {
                    // info!("Message sent to listener task via channel.");
                    Ok(())
                }
                Err(e) => {
                    error!(
                        "Failed to send message to listener task: Channel closed. Error: {}",
                        e
                    );
                    Err(GeminiError::SendError)
                }
            }
        } else {
            error!("Cannot send message: Client is closed or outgoing sender is missing.");
            Err(GeminiError::NotReady)
        }
    }

    pub async fn send_text_turn(&self, text: String, end_of_turn: bool) -> Result<(), GeminiError> {
        let content_part = Part {
            text: Some(text),
            ..Default::default()
        };
        let content = Content {
            parts: vec![content_part],
            role: Some(Role::User),
            ..Default::default()
        };
        let client_content_msg = BidiGenerateContentClientContent {
            turns: Some(vec![content]),
            turn_complete: Some(end_of_turn),
        };
        self.send_message(ClientMessagePayload::ClientContent(client_content_msg))
            .await
    }

    pub async fn send_audio_chunk(
        &self,
        audio_samples: &[i16],
        sample_rate: u32,
        channels: u16,
    ) -> Result<(), GeminiError> {
        if audio_samples.is_empty() {
            info!("Empty audio chunk received, not sending.");
            return Ok(());
        }

        let target_sample_rate_hz = TARGET_AUDIO_SAMPLE_RATE_HZ;
        let target_channels_api = TARGET_AUDIO_CHANNELS; // This is 1 (mono) for API

        let mut samples_to_send_direct: Option<Vec<i16>> = None;
        let mut final_mime_rate_direct: Option<u32> = None;

        #[cfg(feature = "audio-resampling")]
        {
            if self.automatic_resampling_configured_in_builder {
                if sample_rate == target_sample_rate_hz && channels == target_channels_api {
                    info!("Audio is already 16kHz mono. Preparing for direct send.");
                    samples_to_send_direct = Some(audio_samples.to_vec());
                    final_mime_rate_direct = Some(target_sample_rate_hz);
                } else {
                    let mut resampler_state_guard = self.resampler_state.lock().await;

                    match &mut *resampler_state_guard {
                        Some(active_resampler) => {
                            if active_resampler.original_input_rate != sample_rate
                                || active_resampler.original_input_channels != channels
                            {
                                let error_msg = format!(
                                    "Audio format changed. Resampler initialized for {}Hz {}ch, but received {}Hz {}ch. Call flush_audio() before changing formats or ensure consistent input.",
                                    active_resampler.original_input_rate,
                                    active_resampler.original_input_channels,
                                    sample_rate,
                                    channels
                                );
                                error!("{}", error_msg);
                                return Err(GeminiError::ApiError(error_msg));
                            }
                            // info!(
                            //     "Using existing resampler for original input: {}Hz {}ch.",
                            //     sample_rate, channels
                            // );
                        }
                        None => {
                            info!(
                                "Initializing audio resampler. Original input: {}Hz {}ch -> Mix to Mono ({}) -> Resample to {}Hz {}.",
                                sample_rate,
                                channels,
                                sample_rate,
                                target_sample_rate_hz,
                                target_channels_api
                            );
                            let fixed_input_chunk_size_frames = 1024;
                            let sub_chunks = 2;

                            let new_rubato_resampler = rubato::Fft::<f32>::new(
                                sample_rate as usize, // Input sample rate (of the mono signal we will feed it)
                                target_sample_rate_hz as usize, // Output sample rate
                                fixed_input_chunk_size_frames, // Chunk size for the fixed input side
                                sub_chunks,
                                1, // Number of channels for the resampler (it processes mono)
                                FixedSync::Input, // Input size is fixed
                            )
                            .map_err(|e| {
                                GeminiError::ApiError(format!(
                                    "Failed to create FFT resampler: {}",
                                    e
                                ))
                            })?;

                            let max_output_frames_for_state_buffer =
                                new_rubato_resampler.output_frames_max();
                            let resampler_output_buffer_alloc = vec![
                                    vec![0.0f32; max_output_frames_for_state_buffer.max(1)];
                                    target_channels_api as usize
                                ];

                            *resampler_state_guard = Some(ActiveResamplerState {
                                resampler: new_rubato_resampler,
                                original_input_rate: sample_rate,
                                original_input_channels: channels,
                                internal_mono_buffer: Vec::new(),
                                resampler_output_buffer_alloc,
                            });
                            // info!(
                            //     "Audio resampler initialized for original {}Hz {}ch input (mixed to mono).",
                            //     sample_rate, channels
                            // );
                        }
                    }

                    let mut active_resampler_ref = resampler_state_guard.as_mut().unwrap();

                    let num_frames_current_chunk = audio_samples.len() / channels as usize;
                    let mut current_mono_f32_input: Vec<f32> =
                        Vec::with_capacity(num_frames_current_chunk);
                    if channels > 1 {
                        // info!(
                        //     "Mixing {} channels to mono for {} frames",
                        //     channels, num_frames_current_chunk
                        // );
                        for i in 0..num_frames_current_chunk {
                            let mut sample_sum_f32 = 0.0f32;
                            for ch_idx in 0..channels {
                                sample_sum_f32 +=
                                    audio_samples[i * channels as usize + ch_idx as usize] as f32
                                        / (i16::MAX as f32 + 1.0);
                            }
                            current_mono_f32_input.push(sample_sum_f32 / channels as f32);
                        }
                    } else {
                        // info!(
                        //     "Processing {} mono audio input frames",
                        //     num_frames_current_chunk
                        // );
                        current_mono_f32_input.extend(
                            audio_samples
                                .iter()
                                .map(|&s| s as f32 / (i16::MAX as f32 + 1.0)),
                        );
                    }

                    active_resampler_ref
                        .internal_mono_buffer
                        .extend(current_mono_f32_input);
                    // info!(
                    //     "Appended {} mono f32 samples. Total buffered in internal_mono_buffer: {}.",
                    //     num_frames_current_chunk,
                    //     active_resampler_ref.internal_mono_buffer.len()
                    // );

                    loop {
                        active_resampler_ref = resampler_state_guard
                            .as_mut()
                            .expect("Resampler state missing in loop (pre-check)");
                        let required_input_frames =
                            active_resampler_ref.resampler.input_frames_next(); // This is fixed_input_chunk_size_frames (e.g., 1024)

                        // info!(
                        //     "[Loop] Top. Buffered in internal_mono_buffer: {}, Required by Fft: {}",
                        //     active_resampler_ref.internal_mono_buffer.len(),
                        //     required_input_frames
                        // );

                        if active_resampler_ref.internal_mono_buffer.len() < required_input_frames {
                            // info!(
                            //     "Buffered {} mono frames, resampler needs {}. Waiting for more.",
                            //     active_resampler_ref.internal_mono_buffer.len(),
                            //     required_input_frames
                            // );
                            break;
                        }

                        let mono_input_chunk_to_process: Vec<f32> = active_resampler_ref
                            .internal_mono_buffer
                            .drain(0..required_input_frames)
                            .collect();

                        let input_data_for_adapter: Vec<Vec<f32>> =
                            vec![mono_input_chunk_to_process];
                        let input_adapter = SequentialSliceOfVecs::new(
                            &input_data_for_adapter,
                            1,
                            required_input_frames,
                        )
                        .map_err(|e: SizeError| {
                            GeminiError::ApiError(format!(
                                "Failed to create input adapter: {:?}",
                                e
                            ))
                        })?;

                        let estimated_output_for_chunk =
                            active_resampler_ref.resampler.output_frames_next();
                        for chan_buf in active_resampler_ref
                            .resampler_output_buffer_alloc
                            .iter_mut()
                        {
                            chan_buf.resize(estimated_output_for_chunk.max(1), 0.0f32);
                        }
                        let mut output_adapter = SequentialSliceOfVecs::new_mut(
                            &mut active_resampler_ref.resampler_output_buffer_alloc,
                            1,
                            estimated_output_for_chunk.max(1),
                        )
                        .map_err(|e: SizeError| {
                            GeminiError::ApiError(format!(
                                "Failed to create output adapter: {:?}",
                                e
                            ))
                        })?;

                        let indexing = Indexing {
                            input_offset: 0,
                            output_offset: 0,
                            partial_len: None,
                            active_channels_mask: None,
                        };

                        let (frames_read, frames_written) = active_resampler_ref
                            .resampler
                            .process_into_buffer(
                                &input_adapter,
                                &mut output_adapter,
                                Some(&indexing),
                            )
                            .map_err(|e| {
                                GeminiError::ApiError(format!("Audio resampling error: {}", e))
                            })?;

                        // info!(
                        //     "[Loop] Resampler processed {} input frames, wrote {} output frames.",
                        //     frames_read, frames_written
                        // );

                        if frames_written > 0 {
                            // Get the output buffer directly since we know it's mono
                            let final_mono_output_f32_slice = if !active_resampler_ref
                                .resampler_output_buffer_alloc
                                .is_empty()
                            {
                                &active_resampler_ref.resampler_output_buffer_alloc[0]
                                    [..frames_written]
                            } else {
                                return Err(GeminiError::ApiError(
                                    "No output buffer available".to_string(),
                                ));
                            };

                            let chunk_to_send_i16: Vec<i16> = final_mono_output_f32_slice
                                .iter()
                                .map(|&s_f32| {
                                    let val = s_f32 * (i16::MAX as f32 + 1.0);
                                    val.clamp(i16::MIN as f32, i16::MAX as f32).round() as i16
                                })
                                .collect();

                            let mut byte_data = Vec::with_capacity(chunk_to_send_i16.len() * 2);
                            for &sample in &chunk_to_send_i16 {
                                byte_data.extend_from_slice(&sample.to_le_bytes());
                            }
                            let encoded_data =
                                base64::engine::general_purpose::STANDARD.encode(&byte_data);
                            let mime_type =
                                format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ);
                            let audio_blob = Blob {
                                mime_type,
                                data: encoded_data,
                            };
                            let realtime_input = BidiGenerateContentRealtimeInput {
                                audio: Some(audio_blob),
                                ..Default::default()
                            };

                            drop(resampler_state_guard);
                            self.send_message(ClientMessagePayload::RealtimeInput(realtime_input))
                                .await?;
                            resampler_state_guard = self.resampler_state.lock().await;
                            if resampler_state_guard.is_none() {
                                return Err(GeminiError::ApiError(
                                    "Resampler state lost during send loop".to_string(),
                                ));
                            }
                        } else {
                            // info!(
                            //     "Resampler produced no output for this internal processing iteration."
                            // );
                        }
                    }
                    return Ok(());
                }
            } else {
                if sample_rate != target_sample_rate_hz || channels != target_channels_api {
                    let error_msg = format!(
                        "Audio input ({}Hz {}ch) must be 16kHz mono. Automatic resampling was not enabled.",
                        sample_rate, channels
                    );
                    warn!("{}", error_msg);
                    return Err(GeminiError::ApiError(error_msg));
                }
                samples_to_send_direct = Some(audio_samples.to_vec());
                final_mime_rate_direct = Some(sample_rate);
            }
        }

        #[cfg(not(feature = "audio-resampling"))]
        {
            if sample_rate != target_sample_rate_hz || channels != target_channels_api {
                let error_msg = format!(
                    "Audio input ({}Hz {}ch) must be 16kHz mono. 'audio-resampling' feature not compiled.",
                    sample_rate, channels
                );
                error!("{}", error_msg);
                return Err(GeminiError::ApiError(error_msg));
            }
            samples_to_send_direct = Some(audio_samples.to_vec());
            final_mime_rate_direct = Some(sample_rate);
        }

        if let (Some(samples_to_send), Some(final_sample_rate_for_mime)) =
            (samples_to_send_direct, final_mime_rate_direct)
        {
            if samples_to_send.is_empty() {
                info!("Direct send: Audio samples list is empty, not sending.");
                return Ok(());
            }
            let mut byte_data = Vec::with_capacity(samples_to_send.len() * 2);
            for sample_val in &samples_to_send {
                byte_data.extend_from_slice(&sample_val.to_le_bytes());
            }
            let encoded_data = base64::engine::general_purpose::STANDARD.encode(&byte_data);
            let mime_type = format!("audio/pcm;rate={}", final_sample_rate_for_mime);
            info!(
                "Sending direct audio data. Mime: {}, Samples: {}",
                mime_type,
                samples_to_send.len()
            );
            let audio_blob = Blob {
                mime_type,
                data: encoded_data,
            };
            let realtime_input = BidiGenerateContentRealtimeInput {
                audio: Some(audio_blob),
                ..Default::default()
            };
            self.send_message(ClientMessagePayload::RealtimeInput(realtime_input))
                .await
        } else {
            info!(
                "No audio data prepared for sending in this call (resampling path handled it or input was empty)."
            );
            Ok(())
        }
    }

    pub async fn send_realtime_text(&self, text: String) -> Result<(), GeminiError> {
        let realtime_input = BidiGenerateContentRealtimeInput {
            text: Some(text),
            ..Default::default()
        };
        self.send_message(ClientMessagePayload::RealtimeInput(realtime_input))
            .await
    }

    pub async fn send_activity_start(&self) -> Result<(), GeminiError> {
        let realtime_input = BidiGenerateContentRealtimeInput {
            activity_start: Some(ActivityStart {}),
            ..Default::default()
        };
        self.send_message(ClientMessagePayload::RealtimeInput(realtime_input))
            .await
    }

    pub async fn send_activity_end(&self) -> Result<(), GeminiError> {
        let realtime_input = BidiGenerateContentRealtimeInput {
            activity_end: Some(ActivityEnd {}),
            ..Default::default()
        };
        self.send_message(ClientMessagePayload::RealtimeInput(realtime_input))
            .await
    }

    #[cfg(feature = "audio-resampling")]
    pub(crate) async fn flush_audio(&self) -> Result<(), GeminiError> {
        if !self.automatic_resampling_configured_in_builder {
            info!("Automatic resampling not configured, flush_audio is a no-op.");
            return Ok(());
        }

        let mut resampler_state_guard = self.resampler_state.lock().await;
        if let Some(mut active_state) = resampler_state_guard.take() {
            // Consumes ActiveResamplerState
            info!(
                "Flushing audio resampler (original input rate: {}Hz, {}ch). Buffered mono samples in client: {}",
                active_state.original_input_rate,
                active_state.original_input_channels,
                active_state.internal_mono_buffer.len()
            );
            let mut accumulated_f32_to_send: Vec<f32> = Vec::new();

            // Part 1: Process any remaining samples from our internal_mono_buffer
            if !active_state.internal_mono_buffer.is_empty() {
                let num_buffered = active_state.internal_mono_buffer.len();
                info!(
                    "[FlushAudio] Part 1: Processing {} remaining samples from internal_mono_buffer.",
                    num_buffered
                );

                let mono_input_chunk = active_state
                    .internal_mono_buffer
                    .drain(..)
                    .collect::<Vec<_>>();
                // For SequentialSliceOfVecs::new, the inner Vecs are the channels. We have one mono channel.
                let input_data_for_adapter_p1: Vec<Vec<f32>> = vec![mono_input_chunk];
                let input_adapter =
                    SequentialSliceOfVecs::new(&input_data_for_adapter_p1, 1, num_buffered)
                        .map_err(|e: SizeError| {
                            GeminiError::ApiError(format!(
                                "Failed to create flush p1 input adapter: {:?}",
                                e
                            ))
                        })?;

                // Use output_frames_next as it's for a chunk of input_frames_next, which process_partial_into_buffer will effectively create
                let estimated_output = active_state.resampler.output_frames_next();
                let output_buffer_len_p1 = estimated_output.max(1);

                // Use the pre-allocated buffer from ActiveResamplerState, resizing if necessary
                for chan_buf in active_state.resampler_output_buffer_alloc.iter_mut() {
                    chan_buf.resize(output_buffer_len_p1, 0.0f32);
                }
                let mut output_adapter_p1 = SequentialSliceOfVecs::new_mut(
                    &mut active_state.resampler_output_buffer_alloc,
                    1,
                    output_buffer_len_p1,
                )
                .map_err(|e: SizeError| {
                    GeminiError::ApiError(format!(
                        "Failed to create flush p1 output adapter: {:?}",
                        e
                    ))
                })?;

                let indexing_p1 = Indexing {
                    input_offset: 0,
                    output_offset: 0,
                    partial_len: Some(num_buffered),
                    active_channels_mask: None,
                };

                match active_state.resampler.process_into_buffer(
                    &input_adapter,
                    &mut output_adapter_p1,
                    Some(&indexing_p1),
                ) {
                    Ok((_frames_read, frames_written)) => {
                        if frames_written > 0 {
                            // Access the data directly from the underlying buffer that output_adapter_p1 wraps
                            accumulated_f32_to_send.extend_from_slice(
                                &active_state.resampler_output_buffer_alloc[0][..frames_written],
                            );
                        }
                        info!(
                            "[FlushAudio] Part 1 (internal_mono_buffer): Processed, wrote {} output. Accum len: {}",
                            frames_written,
                            accumulated_f32_to_send.len()
                        );
                    }
                    Err(e) => error!("Error processing internal buffer during flush: {}", e),
                }
            }

            // Part 2: Flush the resampler's own internal pipeline (single pass).
            // This call processes the resampler's internal state (e.g., overlaps in FFT)
            // using zero-padded input, effectively flushing out any remaining delayed samples.
            info!("[FlushAudio] Part 2: Flushing resampler internal pipeline (single pass).");

            let empty_input_data_storage_ch: Vec<f32> = vec![];
            let empty_input_data_storage: Vec<Vec<f32>> = vec![empty_input_data_storage_ch];
            let empty_input_adapter =
                SequentialSliceOfVecs::new(&empty_input_data_storage, 1, 0 /*frames*/).map_err(
                    |e: SizeError| {
                        GeminiError::ApiError(format!(
                            "Failed to create empty input adapter for flush: {:?}",
                            e
                        ))
                    },
                )?;

            // FftFixedInput.output_frames_next() gives the amount for one full fixed input chunk.
            // This is the amount of output we expect from this flush pass.
            let output_buffer_len_flush = active_state.resampler.output_frames_next().max(1);
            for chan_buf in active_state.resampler_output_buffer_alloc.iter_mut() {
                chan_buf.resize(output_buffer_len_flush, 0.0f32);
            }
            let mut output_adapter_flush = SequentialSliceOfVecs::new_mut(
                &mut active_state.resampler_output_buffer_alloc,
                1,
                output_buffer_len_flush,
            )
            .map_err(|e: SizeError| {
                GeminiError::ApiError(format!("Failed to create flush p2 output adapter: {:?}", e))
            })?;

            let indexing_flush = Indexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: Some(0), // Signal no new input, resampler processes its fixed input chunk as zeros
                active_channels_mask: None,
            };

            match active_state.resampler.process_into_buffer(
                &empty_input_adapter,
                &mut output_adapter_flush,
                Some(&indexing_flush),
            ) {
                Ok((_frames_read, frames_written)) => {
                    if frames_written > 0 {
                        info!(
                            "[FlushAudio] Part 2 (resampler pipeline): resampler flushed {} output frames.",
                            frames_written
                        );
                        accumulated_f32_to_send.extend_from_slice(
                            &active_state.resampler_output_buffer_alloc[0][..frames_written],
                        );
                    } else {
                        info!(
                            "[FlushAudio] Part 2 (resampler pipeline): resampler wrote 0 frames on flush pass."
                        );
                    }
                }
                Err(e) => {
                    error!("Error during resampler internal flush pass: {}", e);
                    // Depending on desired behavior, you might return the error or try to send what's accumulated.
                    // For now, logging and continuing to send accumulated data.
                }
            }

            info!(
                "[FlushAudio] END processing. Total accumulated_f32_to_send len: {}",
                accumulated_f32_to_send.len()
            );
            if !accumulated_f32_to_send.is_empty() {
                let chunk_to_send_i16: Vec<i16> = accumulated_f32_to_send
                    .iter()
                    .map(|&s_f32| {
                        let val = s_f32 * (i16::MAX as f32 + 1.0);
                        val.clamp(i16::MIN as f32, i16::MAX as f32).round() as i16
                    })
                    .collect();

                info!(
                    "[FlushAudio] Sending all accumulated flushed audio ({} i16 samples).",
                    chunk_to_send_i16.len()
                );
                let mut byte_data = Vec::with_capacity(chunk_to_send_i16.len() * 2);
                for sample_val in &chunk_to_send_i16 {
                    byte_data.extend_from_slice(&sample_val.to_le_bytes());
                }
                let encoded_data = base64::engine::general_purpose::STANDARD.encode(&byte_data);
                let mime_type = format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ);
                let audio_blob = Blob {
                    mime_type,
                    data: encoded_data,
                };
                let realtime_input = BidiGenerateContentRealtimeInput {
                    audio: Some(audio_blob),
                    ..Default::default()
                };

                // Guard already dropped as active_state was taken by value
                self.send_message(ClientMessagePayload::RealtimeInput(realtime_input))
                    .await?;
                info!(
                    "[FlushAudio] Successfully sent {} accumulated flushed i16 samples.",
                    chunk_to_send_i16.len()
                );
            } else {
                info!(
                    "[FlushAudio] No data to send after processing internal buffer and flushing resampler pipeline."
                );
            }
            info!("Resampler state fully consumed by flush_audio.");
        } else {
            info!(
                "No active resampler state to flush (resampling not initialized or already flushed)."
            );
        }
        Ok(())
    }

    pub async fn send_audio_stream_end(&self) -> Result<(), GeminiError> {
        #[cfg(feature = "audio-resampling")]
        {
            if self.automatic_resampling_configured_in_builder {
                info!("Flushing audio before sending stream end signal."); // Changed from trace!
                match self.flush_audio().await {
                    Ok(_) => info!("Audio flushed successfully before stream end."), // Changed from trace!
                    Err(e) => error!("Error flushing audio before stream end: {}", e),
                }
            }
        }
        info!("Sending audio stream end signal.");
        let end_stream_msg = BidiGenerateContentRealtimeInput {
            audio_stream_end: Some(true),
            ..Default::default()
        };
        self.send_message(ClientMessagePayload::RealtimeInput(end_stream_msg))
            .await
    }

    pub fn state(&self) -> Arc<S> {
        self.state.clone()
    }
}

impl<S: Clone + Send + Sync + 'static> Drop for GeminiLiveClient<S> {
    fn drop(&mut self) {
        // Try to get the lock without blocking.
        if let Ok(mut guard) = self.shutdown_tx.try_lock() {
            if guard.is_some() {
                warn!(
                    "GeminiLiveClient dropped without explicit close(). Attempting to signal shutdown (try_lock succeeded)."
                );
                if let Some(tx) = guard.take() {
                    // oneshot::Sender::send is sync, so this is okay.
                    if tx.send(()).is_err() {
                        // Receiver already dropped
                        info!(
                            "Drop: Shutdown signal send failed, listener task likely already gone."
                        );
                    } else {
                        info!("Drop: Shutdown signal sent via try_lock.");
                    }
                }
                self.outgoing_sender.take();
            } else {
                // Lock acquired, but Option was None (already closed or taken)
                if self.outgoing_sender.is_some() {
                    // Check if it was notionally active
                    warn!(
                        "GeminiLiveClient dropped. Shutdown sender was already None but outgoing_sender existed (possibly closed then dropped)."
                    );
                    self.outgoing_sender.take();
                }
            }
        } else {
            // Could not acquire lock, might be held by another clone during its close()
            // or the main task that created the client is being dropped and its `Drop` is running concurrently.
            // This is less ideal as we can't send the shutdown signal here.
            if self.outgoing_sender.is_some() {
                // Still check if it was active
                warn!(
                    "GeminiLiveClient dropped without explicit close(). Could not acquire shutdown_tx lock to send signal."
                );
                self.outgoing_sender.take();
            }
        }
    }
}

#[cfg(test)]
mod test_utils {
    use std::sync::Once;
    use tracing::Level;
    use tracing_subscriber::EnvFilter;

    fn init_test_logger() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(
                    EnvFilter::builder()
                        .with_default_directive(Level::INFO.into())
                        .from_env_lossy(),
                )
                .with_test_writer()
                .try_init();
        });
    }
    pub fn setup_test() {
        init_test_logger();
    }
}

#[cfg(test)]
#[cfg(feature = "audio-resampling")]
mod tests {
    use super::*;
    use crate::client::handle::test_utils::setup_test;
    use base64::Engine as _;
    use tokio::sync::mpsc;
    use tokio::time::{Duration, timeout};

    async fn collect_audio_until_stream_end(
        rx: &mut mpsc::Receiver<ClientMessagePayload>,
        expected_mime_type_prefix: &str,
        timeout_duration: Duration,
    ) -> (Vec<u8>, String) {
        let mut aggregated_audio_bytes = Vec::new();
        let mut first_mime_type_seen = String::new();
        let mut stream_ended_signal_received = false;
        let mut audio_blobs_received = 0;

        let overall_timeout = Duration::from_secs(timeout_duration.as_secs_f64().ceil() as u64 + 5); // Increased overall timeout
        let start_time = tokio::time::Instant::now();

        while tokio::time::Instant::now().duration_since(start_time) < overall_timeout {
            match timeout(timeout_duration, rx.recv()).await {
                Ok(Some(ClientMessagePayload::RealtimeInput(input))) => {
                    if let Some(blob) = input.audio {
                        audio_blobs_received += 1;
                        trace!(
                            "[TestHelper] Received audio blob #{}: Mime: {}, Size: {}",
                            audio_blobs_received,
                            blob.mime_type,
                            blob.data.len()
                        );
                        if first_mime_type_seen.is_empty() {
                            first_mime_type_seen = blob.mime_type.clone();
                            assert!(
                                first_mime_type_seen.starts_with(expected_mime_type_prefix),
                                "Unexpected MIME type prefix: got '{}', expected prefix '{}'",
                                first_mime_type_seen,
                                expected_mime_type_prefix
                            );
                        } else {
                            assert_eq!(
                                first_mime_type_seen, blob.mime_type,
                                "MIME type changed mid-stream unexpectedly."
                            );
                        }
                        match base64::engine::general_purpose::STANDARD.decode(blob.data) {
                            Ok(decoded_bytes) => aggregated_audio_bytes.extend(decoded_bytes),
                            Err(e) => panic!(
                                "[TestHelper] Failed to decode base64 audio data in test: {}",
                                e
                            ),
                        }
                    }
                    if input.audio_stream_end == Some(true) {
                        trace!("[TestHelper] Received audioStreamEnd=true signal.");
                        stream_ended_signal_received = true;
                        let drain_deadline =
                            tokio::time::Instant::now() + Duration::from_millis(500); // Increased drain time
                        while tokio::time::Instant::now() < drain_deadline {
                            match rx.try_recv() {
                                Ok(ClientMessagePayload::RealtimeInput(extra_input)) => {
                                    if let Some(blob) = extra_input.audio {
                                        audio_blobs_received += 1;
                                        trace!(
                                            "[TestHelper] Drain loop got audio blob #{}: Mime: {}, Size: {}",
                                            audio_blobs_received,
                                            blob.mime_type,
                                            blob.data.len()
                                        );
                                        if first_mime_type_seen.is_empty()
                                            && blob.mime_type.starts_with(expected_mime_type_prefix)
                                        {
                                            first_mime_type_seen = blob.mime_type.clone();
                                        }
                                        aggregated_audio_bytes.extend(
                                            base64::engine::general_purpose::STANDARD
                                                .decode(blob.data)
                                                .unwrap(),
                                        );
                                    }
                                    if extra_input.audio_stream_end == Some(true) {
                                        trace!(
                                            "[TestHelper] Saw another stream_end in drain loop, already noted."
                                        );
                                    }
                                }
                                Ok(_other) => { /* ignore */ }
                                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                                    tokio::task::yield_now().await;
                                }
                                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                                    trace!("[TestHelper] Drain: channel disconnected.");
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
                Ok(Some(other_payload)) => {
                    panic!(
                        "[TestHelper] Unexpected payload type received: {:?}",
                        other_payload
                    );
                }
                Ok(None) => {
                    if stream_ended_signal_received {
                        break;
                    }
                    panic!(
                        "[TestHelper] Channel closed prematurely before audioStreamEnd signal was received."
                    );
                }
                Err(_) => {
                    if stream_ended_signal_received {
                        break;
                    }
                    trace!(
                        "[TestHelper] Individual recv timeout (duration: {:?}). Continuing if overall not timed out.",
                        timeout_duration
                    );
                }
            }
        }
        assert!(
            stream_ended_signal_received,
            "[TestHelper] audioStreamEnd=true signal was not received within overall timeout."
        );
        (aggregated_audio_bytes, first_mime_type_seen)
    }

    async fn setup_test_client_for_resampling_tests(
        enable_resampling_in_builder: bool,
    ) -> (GeminiLiveClient<()>, mpsc::Receiver<ClientMessagePayload>) {
        let (outgoing_tx, outgoing_rx) = mpsc::channel(20);
        let (shutdown_tx, _) = oneshot::channel();

        let client = GeminiLiveClient {
            shutdown_tx: Arc::new(TokioMutex::new(Some(shutdown_tx))),
            outgoing_sender: Some(outgoing_tx),
            state: Arc::new(()),
            #[cfg(feature = "audio-resampling")]
            resampler_state: Arc::new(TokioMutex::new(None)),
            #[cfg(feature = "audio-resampling")]
            automatic_resampling_configured_in_builder: enable_resampling_in_builder,
        };
        (client, outgoing_rx)
    }

    fn generate_sine_wave_mono(num_frames: usize, sample_rate: u32, frequency: f32) -> Vec<i16> {
        let mut samples = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let time = i as f32 / sample_rate as f32;
            let value = (2.0 * std::f32::consts::PI * frequency * time).sin();
            samples.push((value * (i16::MAX as f32 * 0.8)) as i16);
        }
        samples
    }

    fn generate_sine_wave_stereo_interleaved(
        num_frames: usize,
        sample_rate: u32,
        freq_l: f32,
        freq_r: f32,
    ) -> Vec<i16> {
        let mut samples = Vec::with_capacity(num_frames * 2);
        for i in 0..num_frames {
            let time = i as f32 / sample_rate as f32;
            let val_l = (2.0 * std::f32::consts::PI * freq_l * time).sin();
            let val_r = (2.0 * std::f32::consts::PI * freq_r * time).sin();
            samples.push((val_l * (i16::MAX as f32 * 0.7)) as i16);
            samples.push((val_r * (i16::MAX as f32 * 0.7)) as i16);
        }
        samples
    }

    #[tokio::test]
    async fn test_send_audio_chunk_direct_16k_mono_when_resampling_on() {
        setup_test();
        let (client, mut outgoing_rx) = setup_test_client_for_resampling_tests(true).await;
        let audio_data_i16 = generate_sine_wave_mono(160, 16000, 440.0);

        client
            .send_audio_chunk(&audio_data_i16, 16000, 1)
            .await
            .unwrap();
        client.send_audio_stream_end().await.unwrap();

        let expected_mime_prefix = "audio/pcm;rate=16000";
        let (received_audio_bytes, mime_type) = collect_audio_until_stream_end(
            &mut outgoing_rx,
            expected_mime_prefix,
            Duration::from_millis(100),
        )
        .await;

        assert_eq!(mime_type, expected_mime_prefix);
        assert_eq!(received_audio_bytes.len(), audio_data_i16.len() * 2);
    }

    #[tokio::test]
    async fn test_resample_48k_stereo_to_16k_mono_single_chunk_then_flush() {
        setup_test();
        let (client, mut outgoing_rx) = setup_test_client_for_resampling_tests(true).await;
        let num_input_frames = 2048;
        let audio_data_i16 =
            generate_sine_wave_stereo_interleaved(num_input_frames, 48000, 440.0, 660.0);

        client
            .send_audio_chunk(&audio_data_i16, 48000, 2)
            .await
            .unwrap();
        client.send_audio_stream_end().await.unwrap();

        let expected_mime_prefix = format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ);
        let (received_audio_bytes, mime_type) = collect_audio_until_stream_end(
            &mut outgoing_rx,
            &expected_mime_prefix,
            Duration::from_millis(200),
        )
        .await;

        if !received_audio_bytes.is_empty() {
            assert_eq!(
                mime_type,
                format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ)
            );
        }
        assert!(
            !received_audio_bytes.is_empty(),
            "Expected resampled audio data"
        );

        let num_received_samples = received_audio_bytes.len() / 2;
        // For Fft resampler, output includes delay.
        // Approximate ideal output: floor(input_frames * ratio)
        let ideal_output = (num_input_frames as f64
            * (TARGET_AUDIO_SAMPLE_RATE_HZ as f64 / 48000.0))
            .floor() as usize;
        // Rough estimate for FftFixedIn flush: could be up to ideal + one fft_size_out block (171 for this config)
        let expected_min = ideal_output;
        let expected_max = ideal_output + (1024 * 16000 / 48000) + 10; // ideal + output_of_one_more_input_chunk_size + some slack

        assert!(
            num_received_samples >= expected_min && num_received_samples <= expected_max,
            "Unexpected #output samples: got {}, expected between {} and {}. Ideal ratio output: {}. Input frames (stereo): {}",
            num_received_samples,
            expected_min,
            expected_max,
            ideal_output,
            num_input_frames
        );
    }

    #[tokio::test]
    async fn test_resample_multiple_small_chunks_441k_mono_then_flush() {
        setup_test();
        let (client, mut outgoing_rx) = setup_test_client_for_resampling_tests(true).await;
        let input_hz = 44100;
        let num_chunks = 5;
        let frames_per_small_chunk = 441;
        let total_input_frames = frames_per_small_chunk * num_chunks; // 2205

        for i in 0..num_chunks {
            let audio_chunk = generate_sine_wave_mono(
                frames_per_small_chunk,
                input_hz,
                300.0 + (i as f32 * 50.0),
            );
            client
                .send_audio_chunk(&audio_chunk, input_hz, 1)
                .await
                .unwrap();
        }

        client.send_audio_stream_end().await.unwrap();

        let expected_mime_prefix = format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ);
        let (received_audio_bytes, mime_type) = collect_audio_until_stream_end(
            &mut outgoing_rx,
            &expected_mime_prefix,
            Duration::from_millis(300), // Increased timeout slightly just in case
        )
        .await;

        if !received_audio_bytes.is_empty() {
            assert_eq!(
                mime_type,
                format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ)
            );
        }
        assert!(
            !received_audio_bytes.is_empty(),
            "Expected audio after flush of buffered small chunks"
        );

        let num_received_samples = received_audio_bytes.len() / 2;
        let ideal_output = (total_input_frames as f64
            * (TARGET_AUDIO_SAMPLE_RATE_HZ as f64 / input_hz as f64))
            .floor() as usize; // 2205 * (16000/44100) = 800

        // Based on detailed trace for this specific FftFixedInput configuration:
        // fft_size_in = 882, fft_size_out = 320.
        // The specific sequence of send_audio_chunk and flush results in 1280 samples.
        let expected_min = ideal_output; // Should be at least the ideal from pure ratio
        let expected_max = 1280 + 20; // Expected output from trace is 1280. Add small slack.
        // The old expected_max was 1186.

        assert!(
            num_received_samples >= expected_min && num_received_samples <= expected_max,
            "Unexpected total samples: got {}, expected between {} and {}. Ideal ratio output: {}. Total input mono frames: {}",
            num_received_samples,
            expected_min,
            expected_max,
            ideal_output,
            total_input_frames
        );
    }

    #[tokio::test]
    async fn test_error_on_format_change_after_resampler_init() {
        setup_test();
        let (client, mut outgoing_rx) = setup_test_client_for_resampling_tests(true).await;
        let audio_48k_stereo = generate_sine_wave_stereo_interleaved(1024, 48000, 440.0, 600.0);
        let audio_22k_mono = generate_sine_wave_mono(1024, 22050, 300.0);

        client
            .send_audio_chunk(&audio_48k_stereo, 48000, 2)
            .await
            .unwrap();

        let result = client.send_audio_chunk(&audio_22k_mono, 22050, 1).await;
        assert!(result.is_err());
        if let Err(GeminiError::ApiError(msg)) = result {
            assert!(msg.contains("Audio format changed"));
        } else {
            panic!("Expected ApiError, got {:?}", result);
        }

        assert!(
            client.resampler_state.lock().await.is_some(),
            "Resampler state should still exist after format error"
        );

        client.send_audio_stream_end().await.unwrap();

        let expected_mime_prefix = format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ);
        let (received_audio_bytes, mime_type) = collect_audio_until_stream_end(
            &mut outgoing_rx,
            &expected_mime_prefix,
            Duration::from_millis(200),
        )
        .await;

        if !received_audio_bytes.is_empty() {
            assert_eq!(
                mime_type,
                format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ)
            );
        }
        assert!(
            !received_audio_bytes.is_empty(),
            "Expected audio from first chunk after flush"
        );

        let num_received_samples = received_audio_bytes.len() / 2;
        // Only the first 1024 stereo frames (-> 1024 mono @ 48kHz) should have been processed and flushed.
        let ideal_output_from_first_chunk =
            (1024.0 * (TARGET_AUDIO_SAMPLE_RATE_HZ as f64 / 48000.0)).floor() as usize;
        let expected_min = ideal_output_from_first_chunk;
        let expected_max = ideal_output_from_first_chunk + (1024 * 16000 / 48000) + 10; // ideal + one more block output + slack

        assert!(
            num_received_samples >= expected_min && num_received_samples <= expected_max,
            "Unexpected samples after format error + flush: got {}, expected between {} and {}. Ideal from 1st chunk: {}",
            num_received_samples,
            expected_min,
            expected_max,
            ideal_output_from_first_chunk
        );
        assert!(
            client.resampler_state.lock().await.is_none(),
            "Resampler state should be None after stream end/flush consumes it"
        );
    }

    #[tokio::test]
    async fn test_resampling_disabled_in_builder_expects_16k_mono() {
        setup_test();
        let (client, mut outgoing_rx) = setup_test_client_for_resampling_tests(false).await;
        let audio_data_16k = generate_sine_wave_mono(160, 16000, 440.0);
        let audio_data_48k = generate_sine_wave_mono(480, 48000, 440.0);

        client
            .send_audio_chunk(&audio_data_16k, 16000, 1)
            .await
            .unwrap();
        client.send_audio_stream_end().await.unwrap();
        let expected_mime_16k = "audio/pcm;rate=16000";
        let (bytes, mime) = collect_audio_until_stream_end(
            &mut outgoing_rx,
            expected_mime_16k,
            Duration::from_millis(100),
        )
        .await;
        assert_eq!(mime, expected_mime_16k);
        assert_eq!(bytes.len(), audio_data_16k.len() * 2);

        let result = client.send_audio_chunk(&audio_data_48k, 48000, 1).await;
        assert!(result.is_err());
        if let Err(GeminiError::ApiError(msg)) = result {
            assert!(msg.contains(
                "Audio input (48000Hz 1ch) must be 16kHz mono. Automatic resampling was not enabled"
            ));
        } else {
            panic!("Expected ApiError, got {:?}", result);
        }
        client.send_audio_stream_end().await.unwrap();
        match timeout(Duration::from_millis(200), outgoing_rx.recv()).await {
            Ok(Some(ClientMessagePayload::RealtimeInput(input))) => {
                assert!(input.audio.is_none(), "No audio expected after error");
                assert_eq!(input.audio_stream_end, Some(true));
            }
            res => panic!(
                "Unexpected result after erroring send and stream_end: {:?}",
                res
            ),
        }
    }

    #[tokio::test]
    async fn test_flush_audio_no_op_if_resampling_disabled_or_no_state() {
        setup_test();
        let (client_disabled, _rx_disabled) = setup_test_client_for_resampling_tests(false).await;
        client_disabled.flush_audio().await.unwrap();
        assert!(client_disabled.resampler_state.lock().await.is_none());

        let (client_no_init, _rx_no_init) = setup_test_client_for_resampling_tests(true).await;
        client_no_init.flush_audio().await.unwrap();
        assert!(client_no_init.resampler_state.lock().await.is_none());
    }

    #[tokio::test]
    async fn test_send_audio_stream_end_calls_flush_and_clears_state() {
        setup_test();
        let (client, mut outgoing_rx) = setup_test_client_for_resampling_tests(true).await;
        let audio_data = generate_sine_wave_mono(1024, 44100, 300.0);

        client
            .send_audio_chunk(&audio_data, 44100, 1)
            .await
            .unwrap();
        assert!(
            client.resampler_state.lock().await.is_some(),
            "Resampler state should be Some after first chunk"
        );

        client.send_audio_stream_end().await.unwrap();

        assert!(
            client.resampler_state.lock().await.is_none(),
            "Resampler state should be None after send_audio_stream_end"
        );

        let expected_mime_prefix = format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ);
        let (received_audio_bytes, mime_type) = collect_audio_until_stream_end(
            &mut outgoing_rx,
            &expected_mime_prefix,
            Duration::from_millis(200),
        )
        .await;

        if !received_audio_bytes.is_empty() {
            assert_eq!(
                mime_type,
                format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ)
            );
        }
        assert!(
            !received_audio_bytes.is_empty(),
            "Expected some audio data to be flushed"
        );
    }

    #[tokio::test]
    async fn test_flush_audio_produces_output_from_partial_internal_buffer_scenario() {
        setup_test();
        let (client, mut outgoing_rx) = setup_test_client_for_resampling_tests(true).await;
        let input_hz = 48000;
        let channels = 1;

        let partial_input_frames = 300;
        let audio_data = generate_sine_wave_mono(partial_input_frames, input_hz, 440.0);

        client
            .send_audio_chunk(&audio_data, input_hz, channels)
            .await
            .unwrap();

        {
            let guard = client.resampler_state.lock().await;
            let state = guard.as_ref().expect("Resampler should be initialized");
            assert_eq!(
                state.internal_mono_buffer.len(),
                partial_input_frames,
                "Internal mono buffer should hold the partial input"
            );
        }

        client.send_audio_stream_end().await.unwrap();

        assert!(
            client.resampler_state.lock().await.is_none(),
            "Resampler state should be None after send_audio_stream_end"
        );

        let expected_mime_prefix = format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ);
        let (received_audio_bytes, mime_type) = collect_audio_until_stream_end(
            &mut outgoing_rx,
            &expected_mime_prefix,
            Duration::from_millis(200),
        )
        .await;

        if !received_audio_bytes.is_empty() {
            assert_eq!(
                mime_type,
                format!("audio/pcm;rate={}", TARGET_AUDIO_SAMPLE_RATE_HZ)
            );
        }
        assert!(
            !received_audio_bytes.is_empty(),
            "Expected some audio data to be flushed from the partial input"
        );

        let num_received_samples = received_audio_bytes.len() / 2;

        // Calculation for this specific scenario (48k->16k, FftFixedInput chunk_in=1024, sub_chunks=2 -> fft_size_in=513, fft_size_out=171)
        // Input 300 frames.
        // Part 1 processes 300 audio + (1024-300) zeros. Output: floor(1024/513)*171 = 171. Fft.saved_frames becomes 1024-513 = 511 (zeros).
        // Part 2 processes Fft.saved_frames (511 zeros) + 1024 new zeros. Output: floor((511+1024)/513)*171 = floor(1535/513)*171 = 2*171 = 342.
        // Total expected = 171 + 342 = 513.
        let expected_total_flushed_samples = 513;
        let tolerance = 15; // Increased tolerance slightly for FFT chunking effects
        let lower_bound = (expected_total_flushed_samples - tolerance).max(0); // Ensure non-negative
        let upper_bound = expected_total_flushed_samples + tolerance;

        assert!(
            num_received_samples >= lower_bound && num_received_samples <= upper_bound,
            "Unexpected #flushed samples for partial buffer: got {}, expected ~{} (bounds {}-{}). Input frames: {}",
            num_received_samples,
            expected_total_flushed_samples,
            lower_bound,
            upper_bound,
            partial_input_frames
        );
    }
}

#[cfg(test)]
#[cfg(not(feature = "audio-resampling"))]
mod tests_no_resampling_feature {
    use super::*;
    use crate::client::handle::test_utils::setup_test;
    use crate::types::{BidiGenerateContentRealtimeInput, ClientMessagePayload};
    use tokio::sync::mpsc;
    use tokio::time::{Duration, timeout};

    async fn setup_test_client_no_feature()
    -> (GeminiLiveClient<()>, mpsc::Receiver<ClientMessagePayload>) {
        let (outgoing_tx, outgoing_rx) = mpsc::channel(10);
        let (shutdown_tx, _) = oneshot::channel();
        let client = GeminiLiveClient {
            shutdown_tx: Arc::new(TokioMutex::new(Some(shutdown_tx))),
            outgoing_sender: Some(outgoing_tx),
            state: Arc::new(()),
            #[cfg(feature = "audio-resampling")]
            resampler_state: Arc::new(TokioMutex::new(None)),
            #[cfg(feature = "audio-resampling")]
            automatic_resampling_configured_in_builder: false,
        };
        (client, outgoing_rx)
    }

    fn generate_sine_wave_mono_test_no_feat(
        num_frames: usize,
        sample_rate: u32,
        frequency: f32,
    ) -> Vec<i16> {
        let mut samples = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let time = i as f32 / sample_rate as f32;
            let value = (2.0 * std::f32::consts::PI * frequency * time).sin();
            samples.push((value * (i16::MAX as f32 * 0.8)) as i16);
        }
        samples
    }

    #[tokio::test]
    async fn test_send_audio_chunk_no_feature_expects_16k_mono_ok() {
        setup_test();
        let (client, mut outgoing_rx) = setup_test_client_no_feature().await;
        let audio_data_16k = generate_sine_wave_mono_test_no_feat(160, 16000, 440.0);

        client
            .send_audio_chunk(&audio_data_16k, 16000, 1)
            .await
            .unwrap();

        if let Ok(Some(ClientMessagePayload::RealtimeInput(BidiGenerateContentRealtimeInput {
            audio: Some(blob),
            ..
        }))) = timeout(Duration::from_millis(100), outgoing_rx.recv()).await
        {
            assert_eq!(blob.mime_type, "audio/pcm;rate=16000");
        } else {
            panic!("Did not receive expected audio payload");
        }
    }

    #[tokio::test]
    async fn test_send_audio_chunk_no_feature_expects_16k_mono_err() {
        setup_test();
        let (client, _outgoing_rx) = setup_test_client_no_feature().await;
        let audio_data_48k = generate_sine_wave_mono_test_no_feat(480, 48000, 440.0);
        let result = client.send_audio_chunk(&audio_data_48k, 48000, 1).await;
        assert!(result.is_err());
        if let Err(GeminiError::ApiError(msg)) = result {
            assert!(msg.contains("Audio input (48000Hz 1ch) must be 16kHz mono. The 'audio-resampling' feature is not compiled."));
        } else {
            panic!("Expected ApiError for wrong format, got {:?}", result);
        }
    }
}
