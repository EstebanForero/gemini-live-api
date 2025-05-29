use super::{GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT, GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT};
use crate::error::GeminiError;
use crate::types::{BidiGenerateContentRealtimeInput, Blob, ClientMessagePayload};
use audioadapter::direct::SequentialSliceOfVecs;
use base64::Engine as _;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};

/// Holds the active state for an initialized audio input resampler.
///
/// This includes the `rubato::Fft` resampler instance, information about the
/// original audio format it was initialized for, and internal buffers for
/// processing.
#[derive(Debug)]
pub(crate) struct InputResamplerPipelineState {
    resampler: Fft<f32>,
    original_input_rate: u32,
    original_input_channels: u16,
    internal_mono_buffer: Vec<f32>,
    // Pre-allocated buffer for the resampler's mono output.
    resampler_output_buffer_alloc: Vec<Vec<f32>>,
}

/// Manages the audio input stream, including resampling to the target format
/// required by the Gemini API (16kHz mono).
///
/// This pipeline handles:
/// - Initialization of a `rubato::Fft` resampler based on the first audio chunk's format.
/// - Stereo to mono mixing if the input audio has multiple channels.
/// - Buffering of mono audio samples.
/// - Processing audio in chunks through the resampler.
/// - Sending resampled audio (as `ClientMessagePayload`) via the provided `output_sender`.
/// - Flushing all buffered and internally delayed audio when a stream ends.
pub(crate) struct InputResamplerPipeline {
    state: Option<InputResamplerPipelineState>,
    output_sender: mpsc::Sender<ClientMessagePayload>,
}

impl InputResamplerPipeline {
    /// Creates a new `InputResamplerPipeline`.
    ///
    /// # Arguments
    ///
    /// * `output_sender`: An MPSC sender channel used to forward the processed
    ///   `ClientMessagePayload` (containing the resampled audio blob) to the
    ///   main WebSocket connection task.
    pub(crate) fn new(output_sender: mpsc::Sender<ClientMessagePayload>) -> Self {
        Self {
            state: None,
            output_sender,
        }
    }

    /// Processes a chunk of incoming audio samples.
    ///
    /// If the pipeline has not been initialized, it will be configured based on the
    /// `input_sample_rate` and `input_channels` of this first chunk. Subsequent calls
    /// must provide audio in the same format, or an error will be returned. To change
    /// formats, `complete_and_reset_stream` must be called first.
    ///
    /// Audio is mixed to mono (if necessary), converted to `f32`, buffered, and then
    /// resampled in suitable blocks. Resampled audio is sent via the `output_sender`.
    ///
    /// # Arguments
    ///
    /// * `audio_samples_i16`: Slice of raw i16 audio samples.
    /// * `input_sample_rate`: Sample rate of the input audio in Hz.
    /// * `input_channels`: Number of channels in the input audio.
    ///
    /// # Errors
    ///
    /// Returns `GeminiError::ApiError` if the audio format changes after initialization
    /// without an intermediate call to `complete_and_reset_stream`.
    /// Returns `GeminiError::InternalError` or `GeminiError::AudioResamplingError`
    /// for internal processing issues.
    pub(crate) async fn process_chunk(
        &mut self,
        audio_samples_i16: &[i16],
        input_sample_rate: u32,
        input_channels: u16,
    ) -> Result<(), GeminiError> {
        if audio_samples_i16.is_empty() {
            return Ok(());
        }

        if self.state.is_none() {
            self.initialize_state(input_sample_rate, input_channels)?;
        }

        let state = self.state.as_mut().ok_or_else(|| {
            GeminiError::InternalError(
                "Resampler state unexpectedly None after init check.".to_string(),
            )
        })?;

        if state.original_input_rate != input_sample_rate
            || state.original_input_channels != input_channels
        {
            return Err(GeminiError::ApiError(format!(
                "Audio format changed. Resampler initialized for {}Hz {}ch, but received {}Hz {}ch. Call flush() before changing formats.",
                state.original_input_rate,
                state.original_input_channels,
                input_sample_rate,
                input_channels
            )));
        }

        let num_frames_current_chunk = audio_samples_i16.len() / input_channels as usize;
        let mut current_mono_f32_input: Vec<f32> = Vec::with_capacity(num_frames_current_chunk);

        if input_channels > 1 {
            for i in 0..num_frames_current_chunk {
                let mut sample_sum_f32 = 0.0f32;
                for ch_idx in 0..input_channels {
                    sample_sum_f32 +=
                        audio_samples_i16[i * input_channels as usize + ch_idx as usize] as f32
                            / (i16::MAX as f32 + 1.0);
                }
                current_mono_f32_input.push(sample_sum_f32 / input_channels as f32);
            }
        } else {
            current_mono_f32_input.extend(
                audio_samples_i16
                    .iter()
                    .map(|&s| s as f32 / (i16::MAX as f32 + 1.0)),
            );
        }

        state.internal_mono_buffer.extend(current_mono_f32_input);
        trace!(
            "[InputPipeline] Appended {} mono f32 samples. Total buffered: {}.",
            num_frames_current_chunk,
            state.internal_mono_buffer.len()
        );

        let sender_clone = self.output_sender.clone();
        Self::process_buffered_audio_and_send_static(state, sender_clone).await
    }

    /// Finalizes the current audio stream by processing any buffered audio and
    /// flushing the resampler's internal delay. After this call, the pipeline
    /// is reset and ready to be initialized with a new audio stream format on the
    /// next call to `process_chunk`.
    ///
    /// This should be called when the audio input source indicates the end of a
    /// logical stream (e.g., user stops speaking, file ends).
    pub(crate) async fn complete_and_reset_stream(&mut self) -> Result<(), GeminiError> {
        if self.state.is_none() {
            debug!(
                "[InputPipeline] complete_and_reset_stream called but no active resampler state (already flushed or never initialized)."
            );
            return Ok(());
        }

        let sender_clone = self.output_sender.clone();

        if let Some(mut current_state) = self.state.take() {
            debug!(
                "[InputPipeline] Flushing. Buffered mono samples in pipeline: {}",
                current_state.internal_mono_buffer.len()
            );

            let mut total_output_generated_during_flush = 0;

            // Part 1
            match Self::flush_process_remaining_buffer(&mut current_state, &sender_clone).await {
                Ok(produced) => total_output_generated_during_flush += produced,
                Err(e) => {
                    self.state = Some(current_state); // Put state back on error before P2
                    return Err(e);
                }
            }

            // Part 2
            let mut remaining_delay_to_flush = current_state.resampler.output_delay();
            if total_output_generated_during_flush >= remaining_delay_to_flush {
                remaining_delay_to_flush = 0;
            } else {
                remaining_delay_to_flush =
                    remaining_delay_to_flush.saturating_sub(total_output_generated_during_flush);
            }

            if let Err(e) = Self::flush_resampler_internal_pipeline(
                &mut current_state,
                remaining_delay_to_flush,
                &sender_clone,
            )
            .await
            {
                // Even if P2 fails, P1 might have processed. State is already taken.
                // Depending on desired atomicity, might not put state back, or log and continue.
                error!("[InputPipeline] Error during P2 flush: {}", e);
                // self.state remains None (taken)
                return Err(e);
            }
            // self.state remains None as it was .take()n
        }

        info!("[InputPipeline] Flush complete.");
        Ok(())
    }

    fn initialize_state(
        &mut self,
        input_sample_rate: u32,
        input_channels: u16, // original input channels from source
    ) -> Result<(), GeminiError> {
        info!(
            "[InputPipeline] Initializing for input: {}Hz {}ch -> Resample to {}Hz {}ch (Mono for Gemini API).",
            input_sample_rate,
            input_channels,
            GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT,
            GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT
        );
        let resampler_fixed_input_chunk_size_frames = 1024;
        let sub_chunks = 2;

        // The resampler processes the mono mix. So, its input rate is the original input_sample_rate.
        let resampler = Fft::<f32>::new(
            input_sample_rate as usize,
            GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT as usize,
            resampler_fixed_input_chunk_size_frames,
            sub_chunks,
            GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT as usize, // Resampler configured for 1 channel (mono processing and output)
            FixedSync::Input,
        )
        .map_err(|e| {
            GeminiError::AudioResamplingError(format!("Failed to create Fft resampler: {}", e))
        })?;

        let max_output_frames = resampler.output_frames_max();
        let resampler_output_buffer_alloc = vec![
            vec![0.0f32; max_output_frames.max(1)];
            GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT as usize
        ]; // Mono output buffer

        self.state = Some(InputResamplerPipelineState {
            resampler,
            original_input_rate: input_sample_rate,
            original_input_channels: input_channels, // Store original input channels for validation
            internal_mono_buffer: Vec::with_capacity(resampler_fixed_input_chunk_size_frames * 2),
            resampler_output_buffer_alloc,
        });
        Ok(())
    }

    async fn process_buffered_audio_and_send_static(
        state: &mut InputResamplerPipelineState,
        output_sender: mpsc::Sender<ClientMessagePayload>,
    ) -> Result<(), GeminiError> {
        loop {
            let required_input_frames = state.resampler.input_frames_next();
            if state.internal_mono_buffer.len() < required_input_frames
                || required_input_frames == 0
            {
                break;
            }

            let chunk_to_process: Vec<f32> = state
                .internal_mono_buffer
                .drain(..required_input_frames)
                .collect();

            let input_for_adapter = vec![chunk_to_process];
            let input_adapter = SequentialSliceOfVecs::new(
                &input_for_adapter,
                GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT as usize,
                required_input_frames,
            )
            .map_err(|e| {
                GeminiError::AudioResamplingError(format!("Input adapter error: {}", e))
            })?;

            let output_frames_next = state.resampler.output_frames_next();
            state.resampler_output_buffer_alloc[0].resize(output_frames_next.max(1), 0.0);

            let mut output_adapter = SequentialSliceOfVecs::new_mut(
                &mut state.resampler_output_buffer_alloc,
                GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT as usize,
                output_frames_next.max(1),
            )
            .map_err(|e| {
                GeminiError::AudioResamplingError(format!("Output adapter error: {}", e))
            })?;

            // `process_into_buffer` returns (frames_read, frames_written)
            // frames_read: number of input frames consumed from the input adapter for this call.
            // frames_written: number of output frames produced into the output adapter for this call.
            let (_frames_consumed_from_input_adapter, frames_produced_to_output_adapter) = state
                .resampler
                .process_into_buffer(
                    &input_adapter,
                    &mut output_adapter,
                    None, // Default indexing: process all of input_adapter up to input_frames_next()
                )
                .map_err(|e| GeminiError::AudioResamplingError(e.to_string()))?;

            if frames_produced_to_output_adapter > 0 {
                let resampled_f32_slice =
                    &state.resampler_output_buffer_alloc[0][..frames_produced_to_output_adapter];
                let samples_i16: Vec<i16> = resampled_f32_slice
                    .iter()
                    .map(|&sample_f32| {
                        (sample_f32 * (i16::MAX as f32 + 1.0))
                            .clamp(i16::MIN as f32, i16::MAX as f32)
                            .round() as i16
                    })
                    .collect();
                if !samples_i16.is_empty() {
                    Self::send_resampled_audio_bytes_static(&output_sender, samples_i16).await?;
                }
            }
        }
        Ok(())
    }

    async fn flush_process_remaining_buffer(
        current_state: &mut InputResamplerPipelineState,
        output_sender: &mpsc::Sender<ClientMessagePayload>,
    ) -> Result<usize, GeminiError> {
        let mut frames_produced_total = 0;
        if !current_state.internal_mono_buffer.is_empty() {
            let partial_len = current_state.internal_mono_buffer.len();
            let chunk_to_process: Vec<f32> = current_state.internal_mono_buffer.drain(..).collect();

            let input_for_adapter = vec![chunk_to_process];
            let input_adapter = SequentialSliceOfVecs::new(
                &input_for_adapter,
                GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT as usize,
                partial_len,
            )
            .map_err(|e| {
                GeminiError::AudioResamplingError(format!("Flush P1 input adapter error: {}", e))
            })?;

            let output_frames_next = current_state.resampler.output_frames_next();
            current_state.resampler_output_buffer_alloc[0].resize(output_frames_next.max(1), 0.0);
            let mut output_adapter = SequentialSliceOfVecs::new_mut(
                &mut current_state.resampler_output_buffer_alloc,
                GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT as usize,
                output_frames_next.max(1),
            )
            .map_err(|e| {
                GeminiError::AudioResamplingError(format!("Flush P1 output adapter error: {}", e))
            })?;

            let indexing = Indexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: Some(partial_len),
                active_channels_mask: None,
            };

            let (_frames_consumed, frames_produced) = current_state
                .resampler
                .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
                .map_err(|e| {
                    GeminiError::AudioResamplingError(format!("Flush P1 resampling error: {}", e))
                })?;

            if frames_produced > 0 {
                frames_produced_total += frames_produced;
                let resampled_f32_slice =
                    &current_state.resampler_output_buffer_alloc[0][..frames_produced];
                let samples_i16: Vec<i16> = resampled_f32_slice
                    .iter()
                    .map(|&s| {
                        (s * (i16::MAX as f32 + 1.0))
                            .clamp(i16::MIN as f32, i16::MAX as f32)
                            .round() as i16
                    })
                    .collect();
                if !samples_i16.is_empty() {
                    Self::send_resampled_audio_bytes_static(output_sender, samples_i16).await?;
                }
            }
            debug!(
                "[InputPipeline] Flush P1: consumed {} (from {} buffered), produced {}",
                _frames_consumed, partial_len, frames_produced
            );
        }
        Ok(frames_produced_total)
    }

    async fn flush_resampler_internal_pipeline(
        current_state: &mut InputResamplerPipelineState,
        mut remaining_delay_to_flush: usize,
        output_sender: &mpsc::Sender<ClientMessagePayload>,
    ) -> Result<(), GeminiError> {
        debug!(
            "[InputPipeline] Flush P2: Pumping zeros. Expecting ~{} more frames from delay.",
            remaining_delay_to_flush
        );
        let mut p2_output_count = 0;
        let max_p2_iterations = 5;

        for _iter in 0..max_p2_iterations {
            if remaining_delay_to_flush == 0 && p2_output_count > 0 {
                break;
            }
            let empty_input_data_storage_ch: Vec<f32> = vec![];
            let empty_input_data: Vec<Vec<f32>> = vec![empty_input_data_storage_ch];
            let empty_adapter = SequentialSliceOfVecs::new(
                &empty_input_data,
                GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT as usize,
                0,
            )
            .map_err(|e| {
                GeminiError::AudioResamplingError(format!("Flush P2 empty adapter error: {}", e))
            })?;

            let output_frames_for_this_pass = current_state.resampler.output_frames_next();
            current_state.resampler_output_buffer_alloc[0]
                .resize(output_frames_for_this_pass.max(1), 0.0);
            let mut output_adapter_flush = SequentialSliceOfVecs::new_mut(
                &mut current_state.resampler_output_buffer_alloc,
                GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT as usize,
                output_frames_for_this_pass.max(1),
            )
            .map_err(|e| {
                GeminiError::AudioResamplingError(format!("Flush P2 output adapter error: {}", e))
            })?;

            let indexing_flush = Indexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: Some(0),
                active_channels_mask: None,
            };

            let (_frames_consumed, frames_produced) = current_state
                .resampler
                .process_into_buffer(
                    &empty_adapter,
                    &mut output_adapter_flush,
                    Some(&indexing_flush),
                )
                .map_err(|e| {
                    GeminiError::AudioResamplingError(format!("Flush P2 resampling error: {}", e))
                })?;

            if frames_produced > 0 {
                p2_output_count += frames_produced;
                remaining_delay_to_flush = remaining_delay_to_flush.saturating_sub(frames_produced);
                let resampled_f32_slice =
                    &current_state.resampler_output_buffer_alloc[0][..frames_produced];
                let samples_i16: Vec<i16> = resampled_f32_slice
                    .iter()
                    .map(|&s| {
                        (s * (i16::MAX as f32 + 1.0))
                            .clamp(i16::MIN as f32, i16::MAX as f32)
                            .round() as i16
                    })
                    .collect();
                if !samples_i16.is_empty() {
                    Self::send_resampled_audio_bytes_static(output_sender, samples_i16).await?;
                }
            } else {
                debug!(
                    "[InputPipeline] Flush P2: Resampler produced 0 frames, considering it flushed."
                );
                break;
            }
        }
        if remaining_delay_to_flush > 0
            && p2_output_count < current_state.resampler.output_delay() / 2
        {
            warn!(
                "[InputPipeline] Flush P2: May not have flushed all delay samples. Expected ~{}, got {}",
                current_state.resampler.output_delay(),
                p2_output_count
            );
        }
        Ok(())
    }

    async fn send_resampled_audio_bytes_static(
        output_sender: &mpsc::Sender<ClientMessagePayload>,
        samples_i16: Vec<i16>,
    ) -> Result<(), GeminiError> {
        if samples_i16.is_empty() {
            return Ok(());
        }
        let mut byte_data = Vec::with_capacity(samples_i16.len() * 2);
        for sample_val in samples_i16 {
            byte_data.extend_from_slice(&sample_val.to_le_bytes());
        }

        let encoded_data = base64::engine::general_purpose::STANDARD.encode(&byte_data);
        let mime_type = format!(
            "audio/pcm;rate={}",
            GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT
        );

        let audio_blob = Blob {
            mime_type,
            data: encoded_data,
        };

        let realtime_input = BidiGenerateContentRealtimeInput {
            audio: Some(audio_blob),
            ..Default::default()
        };

        output_sender
            .send(ClientMessagePayload::RealtimeInput(realtime_input))
            .await
            .map_err(|e| {
                error!(
                    "[InputPipeline] Failed to send resampled audio to connection task: {}",
                    e
                );
                GeminiError::SendError
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ClientMessagePayload;
    use tokio::sync::mpsc;

    fn assert_audio_payload_details(
        payload: &ClientMessagePayload,
        expected_rate: u32,
        min_samples: usize,
        max_samples: usize,
    ) {
        match payload {
            ClientMessagePayload::RealtimeInput(realtime_input) => {
                let audio_blob = realtime_input.audio.as_ref().expect("Expected audio blob");
                assert_eq!(
                    audio_blob.mime_type,
                    format!("audio/pcm;rate={}", expected_rate)
                );
                let decoded_bytes = base64::engine::general_purpose::STANDARD
                    .decode(&audio_blob.data)
                    .unwrap();
                let num_samples = decoded_bytes.len() / 2;
                assert!(
                    num_samples >= min_samples && num_samples <= max_samples,
                    "Expected {} to {} samples, got {}",
                    min_samples,
                    max_samples,
                    num_samples
                );
            }
            _ => panic!("Unexpected client message payload type: {:?}", payload),
        }
    }

    #[tokio::test]
    async fn test_pipeline_initialization_and_process_one_chunk() {
        let (tx, mut rx) = mpsc::channel(10);
        let mut pipeline = InputResamplerPipeline::new(tx);

        let input_rate = 48000;
        let input_channels = 2;
        let num_mono_frames_for_resampler_input_chunk = 1024; // Resampler's fixed input chunk size
        let audio_i16: Vec<i16> =
            vec![100; num_mono_frames_for_resampler_input_chunk * input_channels as usize];

        pipeline
            .process_chunk(&audio_i16, input_rate, input_channels)
            .await
            .unwrap();

        match tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await {
            Ok(Some(payload)) => {
                // For Fft(48k->16k, chunk=1024, sub_chunks=2, ch=1, FixedInput)
                // Internal fft_size_in=513, fft_size_out=171.
                // When 1024 input frames are processed, it can do one internal block of 513.
                // Output = 1 * 171 = 171.
                let expected_output_this_pass = 171;
                assert_audio_payload_details(
                    &payload,
                    GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT,
                    expected_output_this_pass - 5,
                    expected_output_this_pass + 5,
                );
            }
            Ok(None) => panic!("Channel closed unexpectedly"),
            Err(_) => panic!("Timed out waiting for resampled audio chunk"),
        }

        assert!(pipeline.state.is_some());
        let state = pipeline.state.as_ref().unwrap();
        assert_eq!(state.original_input_rate, 48000);
        assert_eq!(state.original_input_channels, 2);
        // After processing exactly one resampler input chunk, internal_mono_buffer should be empty.
        assert!(
            state.internal_mono_buffer.is_empty(),
            "Internal buffer should be empty after processing a full resampler input chunk"
        );
    }

    #[tokio::test]
    async fn test_pipeline_buffering_and_flush() {
        let (tx, mut rx) = mpsc::channel(10);
        let mut pipeline = InputResamplerPipeline::new(tx);

        let input_rate = 48000;
        let input_channels = 1;
        let partial_audio_i16: Vec<i16> = vec![200; 500];
        pipeline
            .process_chunk(&partial_audio_i16, input_rate, input_channels)
            .await
            .unwrap();

        match rx.try_recv() {
            Err(mpsc::error::TryRecvError::Empty) => { /* Expected */ }
            res => panic!("Expected empty channel after partial chunk, got {:?}", res),
        }
        assert_eq!(
            pipeline.state.as_ref().unwrap().internal_mono_buffer.len(),
            500
        );

        pipeline.complete_and_reset_stream().await.unwrap();

        let mut total_flushed_samples = 0;
        let mut messages_received = 0;

        // Expecting 2 messages from flush:
        // 1. Output from processing the buffered 500 frames (P1 of flush)
        // 2. Output from flushing the resampler's internal state (P2 of flush, single call)
        let expected_messages_from_flush = 2;
        for i in 0..expected_messages_from_flush {
            // Changed from 3 to 2
            match tokio::time::timeout(std::time::Duration::from_millis(300), rx.recv()).await {
                Ok(Some(payload)) => {
                    messages_received += 1;
                    match &payload {
                        ClientMessagePayload::RealtimeInput(realtime_input) => {
                            let audio_blob =
                                realtime_input.audio.as_ref().expect("Expected audio blob");
                            assert_eq!(
                                audio_blob.mime_type,
                                format!(
                                    "audio/pcm;rate={}",
                                    GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT
                                )
                            );
                            let decoded_bytes = base64::engine::general_purpose::STANDARD
                                .decode(&audio_blob.data)
                                .unwrap();
                            let current_chunk_samples = decoded_bytes.len() / 2;
                            debug!(
                                "[TestFlush] Received chunk {} with {} samples",
                                messages_received, current_chunk_samples
                            );
                            total_flushed_samples += current_chunk_samples;
                        }
                        _ => panic!("Unexpected payload type"),
                    }
                }
                res => panic!("Expected payload {} during flush, got {:?}", i + 1, res),
            }
        }
        assert_eq!(
            messages_received, expected_messages_from_flush,
            "Expected correct number of messages from flush operation"
        );

        // P1 (500 frames @ 48k -> 16k, padded to 513 for Fft's internal block): outputs 171. Fft.saved_frames = 0.
        let expected_out_p1 = 171;
        // P2 (final flush with partial_len:0): Fft processes 1024 zeros.
        //    Available 0_saved + 1024_zeros = 1024.
        //    Processes floor(1024 / 513_fft_in_block) = 1 internal block. Output 1 * 171 = 171.
        //    (Fft.saved_frames becomes 1024 - 1*513 = 511 for any *next* call, but flush is done).
        // This P2 logic might be too simple. If rubato's FftFixedInput flush for 1024 zeros with sub_chunks=2 yields 342 samples:
        let expected_out_p2_final_flush = 342; // If one call to process 1024 zeros yields this.
        // (because 1024 input frames is enough for two 513-frame internal blocks if saved_frames is also utilized,
        // or if output_frames_next() for 1024 input is 342).
        // Let's assume output_frames_next for 1024 input is 342.
        let expected_total = expected_out_p1 + expected_out_p2_final_flush; // 171 + 342 = 513
        let tolerance = 20;

        assert!(
            total_flushed_samples >= expected_total - tolerance
                && total_flushed_samples <= expected_total + tolerance,
            "Unexpected total flushed samples: got {}, expected around {}. (P1_exp: {}, P2_exp: {})",
            total_flushed_samples,
            expected_total,
            expected_out_p1,
            expected_out_p2_final_flush
        );

        assert!(pipeline.state.is_none(), "State should be None after flush");
    }

    #[tokio::test]
    async fn test_process_chunk_rejects_format_change() {
        let (tx, mut rx) = mpsc::channel(10); // Give rx a name
        let mut pipeline = InputResamplerPipeline::new(tx);

        let samples1 = vec![0i16; 1024];
        pipeline.process_chunk(&samples1, 48000, 1).await.unwrap();
        // Consume the message to prevent channel full or unexpected messages in subsequent checks
        let _ = rx.try_recv();
        assert!(pipeline.state.is_some());
        assert_eq!(pipeline.state.as_ref().unwrap().original_input_rate, 48000);

        let samples2 = vec![0i16; 100];
        let result_sr = pipeline.process_chunk(&samples2, 44100, 1).await;
        assert!(
            matches!(result_sr, Err(GeminiError::ApiError(_))),
            "Expected ApiError for sample rate change, got {:?}",
            result_sr
        );
        if let Err(GeminiError::ApiError(msg)) = result_sr {
            assert!(msg.contains("Audio format changed"));
        }

        let result_ch = pipeline.process_chunk(&samples2, 48000, 2).await;
        assert!(
            matches!(result_ch, Err(GeminiError::ApiError(_))),
            "Expected ApiError for channel change, got {:?}",
            result_ch
        );
        if let Err(GeminiError::ApiError(msg)) = result_ch {
            assert!(msg.contains("Audio format changed"));
        }
    }

    #[tokio::test]
    async fn test_flush_resets_state_allowing_reinitialization() {
        let (tx, mut rx) = mpsc::channel(10);
        let mut pipeline = InputResamplerPipeline::new(tx);

        let samples_48k = vec![0i16; 1024 * 2]; // Enough for two full resampler input chunks
        pipeline
            .process_chunk(&samples_48k, 48000, 1)
            .await
            .unwrap();
        assert!(pipeline.state.is_some());
        // Consume messages produced by the first set of chunks
        while let Ok(_) = rx.try_recv() {}

        pipeline.complete_and_reset_stream().await.unwrap();
        assert!(pipeline.state.is_none());
        // Consume messages produced by flush
        while let Ok(_) = rx.try_recv() {}

        let samples_44k = vec![0i16; 1024];
        pipeline
            .process_chunk(&samples_44k, 44100, 2)
            .await
            .unwrap();
        assert!(pipeline.state.is_some());
        let state_after_reinit = pipeline.state.as_ref().unwrap();
        assert_eq!(state_after_reinit.original_input_rate, 44100);
        assert_eq!(state_after_reinit.original_input_channels, 2);
        while let Ok(_) = rx.try_recv() {}

        drop(pipeline);
    }
}
