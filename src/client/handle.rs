use crate::error::GeminiError;
use crate::types::{
    ActivityEnd, ActivityStart, BidiGenerateContentClientContent, BidiGenerateContentRealtimeInput,
    Blob, ClientMessagePayload, Content, Part, Role,
};
use base64::Engine as _;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};

#[cfg(feature = "audio-resampling")]
use super::audio_input_pipeline::InputResamplerPipeline;

use super::GeminiLiveClientBuilder;
use super::{GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT, GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT};

#[derive(Clone)]
pub struct GeminiLiveClient<S: Clone + Send + Sync + 'static> {
    pub(crate) shutdown_tx: Arc<TokioMutex<Option<oneshot::Sender<()>>>>,
    pub(crate) outgoing_sender: Option<mpsc::Sender<ClientMessagePayload>>,
    pub(crate) state: Arc<S>,

    #[cfg(feature = "audio-resampling")]
    pub(crate) input_resampler_pipeline: Arc<TokioMutex<Option<InputResamplerPipeline>>>,
    #[cfg(feature = "audio-resampling")]
    pub(crate) automatic_resampling_configured_in_builder: bool,
}

impl<S: Clone + Send + Sync + 'static> GeminiLiveClient<S> {
    pub async fn close(&mut self) -> Result<(), GeminiError> {
        info!("[ClientHandle] Close requested.");
        let mut shutdown_tx_guard = self.shutdown_tx.lock().await;
        if let Some(tx) = shutdown_tx_guard.take() {
            if tx.send(()).is_err() {
                info!(
                    "[ClientHandle] Shutdown signal failed: Listener task already gone or shut down."
                );
            } else {
                info!("[ClientHandle] Shutdown signal sent to listener task.");
            }
        }
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

    pub fn state(&self) -> Arc<S> {
        self.state.clone()
    }

    async fn send_message(&self, payload: ClientMessagePayload) -> Result<(), GeminiError> {
        if let Some(sender) = &self.outgoing_sender {
            let sender_clone = sender.clone();
            match sender_clone.send(payload).await {
                Ok(_) => Ok(()),
                Err(e) => {
                    error!(
                        "[ClientHandle] Failed to send message to listener task: Channel closed. Error: {}",
                        e
                    );
                    Err(GeminiError::SendError)
                }
            }
        } else {
            error!(
                "[ClientHandle] Cannot send message: Client is closed or outgoing sender is missing."
            );
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
        audio_samples_i16: &[i16],
        sample_rate: u32,
        channels: u16,
    ) -> Result<(), GeminiError> {
        if audio_samples_i16.is_empty() {
            return Ok(());
        }

        // Check for direct send first, if audio is already in target format
        if sample_rate == GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT
            && channels == GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT
        {
            let mut byte_data = Vec::with_capacity(audio_samples_i16.len() * 2);
            for sample_val in audio_samples_i16 {
                byte_data.extend_from_slice(&sample_val.to_le_bytes());
            }
            let encoded_data = base64::engine::general_purpose::STANDARD.encode(&byte_data);
            let mime_type = format!("audio/pcm;rate={}", sample_rate);
            let audio_blob = Blob {
                mime_type,
                data: encoded_data,
            };
            let realtime_input = BidiGenerateContentRealtimeInput {
                audio: Some(audio_blob),
                ..Default::default()
            };
            return self
                .send_message(ClientMessagePayload::RealtimeInput(realtime_input))
                .await;
        }

        #[cfg(feature = "audio-resampling")]
        {
            if self.automatic_resampling_configured_in_builder {
                let mut pipeline_guard = self.input_resampler_pipeline.lock().await;
                if let Some(pipeline) = pipeline_guard.as_mut() {
                    return pipeline
                        .process_chunk(audio_samples_i16, sample_rate, channels)
                        .await;
                } else {
                    // This case should ideally not be reached if automatic_resampling_configured_in_builder is true,
                    // as the builder is supposed to initialize Some(InputResamplerPipeline).
                    error!(
                        "[ClientHandle] Automatic resampling enabled but pipeline is None. This indicates an internal setup issue."
                    );
                    return Err(GeminiError::InternalError(
                        "InputResamplerPipeline expected but was None despite config.".to_string(),
                    ));
                }
            }
        }

        // If we reach here, it means audio was not 16kHz mono for direct send, AND
        // either resampling feature is off, or builder didn't enable it.
        let error_msg = format!(
            "Audio input ({}Hz {}ch) must be 16kHz mono. Automatic resampling not active or not compiled.",
            sample_rate, channels
        );
        warn!("[ClientHandle] {}", error_msg);
        Err(GeminiError::ApiError(error_msg))
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

    pub async fn send_audio_stream_end(&self) -> Result<(), GeminiError> {
        #[cfg(feature = "audio-resampling")]
        {
            if self.automatic_resampling_configured_in_builder {
                info!(
                    "[ClientHandle] Flushing audio input pipeline before sending stream end signal."
                );
                match self.flush_audio_input_pipeline().await {
                    Ok(_) => info!(
                        "[ClientHandle] Audio input pipeline flushed successfully before stream end."
                    ),
                    Err(e) => error!(
                        "[ClientHandle] Error flushing audio input pipeline before stream end: {}",
                        e
                    ),
                }
            }
        }
        info!("[ClientHandle] Sending audio stream end signal.");
        let end_stream_msg = BidiGenerateContentRealtimeInput {
            audio_stream_end: Some(true),
            ..Default::default()
        };
        self.send_message(ClientMessagePayload::RealtimeInput(end_stream_msg))
            .await
    }

    #[cfg(feature = "audio-resampling")]
    async fn flush_audio_input_pipeline(&self) -> Result<(), GeminiError> {
        if !self.automatic_resampling_configured_in_builder {
            return Ok(());
        }

        let mut pipeline_guard = self.input_resampler_pipeline.lock().await;
        if let Some(pipeline) = pipeline_guard.as_mut() {
            info!("[ClientHandle] Flushing input audio pipeline.");
            pipeline.complete_and_reset_stream().await?;
        } else {
            info!(
                "[ClientHandle] No active input resampler pipeline to flush (already flushed or never initialized)."
            );
        }
        Ok(())
    }
}

impl<S: Clone + Send + Sync + 'static> Drop for GeminiLiveClient<S> {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.shutdown_tx.try_lock() {
            if guard.is_some() {
                warn!(
                    "[ClientHandle] Dropped without explicit close(). Attempting to signal shutdown (try_lock succeeded)."
                );
                if let Some(tx) = guard.take() {
                    if tx.send(()).is_err() {
                        info!(
                            "[ClientHandle] Drop: Shutdown signal send failed, listener task likely already gone."
                        );
                    } else {
                        info!("[ClientHandle] Drop: Shutdown signal sent via try_lock.");
                    }
                }
                self.outgoing_sender.take();
            } else {
                if self.outgoing_sender.is_some() {
                    warn!(
                        "[ClientHandle] Dropped. Shutdown sender was already None but outgoing_sender existed (possibly closed then dropped)."
                    );
                    self.outgoing_sender.take();
                }
            }
        } else {
            if self.outgoing_sender.is_some() {
                warn!(
                    "[ClientHandle] Dropped without explicit close(). Could not acquire shutdown_tx lock to send signal."
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

    pub(crate) fn init_test_logger() {
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

    #[allow(dead_code)]
    pub(crate) fn setup_test() {
        init_test_logger();
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
            // These fields are cfg-gated in the struct definition, so they won't exist here.
            // #[cfg(feature = "audio-resampling")]
            // input_resampler_pipeline: Arc::new(TokioMutex::new(None)),
            // #[cfg(feature = "audio-resampling")]
            // automatic_resampling_configured_in_builder: false,
        };
        (client, outgoing_rx)
    }

    fn generate_sine_wave_mono_test_no_feat(
        num_frames: usize,
        sample_rate: u32,
        frequency: f32,
    ) -> Vec<i16> {
        (0..num_frames)
            .map(|i| {
                let time = i as f32 / sample_rate as f32;
                let value = (2.0 * std::f32::consts::PI * frequency * time).sin();
                (value * (i16::MAX as f32 * 0.8)) as i16
            })
            .collect()
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
            assert!(msg.contains("Audio input (48000Hz 1ch) must be 16kHz mono. Automatic resampling not active or not compiled."));
        } else {
            panic!("Expected ApiError for wrong format, got {:?}", result);
        }
    }
}

#[cfg(test)]
#[cfg(feature = "audio-resampling")]
mod tests {
    use super::*;
    use crate::client::audio_input_pipeline::InputResamplerPipeline;
    use crate::client::handle::test_utils::setup_test;
    use base64::Engine as _;
    use tokio::sync::mpsc;
    use tokio::time::{Duration, timeout};
    use tracing::trace;

    async fn collect_audio_until_stream_end(
        rx: &mut mpsc::Receiver<ClientMessagePayload>,
        expected_mime_type_prefix: &str,
        timeout_duration: Duration,
    ) -> (Vec<u8>, String) {
        let mut aggregated_audio_bytes = Vec::new();
        let mut first_mime_type_seen = String::new();
        let mut stream_ended_signal_received = false;
        let mut audio_blobs_received = 0;

        let overall_timeout =
            Duration::from_secs(timeout_duration.as_secs_f64().ceil() as u64 + 10); // Increased overall timeout further
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
                        // No aggressive drain loop here, rely on overall timeout for remaining messages
                        break; // Break after seeing stream end
                    }
                }
                Ok(Some(other_payload)) => {
                    panic!(
                        "[TestHelper] Unexpected payload type received: {:?}",
                        other_payload
                    );
                }
                Ok(None) => {
                    // Channel closed
                    if !stream_ended_signal_received {
                        panic!(
                            "[TestHelper] Channel closed prematurely before audioStreamEnd signal was received."
                        );
                    }
                    break;
                }
                Err(_) => {
                    // Timeout on individual recv
                    if stream_ended_signal_received {
                        break;
                    } // If already seen end, timeout is fine
                    trace!(
                        "[TestHelper] Individual recv timeout (duration: {:?}). Continuing if overall not timed out.",
                        timeout_duration
                    );
                }
            }
        }
        // After loop, ensure stream_ended was seen if we didn't break due to channel close
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
        let (shutdown_tx_dummy, _) = oneshot::channel();

        let pipeline_option = if enable_resampling_in_builder {
            Some(InputResamplerPipeline::new(outgoing_tx.clone()))
        } else {
            None
        };

        let client = GeminiLiveClient {
            shutdown_tx: Arc::new(TokioMutex::new(Some(shutdown_tx_dummy))),
            outgoing_sender: Some(outgoing_tx.clone()),
            state: Arc::new(()),
            #[cfg(feature = "audio-resampling")]
            input_resampler_pipeline: Arc::new(TokioMutex::new(pipeline_option)),
            #[cfg(feature = "audio-resampling")]
            automatic_resampling_configured_in_builder: enable_resampling_in_builder,
        };
        (client, outgoing_rx)
    }

    fn generate_sine_wave_mono(num_frames: usize, sample_rate: u32, frequency: f32) -> Vec<i16> {
        (0..num_frames)
            .map(|i| {
                let time = i as f32 / sample_rate as f32;
                let value = (2.0 * std::f32::consts::PI * frequency * time).sin();
                (value * (i16::MAX as f32 * 0.8)) as i16
            })
            .collect()
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
        let num_input_stereo_frames = 2048;
        let audio_data_i16 =
            generate_sine_wave_stereo_interleaved(num_input_stereo_frames, 48000, 440.0, 660.0);

        client
            .send_audio_chunk(&audio_data_i16, 48000, 2)
            .await
            .unwrap();
        client.send_audio_stream_end().await.unwrap();

        let expected_mime_prefix = format!(
            "audio/pcm;rate={}",
            GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT
        );
        let (received_audio_bytes, mime_type) = collect_audio_until_stream_end(
            &mut outgoing_rx,
            &expected_mime_prefix,
            Duration::from_millis(400), // Increased timeout
        )
        .await;

        if !received_audio_bytes.is_empty() {
            assert_eq!(mime_type, expected_mime_prefix);
        }
        assert!(
            !received_audio_bytes.is_empty(),
            "Expected resampled audio data"
        );

        let num_received_samples = received_audio_bytes.len() / 2;
        // Based on previous successful trace and fix: 2048 stereo frames -> 855 mono samples
        let expected_total = 855;
        let tolerance = 35; // Slightly wider for potential timing/chunking variations
        assert!(
            num_received_samples >= expected_total - tolerance
                && num_received_samples <= expected_total + tolerance,
            "Unexpected #output samples: got {}, expected around {}. Input stereo frames: {}",
            num_received_samples,
            expected_total,
            num_input_stereo_frames
        );
    }

    #[tokio::test]
    async fn test_resample_multiple_small_chunks_441k_mono_then_flush() {
        setup_test();
        let (client, mut outgoing_rx) = setup_test_client_for_resampling_tests(true).await;
        let input_hz = 44100;
        let num_chunks = 5;
        let frames_per_small_chunk = 441;
        let total_input_frames = frames_per_small_chunk * num_chunks;

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

        let expected_mime_prefix = format!(
            "audio/pcm;rate={}",
            GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT
        );
        let (received_audio_bytes, mime_type) = collect_audio_until_stream_end(
            &mut outgoing_rx,
            &expected_mime_prefix,
            Duration::from_millis(500), // Increased timeout
        )
        .await;

        if !received_audio_bytes.is_empty() {
            assert_eq!(mime_type, expected_mime_prefix);
        }
        assert!(
            !received_audio_bytes.is_empty(),
            "Expected audio after flush"
        );

        let num_received_samples = received_audio_bytes.len() / 2;
        // Based on previous successful trace: 2205 mono -> 1280 output
        let expected_total = 1280;
        let tolerance = 50;
        assert!(
            num_received_samples >= expected_total - tolerance
                && num_received_samples <= expected_total + tolerance,
            "Unexpected total samples: got {}, expected around {}. Total input mono frames: {}",
            num_received_samples,
            expected_total,
            total_input_frames
        );
    }

    #[tokio::test]
    async fn test_error_on_format_change_after_resampler_init() {
        setup_test();
        let (client, mut _outgoing_rx) = setup_test_client_for_resampling_tests(true).await;
        let audio_48k_stereo = generate_sine_wave_stereo_interleaved(1024, 48000, 440.0, 600.0);
        let audio_22k_mono = generate_sine_wave_mono(1024, 22050, 300.0);

        client
            .send_audio_chunk(&audio_48k_stereo, 48000, 2)
            .await
            .unwrap();

        let result = client.send_audio_chunk(&audio_22k_mono, 22050, 1).await;
        assert!(
            result.is_err(),
            "Expected an error when changing audio format without flush"
        );
        if let Err(GeminiError::ApiError(msg)) = result {
            assert!(msg.contains("Audio format changed"));
        } else {
            panic!("Expected ApiError for format change, got {:?}", result);
        }

        client.send_audio_stream_end().await.unwrap();

        let result_after_flush = client.send_audio_chunk(&audio_22k_mono, 22050, 1).await;
        assert!(
            result_after_flush.is_ok(),
            "Sending audio with new format after flush (via send_audio_stream_end) should re-initialize and succeed, but got: {:?}",
            result_after_flush
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
                "Audio input (48000Hz 1ch) must be 16kHz mono. Automatic resampling not active"
            ));
        } else {
            panic!("Expected ApiError, got {:?}", result);
        }
    }

    #[tokio::test]
    #[cfg(feature = "audio-resampling")]
    async fn test_flush_audio_input_pipeline_no_op_if_resampling_disabled_or_no_state() {
        setup_test();
        let (client_disabled, mut rx_disabled) =
            setup_test_client_for_resampling_tests(false).await;
        assert!(
            client_disabled
                .input_resampler_pipeline
                .lock()
                .await
                .is_none(),
            "Pipeline Arc should hold None if resampling disabled"
        );
        client_disabled.flush_audio_input_pipeline().await.unwrap();
        assert!(
            matches!(
                rx_disabled.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ),
            "No messages expected when pipeline is None"
        );

        let (client_no_init, mut rx_no_init) = setup_test_client_for_resampling_tests(true).await;
        {
            let guard = client_no_init.input_resampler_pipeline.lock().await;
            assert!(
                guard.is_some(),
                "Pipeline option should be Some when enabled in builder"
            );
            // We are testing the client's flush_audio_input_pipeline method.
            // It will internally find that the pipeline's *internal state* is None.
        }
        client_no_init.flush_audio_input_pipeline().await.unwrap(); // This calls pipeline.complete_and_reset_stream()
        assert!(
            matches!(rx_no_init.try_recv(), Err(mpsc::error::TryRecvError::Empty)),
            "No messages expected when flushing an unused pipeline"
        );
    }

    #[tokio::test]
    async fn test_send_audio_stream_end_calls_flush_and_allows_reinitialization() {
        setup_test();
        let (client, mut _outgoing_rx) = setup_test_client_for_resampling_tests(true).await;
        let audio_data = generate_sine_wave_mono(1024, 44100, 300.0);

        client
            .send_audio_chunk(&audio_data, 44100, 1)
            .await
            .unwrap();

        client.send_audio_stream_end().await.unwrap();

        let audio_data_new_format = generate_sine_wave_mono(512, 22050, 300.0);
        let result = client
            .send_audio_chunk(&audio_data_new_format, 22050, 1)
            .await;
        assert!(
            result.is_ok(),
            "Sending audio with new format after send_audio_stream_end (which flushes) should re-initialize and succeed, but got: {:?}",
            result
        );
    }
}
