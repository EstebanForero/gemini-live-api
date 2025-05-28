// examples/continuous_audio_chat.rs
use base64::Engine as _;
use cpal::{
    SampleFormat, SampleRate, StreamConfig, SupportedStreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use crossbeam_channel::{Receiver, Sender, bounded};
use gemini_live_api::GeminiError;
use gemini_live_api::{
    GeminiLiveClient, GeminiLiveClientBuilder,
    client::{ServerContentContext, UsageMetadataContext},
    types::*,
};
use std::{
    env,
    sync::{Arc, Mutex as StdMutex},
    time::Duration,
};
use tokio::sync::{Mutex as TokioMutex, Notify};

// ---- Rubato and AudioAdapter imports for playback resampling ----
use audioadapter::{
    // Aliasing to avoid conflict with gemini_live_api::Adapter if it exists
    Adapter as AudioAdapterTrait,
    AdapterMut as AudioAdapterMutTrait,
    SizeError as AudioAdapterSizeError,
    direct::SequentialSliceOfVecs as RubatoSequentialSlice,
};
use rubato::{Fft as RubatoFftResampler, FixedSync, Indexing as RubatoIndexing, Resampler};
// ---- End Rubato imports ----

use tracing::{debug, error, info, warn};

// Playback Resampler State
struct PlaybackResamplerState {
    resampler: RubatoFftResampler<f32>,
    gemini_audio_buffer_f32: Vec<f32>,
    resampler_output_buffer_alloc: Vec<Vec<f32>>,
    target_playback_channels: u16,
}

// Manual Debug implementation for PlaybackResamplerState
impl std::fmt::Debug for PlaybackResamplerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaybackResamplerState")
            .field(
                "gemini_audio_buffer_f32_len",
                &self.gemini_audio_buffer_f32.len(),
            )
            .field(
                "resampler_output_buffer_alloc_channels",
                &self.resampler_output_buffer_alloc.len(),
            )
            .field(
                "resampler_output_buffer_alloc_frames_per_channel_cap",
                &self
                    .resampler_output_buffer_alloc
                    .get(0)
                    .map_or(0, |v| v.capacity()),
            )
            .field("target_playback_channels", &self.target_playback_channels)
            .field("resampler", &"<RubatoFftResampler<f32> instance>")
            .finish()
    }
}

#[derive(Clone, Debug)]
struct ContinuousAudioAppState {
    full_response_text: Arc<StdMutex<String>>,
    model_turn_complete_signal: Arc<Notify>,
    playback_sender: Arc<Sender<Vec<i16>>>,
    is_microphone_active: Arc<StdMutex<bool>>,
    playback_resampler: Arc<TokioMutex<Option<PlaybackResamplerState>>>,
    actual_cpal_output_sample_rate: Arc<StdMutex<u32>>,
    actual_cpal_output_channels: Arc<StdMutex<u16>>,
}

const PREFERRED_CPAL_INPUT_SAMPLE_RATE_HZ: u32 = 48000;
const PREFERRED_CPAL_INPUT_CHANNELS_COUNT: u16 = 1;

const TARGET_OUTPUT_SAMPLE_RATE_HZ: u32 = 24000;
const TARGET_OUTPUT_CHANNELS_COUNT: u16 = 1;

const GEMINI_AUDIO_OUTPUT_RATE: u32 = 24000;
const GEMINI_AUDIO_OUTPUT_CHANNELS: u16 = 1;

async fn handle_on_content(ctx: ServerContentContext, app_state: Arc<ContinuousAudioAppState>) {
    if let Some(model_turn) = &ctx.content.model_turn {
        for part in &model_turn.parts {
            if let Some(text) = &part.text {
                info!("[Handler] Model Text Part: {}", text.trim());
                let mut full_res = app_state.full_response_text.lock().unwrap();
                *full_res += text;
                *full_res += " ";
            }
            if let Some(blob) = &part.inline_data {
                if blob.mime_type.starts_with("audio/")
                    && blob.mime_type.contains("pcm")
                    && blob.mime_type.contains("rate=24000")
                {
                    debug!(
                        "[Handler] Received audio blob from Gemini ({} bytes)",
                        blob.data.len()
                    );
                    match base64::engine::general_purpose::STANDARD.decode(&blob.data) {
                        Ok(decoded_bytes) => {
                            if decoded_bytes.len() % 2 != 0 {
                                warn!("[Handler] Decoded audio data has odd number of bytes.");
                                continue;
                            }
                            let samples_i16_from_gemini = decoded_bytes
                                .chunks_exact(2)
                                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                                .collect::<Vec<i16>>();

                            let mut playback_resampler_guard =
                                app_state.playback_resampler.lock().await;
                            if let Some(playback_state) = &mut *playback_resampler_guard {
                                let samples_f32_mono_from_gemini: Vec<f32> =
                                    samples_i16_from_gemini
                                        .iter()
                                        .map(|&s| s as f32 / (i16::MAX as f32 + 1.0))
                                        .collect();

                                playback_state
                                    .gemini_audio_buffer_f32
                                    .extend(samples_f32_mono_from_gemini);

                                loop {
                                    let required_input_frames =
                                        playback_state.resampler.input_frames_next();
                                    if playback_state.gemini_audio_buffer_f32.len()
                                        < required_input_frames
                                        || required_input_frames == 0
                                    {
                                        if required_input_frames > 0
                                            && !playback_state.gemini_audio_buffer_f32.is_empty()
                                        {
                                            debug!(
                                                "[Handler] Playback resampler needs {} frames, have {}. Buffering.",
                                                required_input_frames,
                                                playback_state.gemini_audio_buffer_f32.len()
                                            );
                                        }
                                        break;
                                    }

                                    let input_chunk_f32: Vec<f32> = playback_state
                                        .gemini_audio_buffer_f32
                                        .drain(0..required_input_frames)
                                        .collect();

                                    let rubato_input_data = vec![input_chunk_f32];
                                    let rubato_input_adapter = RubatoSequentialSlice::new(
                                        &rubato_input_data,
                                        GEMINI_AUDIO_OUTPUT_CHANNELS as usize,
                                        required_input_frames,
                                    )
                                    .expect("Failed to create playback resampler input adapter");

                                    // In handle_on_content(), inside the loop for processing playback_state.gemini_audio_buffer_f32

                                    // ... (input adapter setup is fine) ...

                                    let estimated_output_frames =
                                        playback_state.resampler.output_frames_next();

                                    // The resampler_output_buffer_alloc is now for MONO output from the resampler
                                    // It should have 1 inner Vec.
                                    if playback_state.resampler_output_buffer_alloc.len()
                                        != GEMINI_AUDIO_OUTPUT_CHANNELS as usize
                                    {
                                        // This case should ideally not happen if initialized correctly
                                        warn!(
                                            "[Handler] Playback resampler output buffer has wrong channel count, re-initializing for mono."
                                        );
                                        playback_state.resampler_output_buffer_alloc = vec![
                                                vec![0.0f32; estimated_output_frames.max(1)];
                                                GEMINI_AUDIO_OUTPUT_CHANNELS as usize
                                            ];
                                    } else {
                                        for chan_buf in
                                            playback_state.resampler_output_buffer_alloc.iter_mut()
                                        {
                                            chan_buf.resize(estimated_output_frames.max(1), 0.0f32);
                                        }
                                    }

                                    let mut rubato_output_adapter = RubatoSequentialSlice::new_mut(
                                        &mut playback_state.resampler_output_buffer_alloc,
                                        GEMINI_AUDIO_OUTPUT_CHANNELS as usize, // Resampler outputs MONO
                                        estimated_output_frames.max(1),
                                    ).expect("Failed to create playback resampler output adapter for MONO");

                                    match playback_state.resampler.process_into_buffer(
                                        &rubato_input_adapter,
                                        &mut rubato_output_adapter,
                                        None,
                                    ) {
                                        Ok((_read, written)) => {
                                            if written > 0 {
                                                // resampler_output_buffer_alloc[0] now contains 'written' frames of MONO f32 audio
                                                // at the target_playback_rate.
                                                let mono_resampled_audio_f32 = &playback_state
                                                    .resampler_output_buffer_alloc[0][..written];

                                                let mut final_i16_for_playback: Vec<i16> =
                                                    Vec::with_capacity(
                                                        written
                                                            * playback_state
                                                                .target_playback_channels
                                                                as usize,
                                                    );

                                                for mono_sample_f32 in
                                                    mono_resampled_audio_f32.iter()
                                                {
                                                    let sample_i16 = (*mono_sample_f32
                                                        * (i16::MAX as f32 + 1.0))
                                                        .clamp(i16::MIN as f32, i16::MAX as f32)
                                                        .round()
                                                        as i16;
                                                    final_i16_for_playback.push(sample_i16); // Push L
                                                    if playback_state.target_playback_channels == 2
                                                    {
                                                        final_i16_for_playback.push(sample_i16); // Push R (duplicate mono)
                                                    }
                                                }

                                                if let Err(e) = app_state
                                                    .playback_sender
                                                    .send(final_i16_for_playback)
                                                {
                                                    error!(
                                                        "[Handler] Failed to send resampled/upmixed audio for playback: {}",
                                                        e
                                                    );
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            error!("[Handler] Playback resampling error: {}", e)
                                        }
                                    }

                                    if playback_state.gemini_audio_buffer_f32.is_empty()
                                        && playback_state.resampler.input_frames_next() == 0
                                    {
                                        break;
                                    }
                                }
                            } else {
                                if let Err(e) =
                                    app_state.playback_sender.send(samples_i16_from_gemini)
                                {
                                    error!(
                                        "[Handler] Failed to send direct audio for playback: {}",
                                        e
                                    );
                                }
                            }
                        }
                        Err(e) => error!("[Handler] Failed to decode base64 audio: {}", e),
                    }
                }
            }
        }
    }
    if let Some(transcription) = &ctx.content.output_transcription {
        info!("[Handler] Transcription: {}", transcription.text);
    }
    if ctx.content.turn_complete {
        info!("[Handler] Model turn_complete.");
        app_state.model_turn_complete_signal.notify_one();
    }
    if ctx.content.generation_complete {
        info!("[Handler] Model generation_complete.");
    }
    if ctx.content.interrupted {
        info!("[Handler] Model interrupted.");
    }
}

async fn handle_usage_metadata(
    _ctx: UsageMetadataContext,
    _app_state: Arc<ContinuousAudioAppState>,
) {
    debug!("[Handler] Usage Metadata: {:?}", _ctx.metadata);
}

fn find_supported_config_generic<F, I>(
    mut configs_iterator_fn: F,
    target_sample_rate: u32,
    target_channels: u16,
) -> Result<SupportedStreamConfig, anyhow::Error>
where
    F: FnMut() -> Result<I, cpal::SupportedStreamConfigsError>,
    I: Iterator<Item = cpal::SupportedStreamConfigRange>,
{
    let mut best_config: Option<SupportedStreamConfig> = None;
    let mut min_rate_diff = u32::MAX;

    for config_range in configs_iterator_fn()? {
        if config_range.channels() != target_channels {
            continue;
        }
        if config_range.sample_format() != SampleFormat::I16 {
            continue;
        }
        let current_min_rate = config_range.min_sample_rate().0;
        let current_max_rate = config_range.max_sample_rate().0;
        let rate_to_check =
            if target_sample_rate >= current_min_rate && target_sample_rate <= current_max_rate {
                target_sample_rate
            } else if target_sample_rate < current_min_rate {
                current_min_rate
            } else {
                current_max_rate
            };
        let rate_diff = (rate_to_check as i32 - target_sample_rate as i32).abs() as u32;
        if best_config.is_none() || rate_diff < min_rate_diff {
            min_rate_diff = rate_diff;
            best_config = Some(config_range.with_sample_rate(SampleRate(rate_to_check)));
        }
        if rate_diff == 0 {
            break;
        }
    }
    best_config.ok_or_else(|| {
        anyhow::anyhow!(
            "No i16 config for ~{}Hz {}ch",
            target_sample_rate,
            target_channels
        )
    })
}

#[derive(Clone, Debug)] // This Debug derive should be fine now
struct AudioInputCallbackData {
    audio_chunk_sender: tokio::sync::mpsc::Sender<Vec<i16>>,
    app_state: Arc<ContinuousAudioAppState>,
}

fn setup_audio_input(
    audio_chunk_sender: tokio::sync::mpsc::Sender<Vec<i16>>,
    app_state: Arc<ContinuousAudioAppState>,
) -> Result<(cpal::Stream, u32, u16), anyhow::Error> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device"))?;
    info!("[AudioInput] Using input: {}", device.name()?);

    let supported_config = find_supported_config_generic(
        || device.supported_input_configs(),
        PREFERRED_CPAL_INPUT_SAMPLE_RATE_HZ,
        PREFERRED_CPAL_INPUT_CHANNELS_COUNT,
    )
    .or_else(|e| {
        warn!(
            "Could not find preferred input config ({}Hz {}ch i16): {}. Trying 44.1kHz mono.",
            PREFERRED_CPAL_INPUT_SAMPLE_RATE_HZ, PREFERRED_CPAL_INPUT_CHANNELS_COUNT, e
        );
        find_supported_config_generic(|| device.supported_input_configs(), 44100, 1)
    })
    .or_else(|e| {
        warn!(
            "Could not find 44.1kHz mono input: {}. Trying 48kHz stereo.",
            e
        );
        find_supported_config_generic(|| device.supported_input_configs(), 48000, 2)
    })
    .or_else(|e| {
        warn!(
            "Could not find 48kHz stereo input: {}. Trying any available i16 config.",
            e
        );
        device
            .supported_input_configs()?
            .find(|c| c.sample_format() == SampleFormat::I16)
            .map(|c| c.with_max_sample_rate())
            .ok_or_else(|| anyhow::anyhow!("No i16 input config found"))
    })?;

    let config: StreamConfig = supported_config.clone().into();
    let actual_input_sample_rate = config.sample_rate.0;
    let actual_input_channels = config.channels;

    info!(
        "[AudioInput] CPAL selected input: {} Hz, {} ch, {:?}",
        actual_input_sample_rate,
        actual_input_channels,
        supported_config.sample_format()
    );

    let callback_data = AudioInputCallbackData {
        audio_chunk_sender,
        app_state,
    };

    let stream = device.build_input_stream(
        &config,
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            if !*callback_data.app_state.is_microphone_active.lock().unwrap() { return; }
            if data.is_empty() { return; }
            match callback_data.audio_chunk_sender.try_send(data.to_vec()) {
                Ok(_) => {}
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => { /* warn!("[AudioInput] Chunk channel full.") */ }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => { error!("[AudioInput] Chunk channel closed.") }
            }
        },
        |err| error!("[AudioInput] CPAL Error: {}", err),
        None,
    )?;
    stream.play()?;
    Ok((stream, actual_input_sample_rate, actual_input_channels))
}

fn setup_audio_output(
    playback_receiver: Receiver<Vec<i16>>,
    app_state: Arc<ContinuousAudioAppState>,
) -> Result<cpal::Stream, anyhow::Error> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("No output device"))?;
    info!("[AudioOutput] Using output: {}", device.name()?);

    let supported_config = find_supported_config_generic(
        || device.supported_output_configs(),
        TARGET_OUTPUT_SAMPLE_RATE_HZ,
        TARGET_OUTPUT_CHANNELS_COUNT,
    )
    .or_else(|e| {
        warn!(
            "Could not find target output config ({}Hz {}ch i16): {}. Trying 48kHz stereo.",
            TARGET_OUTPUT_SAMPLE_RATE_HZ, TARGET_OUTPUT_CHANNELS_COUNT, e
        );
        find_supported_config_generic(|| device.supported_output_configs(), 48000, 2)
    })
    .or_else(|e| {
        warn!(
            "Could not find 48kHz stereo output: {}. Trying 44.1kHz stereo.",
            e
        );
        find_supported_config_generic(|| device.supported_output_configs(), 44100, 2)
    })
    .or_else(|e| {
        warn!(
            "Could not find 44.1kHz stereo output: {}. Trying 48kHz mono.",
            e
        );
        find_supported_config_generic(|| device.supported_output_configs(), 48000, 1)
    })
    .or_else(|e| {
        warn!(
            "Could not find 48kHz mono output: {}. Trying any available i16 config.",
            e
        );
        device
            .supported_output_configs()?
            .find(|c| c.sample_format() == SampleFormat::I16)
            .map(|c| c.with_max_sample_rate())
            .ok_or_else(|| anyhow::anyhow!("No i16 output config found for playback device"))
    })?;

    let config: StreamConfig = supported_config.clone().into();
    let actual_playback_rate = config.sample_rate.0;
    let actual_playback_channels = config.channels;

    *app_state.actual_cpal_output_sample_rate.lock().unwrap() = actual_playback_rate;
    *app_state.actual_cpal_output_channels.lock().unwrap() = actual_playback_channels;

    info!(
        "[AudioOutput] CPAL selected output: {} Hz, {} ch, {:?}",
        actual_playback_rate,
        actual_playback_channels,
        supported_config.sample_format()
    );

    let mut samples_buffer: Vec<i16> = Vec::new();
    let stream = device.build_output_stream(
        &config,
        move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
            let mut new_samples_received_this_callback = 0;
            while samples_buffer.len() < data.len() {
                if let Ok(new_samples) = playback_receiver.try_recv() {
                    new_samples_received_this_callback += new_samples.len();
                    samples_buffer.extend(new_samples);
                } else { break; }
            }
            let len_to_write = std::cmp::min(data.len(), samples_buffer.len());
            if len_to_write > 0 && new_samples_received_this_callback > 0 {
                debug!("[AudioOutputCallback] Writing {} samples. Samples in internal buffer before write: {}. Received new this call: {}.",
                       len_to_write, samples_buffer.len(), new_samples_received_this_callback);
            }
            for i in 0..len_to_write { data[i] = samples_buffer.remove(0); }
            for sample_idx in len_to_write..data.len() { data[sample_idx] = 0; }
        },
        |err| error!("[AudioOutput] CPAL Error: {}", err),
        None,
    )?;
    stream.play()?;
    Ok(stream)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    info!("App start. Logger initialized.");

    dotenv::dotenv().ok();
    let api_key = env::var("GEMINI_API_KEY").map_err(|_| "GEMINI_API_KEY not set")?;
    let model_name = env::var("GEMINI_MODEL")
        .unwrap_or_else(|_| "models/gemini-2.5-flash-preview-native-audio-dialog".to_string());

    let (playback_tx, playback_rx) = bounded::<Vec<i16>>(100);
    let (audio_input_chunk_tx, mut audio_input_chunk_rx) =
        tokio::sync::mpsc::channel::<Vec<i16>>(20);

    let app_state_instance = Arc::new(ContinuousAudioAppState {
        full_response_text: Arc::new(StdMutex::new(String::new())),
        model_turn_complete_signal: Arc::new(Notify::new()),
        playback_sender: Arc::new(playback_tx.clone()), // Clone for final flush logic
        is_microphone_active: Arc::new(StdMutex::new(false)),
        playback_resampler: Arc::new(TokioMutex::new(None)),
        actual_cpal_output_sample_rate: Arc::new(StdMutex::new(0)),
        actual_cpal_output_channels: Arc::new(StdMutex::new(0)),
    });

    let _output_stream = setup_audio_output(playback_rx, app_state_instance.clone())?;
    let actual_playback_rate = *app_state_instance
        .actual_cpal_output_sample_rate
        .lock()
        .unwrap();
    let actual_playback_channels = *app_state_instance
        .actual_cpal_output_channels
        .lock()
        .unwrap();

    if actual_playback_rate == 0 || actual_playback_channels == 0 {
        return Err(anyhow::anyhow!("Failed to get actual CPAL output configuration").into());
    }
    info!(
        "[Main] Actual CPAL output: {} Hz, {} ch",
        actual_playback_rate, actual_playback_channels
    );

    if GEMINI_AUDIO_OUTPUT_RATE != actual_playback_rate
        || GEMINI_AUDIO_OUTPUT_CHANNELS != actual_playback_channels
    {
        info!(
            "[Main] Mismatch: Gemini audio ({}Hz {}ch) vs Playback ({}Hz {}ch). Initializing playback resampler.",
            GEMINI_AUDIO_OUTPUT_RATE,
            GEMINI_AUDIO_OUTPUT_CHANNELS,
            actual_playback_rate,
            actual_playback_channels
        );
        let playback_rubato_resampler = RubatoFftResampler::<f32>::new(
            GEMINI_AUDIO_OUTPUT_RATE as usize, // Input rate from Gemini (24kHz)
            actual_playback_rate as usize,     // Target output rate for CPAL (e.g., 48kHz)
            1024,                              // Chunk size for FFT
            2,                                 // Sub-chunks
            GEMINI_AUDIO_OUTPUT_CHANNELS as usize, // Number of channels this resampler processes (1 for mono)
            FixedSync::Input,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create playback FFT resampler: {}", e))?;

        let max_out_frames = playback_rubato_resampler.output_frames_max();
        // This buffer is for the *direct output* of the resampler, which is MONO
        let initial_resampler_output_buffer_mono =
            vec![vec![0.0f32; max_out_frames.max(1)]; GEMINI_AUDIO_OUTPUT_CHANNELS as usize];

        let mut playback_resampler_guard = app_state_instance.playback_resampler.lock().await;
        *playback_resampler_guard = Some(PlaybackResamplerState {
            resampler: playback_rubato_resampler,
            gemini_audio_buffer_f32: Vec::with_capacity(1024 * 5),
            resampler_output_buffer_alloc: initial_resampler_output_buffer_mono, // Use the mono buffer here
            target_playback_channels: actual_playback_channels, // Store the final target for CPAL
        });
        info!("[Main] Playback resampler (mono output) initialized.");
    } else {
        info!("[Main] Gemini audio matches playback device config. No playback resampling needed.");
    }

    let mut builder = GeminiLiveClientBuilder::<ContinuousAudioAppState>::new_with_state(
        api_key,
        model_name.clone(),
        (*app_state_instance).clone(),
    );
    builder = builder.generation_config(GenerationConfig {
        response_modalities: Some(vec![ResponseModality::Audio]),
        temperature: Some(0.7),
        speech_config: Some(SpeechConfig {
            language_code: Some(SpeechLanguageCode::EnglishUS),
        }),
        ..Default::default()
    });
    builder = builder.realtime_input_config(RealtimeInputConfig {
        automatic_activity_detection: Some(AutomaticActivityDetection {
            disabled: Some(false),
            start_of_speech_sensitivity: Some(StartSensitivity::StartSensitivityLow),
            prefix_padding_ms: Some(50),
            end_of_speech_sensitivity: Some(EndSensitivity::EndSensitivityLow),
            silence_duration_ms: Some(800),
            ..Default::default()
        }),
        activity_handling: Some(ActivityHandling::StartOfActivityInterrupts),
        turn_coverage: Some(TurnCoverage::TurnIncludesOnlyActivity),
    });
    builder = builder.output_audio_transcription(AudioTranscriptionConfig {});
    builder = builder.system_instruction(Content {
        parts: vec![Part {
            text: Some("You are a helpful voice assistant.".to_string()),
            ..Default::default()
        }],
        role: Some(Role::System),
        ..Default::default()
    });
    builder = builder.on_server_content(handle_on_content);
    builder = builder.on_usage_metadata(handle_usage_metadata);
    #[cfg(feature = "audio-resampling")]
    {
        builder = builder.enable_automatic_resampling();
    }

    info!("[Main] Connecting to Gemini model: {}", model_name);
    let mut client = builder.connect().await?;
    let client_clone_for_task = client.clone();

    let (_input_stream, actual_cpal_sample_rate, actual_cpal_channels) =
        setup_audio_input(audio_input_chunk_tx, app_state_instance.clone())?;

    tokio::spawn(async move {
        info!(
            "[AudioProcessingTask] Started. CPAL input: {}Hz {}ch.",
            actual_cpal_sample_rate, actual_cpal_channels
        );
        while let Some(samples_vec) = audio_input_chunk_rx.recv().await {
            if samples_vec.is_empty() {
                continue;
            }
            if let Err(e) = client_clone_for_task
                .send_audio_chunk(&samples_vec, actual_cpal_sample_rate, actual_cpal_channels)
                .await
            {
                error!("[AudioProcessingTask] send_audio_chunk error: {:?}", e);
                if matches!(e, GeminiError::ApiError(ref msg) if msg.contains("Audio format changed"))
                {
                    break;
                }
            }
        }
        info!("[AudioProcessingTask] Stopped.");
    });

    *app_state_instance.is_microphone_active.lock().unwrap() = true;
    info!("[Main] Mic active. Continuous chat started. Ctrl+C to exit.");

    loop {
        tokio::select! {
            _ = app_state_instance.model_turn_complete_signal.notified() => {
                info!("[MainLoop] Model turn complete signal.");
            }
            _ = tokio::signal::ctrl_c() => {
                info!("[MainLoop] Ctrl+C. Shutting down...");
                break;
            }
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
    }

    *app_state_instance.is_microphone_active.lock().unwrap() = false;
    info!("[Main] Sending audioStreamEnd to client...");
    if let Err(e) = client.send_audio_stream_end().await {
        warn!("[Main] Failed to send audioStreamEnd via client: {:?}", e);
    }

    let final_flush_playback_sender = app_state_instance.playback_sender.clone();
    let mut playback_resampler_guard = app_state_instance.playback_resampler.lock().await;
    if let Some(playback_state) = &mut *playback_resampler_guard {
        info!("[Main] Flushing playback resampler...");
        let remaining_buffered_f32 = playback_state
            .gemini_audio_buffer_f32
            .drain(..)
            .collect::<Vec<_>>();
        if !remaining_buffered_f32.is_empty() {
            let rubato_input_data = vec![remaining_buffered_f32];
            let rubato_input_adapter = RubatoSequentialSlice::new(
                &rubato_input_data,
                GEMINI_AUDIO_OUTPUT_CHANNELS as usize,
                rubato_input_data[0].len(),
            )
            .expect("Flush: Failed to create playback resampler input adapter");

            let estimated_output_frames = playback_state.resampler.output_frames_next();
            for chan_buf in playback_state.resampler_output_buffer_alloc.iter_mut() {
                chan_buf.resize(estimated_output_frames.max(1), 0.0f32);
            }
            let mut rubato_output_adapter = RubatoSequentialSlice::new_mut(
                &mut playback_state.resampler_output_buffer_alloc,
                playback_state.target_playback_channels as usize,
                estimated_output_frames.max(1),
            )
            .expect("Flush: Failed to create playback resampler output adapter");

            let indexing = RubatoIndexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: Some(rubato_input_data[0].len()),
                active_channels_mask: None,
            };

            match playback_state.resampler.process_into_buffer(
                &rubato_input_adapter,
                &mut rubato_output_adapter,
                Some(&indexing),
            ) {
                Ok((_read, written)) => {
                    if written > 0 {
                        let mut final_i16_for_playback: Vec<i16> = Vec::with_capacity(
                            written * playback_state.target_playback_channels as usize,
                        );
                        for frame_idx in 0..written {
                            for chan_idx in 0..playback_state.target_playback_channels as usize {
                                let sample_f32 = playback_state.resampler_output_buffer_alloc
                                    [chan_idx][frame_idx];
                                let sample_i16 = (sample_f32 * (i16::MAX as f32 + 1.0))
                                    .clamp(i16::MIN as f32, i16::MAX as f32)
                                    .round()
                                    as i16;
                                final_i16_for_playback.push(sample_i16);
                            }
                        }
                        if let Err(e) = final_flush_playback_sender.send(final_i16_for_playback) {
                            error!(
                                "[MainFlush] Failed to send resampled audio for playback: {}",
                                e
                            );
                        }
                    }
                }
                Err(e) => error!("[MainFlush] Playback resampling error: {}", e),
            }
        }
        let empty_input_data_storage_ch_f: Vec<f32> = vec![];
        let empty_input_data_storage_f: Vec<Vec<f32>> = vec![empty_input_data_storage_ch_f];
        let empty_input_adapter_f = RubatoSequentialSlice::new(
            &empty_input_data_storage_f,
            GEMINI_AUDIO_OUTPUT_CHANNELS as usize,
            0,
        )
        .expect("Flush: Failed to create empty input adapter for playback resampler flush");

        let output_buffer_len_flush_f = playback_state.resampler.output_frames_next().max(1);
        for chan_buf in playback_state.resampler_output_buffer_alloc.iter_mut() {
            chan_buf.resize(output_buffer_len_flush_f, 0.0f32);
        }
        let mut output_adapter_flush_f = RubatoSequentialSlice::new_mut(
            &mut playback_state.resampler_output_buffer_alloc,
            playback_state.target_playback_channels as usize,
            output_buffer_len_flush_f,
        )
        .expect("Flush: Failed to create playback resampler output adapter for flush pass");

        let indexing_flush_f = RubatoIndexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: Some(0),
            active_channels_mask: None,
        };
        match playback_state.resampler.process_into_buffer(
            &empty_input_adapter_f,
            &mut output_adapter_flush_f,
            Some(&indexing_flush_f),
        ) {
            Ok((_read, written)) => {
                if written > 0 {
                    let mut final_i16_for_playback: Vec<i16> = Vec::with_capacity(
                        written * playback_state.target_playback_channels as usize,
                    );
                    for frame_idx in 0..written {
                        for chan_idx in 0..playback_state.target_playback_channels as usize {
                            let sample_f32 =
                                playback_state.resampler_output_buffer_alloc[chan_idx][frame_idx];
                            let sample_i16 = (sample_f32 * (i16::MAX as f32 + 1.0))
                                .clamp(i16::MIN as f32, i16::MAX as f32)
                                .round() as i16;
                            final_i16_for_playback.push(sample_i16);
                        }
                    }
                    if let Err(e) = final_flush_playback_sender.send(final_i16_for_playback) {
                        error!(
                            "[MainFlush] Failed to send final flushed audio for playback: {}",
                            e
                        );
                    }
                }
            }
            Err(e) => error!("[MainFlush] Playback resampler internal flush error: {}", e),
        }
    }
    drop(playback_resampler_guard);

    tokio::time::sleep(Duration::from_millis(500)).await;

    client.close().await?;
    info!("[Main] Client closed. Exiting.");
    Ok(())
}
