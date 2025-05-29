pub mod builder;
pub mod handle;
pub mod handlers;

#[cfg(feature = "audio-resampling")]
pub(crate) mod audio_input_pipeline;

mod connection;

pub use builder::GeminiLiveClientBuilder;
pub use handle::GeminiLiveClient;
pub use handlers::{ServerContentContext, ToolHandler, UsageMetadataContext};

/// Sample rate (16kHz) Gemini accepts for audio sent to the Gemini API.
pub(crate) const GEMINI_AUDIO_SAMPLE_RATE_HZ_ACCEPTED_INPUT: u32 = 16000;
/// Number of audio channels (mono) Gemini accepts for audio sent to the Gemini API.
pub(crate) const GEMINI_AUDIO_CHANNELS_ACCEPTED_INPUT: u16 = 1;
