#[cfg(feature = "audio-resampling")]
use super::handle::ActiveResamplerState;
use super::handle::GeminiLiveClient;
use super::handlers::{
    EventHandlerSimple, Handlers, ServerContentContext, ToolHandler, UsageMetadataContext,
};
use crate::error::GeminiError;
use crate::types::*;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::{mpsc, oneshot};

pub struct GeminiLiveClientBuilder<S: Clone + Send + Sync + 'static> {
    pub(crate) api_key: String,
    pub(crate) initial_setup: BidiGenerateContentSetup,
    pub(crate) handlers: Handlers<S>,
    pub(crate) state: S,

    #[cfg(feature = "audio-resampling")]
    pub(crate) automatic_resampling_enabled: bool,
}

impl<S: Clone + Send + Sync + 'static + Default> GeminiLiveClientBuilder<S> {
    pub fn new(api_key: String, model: String) -> Self {
        Self::new_with_state(api_key, model, S::default())
    }
}

impl<S: Clone + Send + Sync + 'static> GeminiLiveClientBuilder<S> {
    pub fn new_with_state(api_key: String, model: String, state: S) -> Self {
        Self {
            api_key,
            initial_setup: BidiGenerateContentSetup {
                model,
                ..Default::default()
            },
            handlers: Handlers::default(),
            state,
            #[cfg(feature = "audio-resampling")]
            automatic_resampling_enabled: false,
        }
    }

    pub fn generation_config(mut self, config: GenerationConfig) -> Self {
        self.initial_setup.generation_config = Some(config);
        self
    }

    pub fn system_instruction(mut self, instruction: Content) -> Self {
        self.initial_setup.system_instruction = Some(instruction);
        self
    }

    #[doc(hidden)]
    pub fn add_tool_declaration(mut self, declaration: FunctionDeclaration) -> Self {
        let tools_vec = self.initial_setup.tools.get_or_insert_with(Vec::new);
        if let Some(tool_struct) = tools_vec.first_mut() {
            tool_struct.function_declarations.push(declaration);
        } else {
            tools_vec.push(Tool {
                function_declarations: vec![declaration],
            });
        }
        self
    }

    #[doc(hidden)]
    pub fn on_tool_call<F>(mut self, tool_name: impl Into<String>, handler: F) -> Self
    where
        F: ToolHandler<S> + 'static,
    {
        self.handlers
            .tool_handlers
            .insert(tool_name.into(), Arc::new(handler));
        self
    }

    pub fn on_server_content(
        mut self,
        handler: impl EventHandlerSimple<ServerContentContext, S> + 'static,
    ) -> Self {
        self.handlers.on_server_content = Some(Arc::new(handler));
        self
    }

    pub fn on_usage_metadata(
        mut self,
        handler: impl EventHandlerSimple<UsageMetadataContext, S> + 'static,
    ) -> Self {
        self.handlers.on_usage_metadata = Some(Arc::new(handler));
        self
    }

    pub fn realtime_input_config(mut self, config: RealtimeInputConfig) -> Self {
        self.initial_setup.realtime_input_config = Some(config);
        self
    }

    pub fn output_audio_transcription(mut self, config: AudioTranscriptionConfig) -> Self {
        self.initial_setup.output_audio_transcription = Some(config);
        self
    }

    /// Enables automatic audio resampling to 16kHz mono if the input audio
    /// provided to `send_audio_chunk` is not already in that format.
    ///
    /// When enabled, the first call to `send_audio_chunk` with a format
    /// other than 16kHz mono will initialize an internal resampler tailored
    /// to that specific input audio format. Subsequent calls to `send_audio_chunk`
    /// for this client instance are expected to use the *same* input audio format.
    /// If the input audio format changes after the resampler has been initialized,
    /// an error will be returned (unless future versions offer a re-initialization strategy).
    ///
    /// This requires the `audio-resampling` feature to be enabled for the crate.
    /// (e.g., `gemini-live-api = { version = "...", features = ["audio-resampling"] }`)
    ///
    /// If not enabled, or if the `audio-resampling` feature is not compiled,
    /// audio sent via `send_audio_chunk` **must** already be 16kHz mono PCM.
    #[cfg(feature = "audio-resampling")]
    pub fn enable_automatic_resampling(mut self) -> Self {
        self.automatic_resampling_enabled = true;
        tracing::info!(
            "Automatic audio resampling to 16kHz mono has been enabled. Resampler will be initialized on first use if needed."
        );
        self
    }

    pub async fn connect(self) -> Result<GeminiLiveClient<S>, GeminiError> {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let (outgoing_sender, outgoing_receiver) = mpsc::channel(100);

        let state_arc = Arc::new(self.state);
        let handlers_arc = Arc::new(self.handlers);

        super::connection::spawn_processing_task(
            self.api_key.clone(),
            self.initial_setup,
            handlers_arc,
            state_arc.clone(),
            shutdown_rx,
            outgoing_receiver,
        );

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        Ok(GeminiLiveClient {
            shutdown_tx: Arc::new(TokioMutex::new(Some(shutdown_tx))), // Wrapped
            outgoing_sender: Some(outgoing_sender),
            state: state_arc,
            #[cfg(feature = "audio-resampling")]
            resampler_state: Arc::new(TokioMutex::new(None)),
            #[cfg(feature = "audio-resampling")]
            automatic_resampling_configured_in_builder: self.automatic_resampling_enabled,
        })
    }
}
