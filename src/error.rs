use thiserror::Error;
use tokio_tungstenite::tungstenite;

#[derive(Error, Debug)]
pub enum GeminiError {
    #[error("WebSocket connection error: {0}")]
    WebSocketError(#[from] tungstenite::Error),

    #[error("JSON serialization/deserialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Connection not established or setup not complete")]
    NotReady,

    #[error("Message from server was not in expected format")]
    UnexpectedMessage,

    #[error("Attempted to send message on a closed connection")]
    ConnectionClosed,

    #[error("Function call handler not found for: {0}")]
    FunctionHandlerNotFound(String),

    #[error("Error sending message")]
    SendError,

    #[error("Missing API key")]
    MissingApiKey,

    #[error("Audio resampling error: {0}")]
    AudioResamplingError(String),

    #[error("Internal client error: {0}")]
    InternalError(String),
}
