//! # OpenAI-Compatible API — Drop-in Memory Endpoint
//!
//! Exposes the Memory module as an OpenAI-compatible endpoint so any
//! OpenAI client library can use it as a drop-in memory store.
//!
//! ## Supported Endpoints
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/v1/chat/completions` | POST | Search memory and return as chat context |
//! | `/v1/embeddings` | POST | Generate embeddings via configured provider |
//! | `/v1/models` | GET | List available models/providers |
//! | `/v1/health` | GET | Health check |
//!
//! ## Usage
//!
//! ```bash
//! # Start with OpenAI-compatible mode
//! OPENAI_COMPAT=1 cargo run
//!
//! # Use with any OpenAI client
//! export OPENAI_BASE_URL=http://localhost:3111/v1
//! export OPENAI_API_KEY=memory-api
//! ```

use axum::{extract::State, http::StatusCode, Json, response::IntoResponse};
use serde::{Deserialize, Serialize};
use crate::rag::Embedder;

/// OpenAI-compatible chat completion request.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// A chat message.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// A single choice in the response.
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Token usage statistics.
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI-compatible embedding request.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
}

/// Embedding input (string or array of strings).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

/// OpenAI-compatible embedding response.
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

/// A single embedding in the response.
#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: u32,
    pub embedding: Vec<f64>,
}

/// OpenAI-compatible model list response.
#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model information.
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Shared state for the OpenAI-compatible API.
#[derive(Clone)]
pub struct OpenAICompatState {
    pub api_url: String,
    pub embedder: Option<std::sync::Arc<dyn Embedder>>,
}

/// POST /v1/chat/completions — Search memory and return as chat context.
pub async fn chat_completions(
    State(state): State<OpenAICompatState>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Extract the last user message as the search query
    let query = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .unwrap_or("");

    if query.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "No user message found to search with",
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response();
    }

    // Search memory
    let client = reqwest::Client::new();
    let search_url = format!(
        "{}/search?q={}&limit=10",
        state.api_url,
        urlencoding::encode(query)
    );

    let search_results = match client.get(&search_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            resp.json::<Vec<serde_json::Value>>().await.unwrap_or_default()
        }
        _ => vec![],
    };

    // Build context from search results
    let context = if search_results.is_empty() {
        "No relevant memories found.".to_string()
    } else {
        let memories: Vec<String> = search_results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let content = r["record"]["content"]
                    .as_str()
                    .or_else(|| r["content"].as_str())
                    .unwrap_or("");
                let score = r["score"].as_f64().unwrap_or(0.0);
                format!("{}. [relevance: {:.2}] {}", i + 1, score, content)
            })
            .collect();
        format!("Relevant memories:\n{}", memories.join("\n"))
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::now_v7()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: context,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: query.len() as u32 / 4,
            completion_tokens: 100,
            total_tokens: query.len() as u32 / 4 + 100,
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// POST /v1/embeddings — Generate embeddings via configured provider.
pub async fn embeddings(
    State(state): State<OpenAICompatState>,
    Json(req): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    let texts = match req.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Multiple(v) => v,
    };

    let embedder = match &state.embedder {
        Some(e) => e,
        None => {
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({
                    "error": {
                        "message": "No embedding provider configured. Set OLLAMA_BASE_URL env var.",
                        "type": "server_error",
                        "code": "no_embedder_configured"
                    }
                })),
            )
                .into_response();
        }
    };

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let data: Vec<EmbeddingData> = match embedder.embed_batch(&text_refs).await {
        Ok(embeddings) => {
            embeddings.into_iter().enumerate().map(|(i, emb)| EmbeddingData {
                object: "embedding".to_string(),
                index: i as u32,
                embedding: emb,
            }).collect()
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Embedding failed: {}", e),
                        "type": "server_error"
                    }
                })),
            )
                .into_response();
        }
    };

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: req.model,
        usage: Usage {
            prompt_tokens: texts.iter().map(|t| t.len() as u32 / 4).sum(),
            completion_tokens: 0,
            total_tokens: texts.iter().map(|t| t.len() as u32 / 4).sum(),
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// GET /v1/models — List available models/providers.
pub async fn list_models() -> impl IntoResponse {
    let response = ModelListResponse {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: "memory-search".to_string(),
                object: "model".to_string(),
                created: chrono::Utc::now().timestamp(),
                owned_by: "agentic-memory".to_string(),
            },
            ModelInfo {
                id: "memory-embed".to_string(),
                object: "model".to_string(),
                created: chrono::Utc::now().timestamp(),
                owned_by: "agentic-memory".to_string(),
            },
        ],
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// GET /v1/health — Health check.
pub async fn health_check(
    State(state): State<OpenAICompatState>,
) -> impl IntoResponse {
    let client = reqwest::Client::new();
    let health = match client
        .get(format!("{}/health", state.api_url))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            resp.json::<serde_json::Value>().await.unwrap_or(serde_json::json!({}))
        }
        _ => serde_json::json!({"status": "unreachable"}),
    };

    (StatusCode::OK, Json(health)).into_response()
}

/// Build the OpenAI-compatible router.
pub fn router(api_url: &str) -> axum::Router {
    use axum::routing::{get, post as post_route};

    // Create embedder once from env vars — reused across all requests
    let embedder: Option<std::sync::Arc<dyn Embedder>> = {
        let url = std::env::var("OLLAMA_BASE_URL").ok();
        let model = std::env::var("OLLAMA_MODEL").ok();
        match (url, model) {
            (Some(u), Some(m)) if !u.is_empty() && !m.is_empty() => {
                crate::rag::OllamaEmbedder::new(&u, &m)
                    .ok()
                    .map(|e| std::sync::Arc::new(e) as std::sync::Arc<dyn Embedder>)
            }
            _ => None,
        }
    };

    let state = OpenAICompatState {
        api_url: api_url.to_string(),
        embedder,
    };

    axum::Router::new()
        .route("/v1/chat/completions", post_route(chat_completions))
        .route("/v1/embeddings", post_route(embeddings))
        .route("/v1/models", get(list_models))
        .route("/v1/health", get(health_check))
        .with_state(state)
}
