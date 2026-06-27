//! # OpenAI Embedder — Cloud Embedding Provider
//!
//! Connects to OpenAI's embedding API (or any OpenAI-compatible endpoint).
//! Supports models: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002.
//!
//! ## Configuration
//!
//! | Env Var | Default | Description |
//! |---------|---------|-------------|
//! | `OPENAI_API_KEY` | — | Required. Your OpenAI API key |
//! | `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Override for compatible endpoints |
//!
//! ## Usage
//!
//! ```no_run
//! # async fn doc() {
//! use agentic_memory::rag::{Embedder, OpenAIEmbedder};
//! let embedder = OpenAIEmbedder::new(
//!     "https://api.openai.com/v1",
//!     "text-embedding-3-small",
//!     "sk-...",
//! ).unwrap();
//! let vec = embedder.embed("Hello world").await.unwrap();
//! println!("Embedding dimension: {}", vec.len());
//! # }
//! ```

use crate::resilience::ResilientClient;

/// An embedder that connects to OpenAI's embedding API.
pub struct OpenAIEmbedder {
    client: ResilientClient,
    base_url: String,
    model: String,
    api_key: String,
    dimension: usize,
}

impl OpenAIEmbedder {
    /// Create a new OpenAI embedder.
    pub fn new(base_url: &str, model: &str, api_key: &str) -> Result<Self, String> {
        if api_key.is_empty() {
            return Err("OpenAI API key is required".to_string());
        }

        let dimension = match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536, // default for unknown models
        };

        Ok(Self {
            client: ResilientClient::new(3, 5, 30),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
            dimension,
        })
    }

    /// Create from environment variables (OPENAI_API_KEY, OPENAI_BASE_URL).
    pub fn from_env() -> Result<Self, String> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set".to_string())?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model = std::env::var("OPENAI_EMBED_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".to_string());
        Self::new(&base_url, &model, &api_key)
    }

    async fn embed_inner(&self, text: &str) -> Result<Vec<f64>, String> {
        let url = format!("{}/embeddings", self.base_url);

        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });

        let json: serde_json::Value = self.client
            .post_json_with_headers(
                &url,
                &body,
                &[("Authorization", format!("Bearer {}", self.api_key).as_str())],
            )
            .await?;

        let data = json["data"]
            .as_array()
            .and_then(|a| a.first())
            .ok_or_else(|| "No data in response".to_string())?;

        let embedding = data["embedding"]
            .as_array()
            .ok_or_else(|| "No embedding in response".to_string())?;

        let vector: Vec<f64> = embedding
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        Ok(vector)
    }
}

#[async_trait::async_trait]
impl super::Embedder for OpenAIEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f64>, String> {
        self.embed_inner(text).await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, String> {
        let url = format!("{}/embeddings", self.base_url);

        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let json: serde_json::Value = self.client
            .post_json_with_headers(
                &url,
                &body,
                &[("Authorization", format!("Bearer {}", self.api_key).as_str())],
            )
            .await?;

        let data = json["data"]
            .as_array()
            .ok_or_else(|| "No data array in response".to_string())?;

        let vectors: Vec<Vec<f64>> = data
            .iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default()
            })
            .collect();

        Ok(vectors)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
