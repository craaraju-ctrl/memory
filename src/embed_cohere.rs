//! # Cohere Embedder — Cloud Embedding Provider
//!
//! Connects to Cohere's embedding API.
//! Supports models: embed-english-v3.0, embed-multilingual-v3.0, embed-english-light-v3.0.
//!
//! ## Configuration
//!
//! | Env Var | Default | Description |
//! |---------|---------|-------------|
//! | `COHERE_API_KEY` | — | Required. Your Cohere API key |
//!
//! ## Usage
//!
//! ```no_run
//! # async fn doc() {
//! use agentic_memory::rag::{Embedder, CohereEmbedder};
//! let embedder = CohereEmbedder::new("embed-english-v3.0", "your-api-key").unwrap();
//! let vec = embedder.embed("Hello world").await.unwrap();
//! println!("Embedding dimension: {}", vec.len());
//! # }
//! ```

use crate::resilience::ResilientClient;

/// An embedder that connects to Cohere's embedding API.
pub struct CohereEmbedder {
    client: ResilientClient,
    model: String,
    api_key: String,
    dimension: usize,
}

impl CohereEmbedder {
    /// Create a new Cohere embedder.
    pub fn new(model: &str, api_key: &str) -> Result<Self, String> {
        if api_key.is_empty() {
            return Err("Cohere API key is required".to_string());
        }

        let dimension = match model {
            "embed-english-v3.0" => 1024,
            "embed-multilingual-v3.0" => 1024,
            "embed-english-light-v3.0" => 384,
            "embed-v4.0" => 1024,
            _ => 1024,
        };

        Ok(Self {
            client: ResilientClient::new(3, 5, 30),
            model: model.to_string(),
            api_key: api_key.to_string(),
            dimension,
        })
    }

    /// Create from environment variables (COHERE_API_KEY).
    pub fn from_env() -> Result<Self, String> {
        let api_key =
            std::env::var("COHERE_API_KEY").map_err(|_| "COHERE_API_KEY not set".to_string())?;
        let model = std::env::var("COHERE_EMBED_MODEL")
            .unwrap_or_else(|_| "embed-english-v3.0".to_string());
        Self::new(&model, &api_key)
    }

    async fn embed_inner(&self, text: &str) -> Result<Vec<f64>, String> {
        let body = serde_json::json!({
            "texts": [text],
            "model": self.model,
            "input_type": "search_document",
            "embedding_types": ["float"],
        });

        let json: serde_json::Value = self
            .client
            .post_json_with_headers(
                "https://api.cohere.com/v2/embed",
                &body,
                &[("Authorization", format!("Bearer {}", self.api_key).as_str())],
            )
            .await?;

        let embeddings = json["embeddings"]["float"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|v| v.as_array())
            .ok_or_else(|| "No embeddings in response".to_string())?;

        let vector: Vec<f64> = embeddings
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        Ok(vector)
    }
}

#[async_trait::async_trait]
impl super::Embedder for CohereEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f64>, String> {
        self.embed_inner(text).await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, String> {
        let body = serde_json::json!({
            "texts": texts,
            "model": self.model,
            "input_type": "search_document",
            "embedding_types": ["float"],
        });

        let json: serde_json::Value = self
            .client
            .post_json_with_headers(
                "https://api.cohere.com/v2/embed",
                &body,
                &[("Authorization", format!("Bearer {}", self.api_key).as_str())],
            )
            .await?;

        let embeddings = json["embeddings"]["float"]
            .as_array()
            .ok_or_else(|| "No embeddings array in response".to_string())?;

        let vectors: Vec<Vec<f64>> = embeddings
            .iter()
            .map(|arr| {
                arr.as_array()
                    .map(|v| v.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default()
            })
            .collect();

        Ok(vectors)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
