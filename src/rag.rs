//! # RAG Pipeline — Retrieval-Augmented Generation
//!
//! **100% domain-agnostic** — Generic text chunking, embedding interface,
//! and semantic search over stored memory records.

use crate::types::{MemoryRecord, SearchResult, TextChunk};
use crate::vector::cosine_similarity;

/// Known embedding model dimensions (model_name -> dimension)
fn known_model_dim(model: &str) -> Option<usize> {
    match model {
        "nomic-embed-text" => Some(768),
        "mxbai-embed-large" => Some(1024),
        "all-minilm" => Some(384),
        "bge-m3" => Some(1024),
        "snowflake-arctic-embed" => Some(1024),
        "llama3.2" => Some(3072),
        "llama3.1" => Some(4096),
        _ => None,
    }
}

// ── Embedding Trait ─────────────────────────────────────────────────────────

/// Trait for generating embeddings from text.
/// Implementations can use local models (Ollama, llama.cpp) or cloud APIs (OpenAI, etc.).
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    /// Generate an embedding vector for a single text string.
    async fn embed(&self, text: &str) -> Result<Vec<f64>, String>;

    /// Generate embeddings for multiple texts in parallel (if supported).
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, String> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// The dimension of the embedding vectors produced.
    fn dimension(&self) -> usize;
}

// ── Text Chunking ──────────────────────────────────────────────────────────

/// Strategy for splitting text into chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStrategy {
    /// Split by sentence boundaries
    Sentence,
    /// Split by paragraph boundaries
    Paragraph,
    /// Split by token count (approximate: split on whitespace)
    TokenCount { max_tokens: usize, overlap: usize },
}

/// Split text into chunks using the given strategy.
pub fn chunk_text(text: &str, strategy: ChunkStrategy) -> Vec<TextChunk> {
    match strategy {
        ChunkStrategy::Sentence => chunk_by_sentences(text),
        ChunkStrategy::Paragraph => chunk_by_paragraphs(text),
        ChunkStrategy::TokenCount { max_tokens, overlap } => {
            chunk_by_tokens(text, max_tokens, overlap)
        }
    }
}

fn chunk_by_sentences(text: &str) -> Vec<TextChunk> {
    let mut chunks = Vec::new();
    // Simple sentence split on . ! ? followed by space or end
    let mut current = String::new();
    let mut index = 0;

    for sentence in text.split_inclusive(['.', '!', '?']) {
        let trimmed = sentence.trim();
        if trimmed.is_empty() {
            continue;
        }
        current.push_str(trimmed);
        current.push(' ');

        // Group 2-3 sentences per chunk for better context
        if current.split(['.', '!', '?']).count() >= 3 {
            let text = current.trim().to_string();
            let token_count = text.split_whitespace().count();
            chunks.push(TextChunk {
                index,
                text,
                token_count,
            });
            index += 1;
            current = String::new();
        }
    }

    // Don't forget the last partial chunk
    let text = current.trim().to_string();
    if !text.is_empty() {
        let token_count = text.split_whitespace().count();
        chunks.push(TextChunk {
            index,
            text,
            token_count,
        });
    }

    chunks
}

fn chunk_by_paragraphs(text: &str) -> Vec<TextChunk> {
    text.split("\n\n")
        .enumerate()
        .filter(|(_, p)| !p.trim().is_empty())
        .map(|(i, p)| {
            let trimmed = p.trim().to_string();
            let token_count = trimmed.split_whitespace().count();
            TextChunk {
                index: i,
                text: trimmed,
                token_count,
            }
        })
        .collect()
}

fn chunk_by_tokens(text: &str, max_tokens: usize, overlap: usize) -> Vec<TextChunk> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut index = 0;

    while start < tokens.len() {
        let end = std::cmp::min(start + max_tokens, tokens.len());
        let chunk_tokens = &tokens[start..end];
        let text = chunk_tokens.join(" ");
        let token_count = chunk_tokens.len();

        chunks.push(TextChunk {
            index,
            text,
            token_count,
        });
        index += 1;

        if end >= tokens.len() {
            break;
        }
        start = if overlap > 0 {
            start + max_tokens - overlap
        } else {
            end
        };
    }

    chunks
}

// ── RAG Pipeline ────────────────────────────────────────────────────────────

/// The RAG pipeline: embed, store, retrieve, and rerank.
pub struct RagPipeline {
    embedder: Box<dyn Embedder>,
}

impl RagPipeline {
    pub fn new(embedder: Box<dyn Embedder>) -> Self {
        Self { embedder }
    }

    /// Embed a memory record's content and attach the embedding.
    pub async fn embed_record(&self, mut record: MemoryRecord) -> Result<MemoryRecord, String> {
        let embedding = self.embedder.embed(&record.content).await?;
        record.embedding = Some(embedding);
        Ok(record)
    }

    /// Embed multiple records in batch.
    pub async fn embed_batch(&self, records: Vec<MemoryRecord>) -> Result<Vec<MemoryRecord>, String> {
        let texts: Vec<&str> = records.iter().map(|r| r.content.as_str()).collect();
        let embeddings = self.embedder.embed_batch(&texts).await?;

        let mut result = records;
        for (record, embedding) in result.iter_mut().zip(embeddings) {
            record.embedding = Some(embedding);
        }
        Ok(result)
    }

    /// Chunk a long text and embed each chunk.
    pub async fn chunk_and_embed(
        &self,
        doc_id: &str,
        text: &str,
        content_type: &str,
        strategy: ChunkStrategy,
    ) -> Result<Vec<MemoryRecord>, String> {
        let chunks = chunk_text(text, strategy);
        let mut records = Vec::with_capacity(chunks.len());

        for chunk in &chunks {
            let record = MemoryRecord::new(
                format!("{}/chunk-{}", doc_id, chunk.index),
                chunk.text.clone(),
                format!("{}/chunk", content_type),
            )
            .with_metadata("parent_id", doc_id)
            .with_metadata("chunk_index", &chunk.index.to_string())
            .with_metadata("token_count", &chunk.token_count.to_string());

            records.push(record);
        }

        self.embed_batch(records).await
    }

    /// Search records by semantic similarity to a query string.
    /// Requires records to already have embeddings.
    pub fn search_semantic(
        &self,
        query_vec: &[f64],
        records: &[MemoryRecord],
        k: usize,
    ) -> Vec<SearchResult> {
        let mut scored: Vec<SearchResult> = records
            .iter()
            .filter_map(|record| {
                let emb = record.embedding.as_ref()?;
                let score = (cosine_similarity(query_vec, emb) + 1.0) / 2.0; // normalize to [0,1]
                Some(SearchResult {
                    record: record.clone(),
                    score,
                    method: "semantic".into(),
                })
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).collect()
    }

    /// Get the embedder's dimension.
    pub fn dimension(&self) -> usize {
        self.embedder.dimension()
    }
}

pub use crate::embed_openai::OpenAIEmbedder;
pub use crate::embed_cohere::CohereEmbedder;

// ── Ollama Embedder ─────────────────────────────────────────────────────────

/// An embedder that connects to a local Ollama instance for real embeddings.
///
/// Uses the `POST /api/embed` endpoint with configurable model.
/// Defaults to `nomic-embed-text` (768 dims), connecting to `http://localhost:11434`.
///
/// ## Usage
///
/// ```no_run
/// # async fn doc() {
/// use agentic_memory::rag::{Embedder, OllamaEmbedder};
/// let embedder = OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text").unwrap();
/// let vec = embedder.embed("Hello world").await.unwrap();
/// println!("Embedding dimension: {}", vec.len());
/// # }
/// ```
pub struct OllamaEmbedder {
    client: crate::resilience::ResilientClient,
    base_url: String,
    model: String,
    dimension: usize,
}

impl OllamaEmbedder {
    /// Create a new Ollama embedder.
    /// `base_url` should be the Ollama server URL (e.g. `http://localhost:11434`).
    /// `model` should be a model pulled locally (e.g. `nomic-embed-text`, `llama3.2`, `mxbai-embed-large`).
    /// The dimension is inferred from a known-model table; use `detect_dimension()` after creation for accuracy.
    pub fn new(base_url: &str, model: &str) -> Result<Self, String> {
        // Resilience configuration via environment variables
        let max_retries: u32 = std::env::var("MEMORY_OLLAMA_MAX_RETRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);

        let cb_threshold: u32 = std::env::var("MEMORY_OLLAMA_CIRCUIT_BREAKER_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);

        let cb_reset_secs: u64 = std::env::var("MEMORY_OLLAMA_CIRCUIT_BREAKER_RESET_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);

        let client = crate::resilience::ResilientClient::new(max_retries, cb_threshold, cb_reset_secs);

        let dim = known_model_dim(model).unwrap_or(768);

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            dimension: dim,
        })
    }

    /// Dynamically detect the embedding dimension from the model.
    /// Sends a test embedding and reads the response length.
    pub async fn detect_dimension(&self) -> Result<usize, String> {
        let test = self.embed_inner("test").await?;
        Ok(test.len())
    }

    async fn embed_inner(&self, text: &str) -> Result<Vec<f64>, String> {
        let url = format!("{}/api/embed", self.base_url);

        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });

        // Use the resilient client (Circuit Breaker + Retry)
        let json: serde_json::Value = self
            .client
            .post_json(&url, &body)
            .await?;

        let embeddings = json["embeddings"]
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
impl Embedder for OllamaEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f64>, String> {
        self.embed_inner(text).await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, String> {
        let url = format!("{}/api/embed", self.base_url);

        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let json: serde_json::Value = self
            .client
            .post_json(&url, &body)
            .await?;

        let embeddings = json["embeddings"]
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

// ── Simple Embedder (placeholders for testing) ──────────────────────────────

/// A dummy embedder that produces random vectors of fixed dimension.
/// Useful for testing the RAG pipeline without a real embedding model.
pub struct DummyEmbedder {
    dimension: usize,
}

impl DummyEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait::async_trait]
impl Embedder for DummyEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f64>, String> {
        // Deterministic "embedding" based on text length (for testing)
        Ok(vec![0.1; self.dimension])
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_by_paragraphs() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunk_text(text, ChunkStrategy::Paragraph);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_chunk_by_tokens() {
        let text = "a b c d e f g h i j";
        let chunks = chunk_text(text, ChunkStrategy::TokenCount {
            max_tokens: 3,
            overlap: 1,
        });
        assert_eq!(chunks.len(), 5); // 3+3+3+3+2 with overlap 1
        assert_eq!(chunks[0].text, "a b c");
        assert_eq!(chunks[1].text, "c d e");
    }

    #[test]
    fn test_chunk_by_sentences() {
        let text = "First sentence here. Second one here. Third is short. Fourth is here.";
        let chunks = chunk_text(text, ChunkStrategy::Sentence);
        assert!(chunks.len() >= 1);
    }

    #[tokio::test]
    async fn test_rag_pipeline() {
        let embedder = DummyEmbedder::new(64);
        let pipeline = RagPipeline::new(Box::new(embedder));

        let record = MemoryRecord::new("test".into(), "Hello world".into(), "test".into());
        let embedded = pipeline.embed_record(record).await.unwrap();
        assert!(embedded.embedding.is_some());
        assert_eq!(embedded.embedding.unwrap().len(), 64);
    }
}
