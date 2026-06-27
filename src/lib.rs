//! # agentic-memory — Production-Grade Hierarchical Memory Layer
//!
//! **100% domain-agnostic** — Zero dependencies on trading or any specific system.
//!
//! ## Architecture
//!
//! - `store` — SQLite-backed persistent storage with FTS5 full-text search
//! - `tiers` — Hierarchical memory tiers (Working → Episodic → Semantic → Procedural)
//! - `vector` — Vector similarity search with cosine + binary quantization (Hamming)
//! - `graph` — Knowledge graph with recursive BFS traversal and relationship tracking
//! - `resilience` — Reusable Circuit Breaker and Resilient HTTP Client (Retry + Circuit Breaker)
//! - `rag` — RAG pipeline: text chunking, embedding trait, semantic search, Ollama integration
//! - `reasoning` — Chain-of-thought reasoning chains with storage and retrieval
//! - `consolidation` — Memory consolidation pipeline (importance scoring, dedup, merge, conflict detection)
//! - `evolution` — Self-evolving engine (tier tuning, stale pruning, sleep-time compute)
//! - `experts` — Expert modules (Retrieval, Reasoning, Consolidation, Evolution) with Orchestrator
//! - `cache` — In-memory policy cache with TTL and hit-rate analytics
//! - `api` — HTTP REST API (axum) for remote memory access
//! - `types` — All type definitions
//!
//! ## Quick Start
//!
//! ```no_run
//! use agentic_memory::{MemoryStore, MemoryRecord, StorageConfig, MemoryTier};
//!
//! let config = StorageConfig::default();
//! let store = MemoryStore::open(&config).unwrap();
//!
//! store.insert_into_tier(&MemoryRecord::new(
//!     "doc-1".into(),
//!     "Some content to remember".into(),
//!     "note".into(),
//! ), MemoryTier::Episodic, 0.7, None, None).unwrap();
//! ```

pub mod api;
pub mod cache;
pub mod consolidation;
pub mod errors;
pub mod evolution;
pub mod experts;
pub mod graph;
pub mod rag;
pub mod reasoning;
pub mod resilience;
pub mod store;
pub mod tiers;
pub mod types;
pub mod vector;
pub mod staleness;
pub mod client;
pub mod embed_cohere;
pub mod embed_openai;
pub mod llm;
pub mod mcp;
pub mod metrics;
pub mod openai_compat;
pub mod tools;

pub use staleness::StalenessManager;

/// Centralized ID generation (UUIDv7 style for better sortability and collision resistance)
pub fn generate_id() -> String {
    use uuid::Uuid;
    Uuid::now_v7().to_string()
}

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use api::MemoryApi;
pub use cache::{CachedItem, PolicyCache};
pub use consolidation::{ConsolidationEngine, DefaultImportanceScorer};
pub use evolution::{EvolutionEngine, EvolutionConfig, SleepCycleHandle, SleepCycleReport};
pub use experts::RetrievalExpert;
pub use graph::KnowledgeGraph;
pub use rag::{chunk_text, ChunkStrategy, DummyEmbedder, Embedder, OllamaEmbedder, RagPipeline};
pub use reasoning::ReasoningEngine;
pub use store::MemoryStore;
pub use tiers::{PromotionEngine, TieredMemory, WorkingMemory};
pub use types::*;
pub use vector::*;

pub use client::MemoryClient;
pub use embed_cohere::CohereEmbedder;
pub use embed_openai::OpenAIEmbedder;
pub use resilience::{CircuitBreaker, CircuitState, ResilientClient};
pub use llm::{LLMClient, LLMResponse, Message, ToolDefinition, ToolCall};

// Note: The old timestamp-based generate_id() was removed.
// We now exclusively use the UUIDv7 version defined above for better sortability and collision resistance.
