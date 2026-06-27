//! # Memory Types — Production-Grade Hierarchical Memory
//!
//! **100% domain-agnostic** — Zero dependencies on trading or any specific system.
//!
//! ## Memory Tier Architecture
//!
//! - `Working`   — Ephemeral context buffer, TTL-managed, auto-evicts
//! - `Episodic`  — Time-bound events and experiences
//! - `Semantic`  — Facts, preferences, deduplicated knowledge
//! - `Procedural`— Learned tools, workflows, behavioral rules
//!
//! ## Extended Capabilities
//!
//! - Bitemporal metadata (valid_time + transaction_time)
//! - Knowledge graph relationships with recursive traversal
//! - Chain-of-thought reasoning chains with confidence scoring
//! - Expert opinions from specialized modules
//! - Evolution events for self-adaptation tracking

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Memory Record (unchanged, base type) ───────────────────────────────────

/// A generic memory record that can hold any type of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub content: String,
    pub content_type: String,
    pub metadata: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f64>>,
    pub timestamp: String,
}

impl MemoryRecord {
    pub fn new(id: String, content: String, content_type: String) -> Self {
        Self {
            id,
            content,
            content_type,
            metadata: HashMap::new(),
            embedding: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f64>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

// ── Memory Tier Enum ───────────────────────────────────────────────────────

/// The four tiers of the hierarchical memory system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryTier {
    /// Ephemeral context buffer — TTL-managed, auto-evicted, in-memory
    Working,
    /// Time-bound events and experiences — what happened
    Episodic,
    /// Facts, preferences, concepts — what is true
    Semantic,
    /// Learned tools, workflows, behavioral rules — how to act
    Procedural,
}

impl MemoryTier {
    /// All tiers in order from most volatile to most permanent.
    pub fn all() -> Vec<MemoryTier> {
        vec![
            MemoryTier::Working,
            MemoryTier::Episodic,
            MemoryTier::Semantic,
            MemoryTier::Procedural,
        ]
    }

    /// The next more permanent tier for promotion.
    pub fn promote_to(&self) -> Option<MemoryTier> {
        match self {
            MemoryTier::Working => Some(MemoryTier::Episodic),
            MemoryTier::Episodic => Some(MemoryTier::Semantic),
            MemoryTier::Semantic => Some(MemoryTier::Procedural),
            MemoryTier::Procedural => None,
        }
    }

    /// The next more volatile tier for demotion.
    pub fn demote_to(&self) -> Option<MemoryTier> {
        match self {
            MemoryTier::Working => None,
            MemoryTier::Episodic => Some(MemoryTier::Working),
            MemoryTier::Semantic => Some(MemoryTier::Episodic),
            MemoryTier::Procedural => Some(MemoryTier::Semantic),
        }
    }
}

impl std::fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryTier::Working => write!(f, "working"),
            MemoryTier::Episodic => write!(f, "episodic"),
            MemoryTier::Semantic => write!(f, "semantic"),
            MemoryTier::Procedural => write!(f, "procedural"),
        }
    }
}

impl std::str::FromStr for MemoryTier {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "working" => Ok(MemoryTier::Working),
            "episodic" => Ok(MemoryTier::Episodic),
            "semantic" => Ok(MemoryTier::Semantic),
            "procedural" => Ok(MemoryTier::Procedural),
            _ => Err(format!("Unknown memory tier: {}", s)),
        }
    }
}

// ── Tiered Record ───────────────────────────────────────────────────────────

/// A memory record with tier, importance, and access tracking metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredRecord {
    pub record: MemoryRecord,
    pub tier: MemoryTier,
    /// Importance score 0.0–1.0 (used for promotion/eviction decisions)
    pub importance: f64,
    /// How many times this record has been accessed
    pub access_count: u64,
    /// Last access timestamp (RFC 3339)
    pub last_accessed: String,
    /// TTL in seconds (None = permanent in that tier)
    pub ttl_seconds: Option<u64>,
    /// Parent record ID for hierarchical relationships
    pub parent_id: Option<String>,
    /// Free-form tags for filtering
    pub tags: Vec<String>,
}

// ── Tier Configuration ─────────────────────────────────────────────────────

/// Configuration for a specific memory tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// Maximum number of records in this tier
    pub max_records: usize,
    /// Default TTL in seconds (None = permanent)
    pub default_ttl_seconds: Option<u64>,
    /// Importance threshold for promotion to the next tier (0.0–1.0)
    pub promotion_threshold: f64,
    /// Importance threshold for demotion to the previous tier (0.0–1.0)
    pub demotion_threshold: f64,
    /// Whether auto-promotion is enabled for this tier
    pub auto_promote: bool,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            max_records: 1000,
            default_ttl_seconds: None,
            promotion_threshold: 0.7,
            demotion_threshold: 0.2,
            auto_promote: true,
        }
    }
}

impl TierConfig {
    /// Sensible defaults for each tier.
    pub fn for_tier(tier: MemoryTier) -> Self {
        match tier {
            MemoryTier::Working => Self {
                max_records: 100,
                default_ttl_seconds: Some(3600), // 1 hour
                promotion_threshold: 0.5,
                demotion_threshold: 0.1,
                auto_promote: true,
            },
            MemoryTier::Episodic => Self {
                max_records: 10_000,
                default_ttl_seconds: Some(86400 * 30), // 30 days
                promotion_threshold: 0.7,
                demotion_threshold: 0.2,
                auto_promote: true,
            },
            MemoryTier::Semantic => Self {
                max_records: 100_000,
                default_ttl_seconds: None, // permanent
                promotion_threshold: 0.85,
                demotion_threshold: 0.15,
                auto_promote: false, // manual or expert-driven
            },
            MemoryTier::Procedural => Self {
                max_records: 10_000,
                default_ttl_seconds: None, // permanent
                promotion_threshold: 0.95,
                demotion_threshold: 0.1,
                auto_promote: false,
            },
        }
    }
}

// ── Tier Statistics ────────────────────────────────────────────────────────

/// Statistics for a specific tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierStats {
    pub tier: MemoryTier,
    pub total_records: u64,
    pub total_with_embeddings: u64,
    pub average_importance: f64,
    pub total_accesses: u64,
    pub storage_bytes: u64,
    pub config: TierConfig,
}

/// Aggregated stats across all tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_records: u64,
    pub total_with_embeddings: u64,
    pub content_types: HashMap<String, u64>,
    pub storage_bytes: u64,
    pub tier_breakdown: HashMap<String, TierStats>,
}

// ── Knowledge Graph Types ──────────────────────────────────────────────────

/// A directed edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub edge_id: String,
    pub source_id: String,
    pub target_id: String,
    /// Relationship type, e.g. "related_to", "causes", "depends_on", "part_of"
    pub relation_type: String,
    /// Edge weight 0.0–1.0 (strength of relationship)
    pub weight: f64,
    /// Arbitrary metadata
    pub metadata: HashMap<String, String>,
    pub created_at: String,
}

/// Result of a graph traversal step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTraversalResult {
    pub node_id: String,
    pub depth: u32,
    pub path: Vec<String>,
    pub cumulative_weight: f64,
}

// ── Temporal / Bitemporal Metadata ─────────────────────────────────────────

/// Bitemporal metadata for tracking when a fact was true vs when it was recorded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMetadata {
    /// When this fact was true in the real world (RFC 3339)
    pub valid_from: String,
    /// When this fact ceased being true (None = still valid)
    pub valid_to: Option<String>,
    /// When this fact was recorded in the system (RFC 3339)
    pub sys_start: String,
    /// When this record was superseded (None = current version)
    pub sys_end: Option<String>,
}

// ── Reasoning / Chain-of-Thought Types ─────────────────────────────────────

/// A single step in a chain of reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_index: u32,
    /// The premise or context for this step
    pub premise: String,
    /// The inference or action taken
    pub inference: String,
    /// The conclusion drawn from this step
    pub conclusion: String,
    /// Confidence in this step (0.0–1.0)
    pub confidence: f64,
    /// What tool or method was used
    pub tool_used: Option<String>,
    /// Whether this step succeeded
    pub success: bool,
    /// Timestamp of this step
    pub timestamp: String,
}

/// A complete chain of reasoning for a task or goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    pub chain_id: String,
    pub goal: String,
    pub steps: Vec<ReasoningStep>,
    pub final_conclusion: Option<String>,
    pub overall_confidence: f64,
    pub success: bool,
    /// Memory record IDs that were consulted during reasoning
    pub consulted_records: Vec<String>,
    /// Tags for retrieval
    pub tags: Vec<String>,
    pub created_at: String,
    pub duration_ms: u64,
}

// ── Expert Module Types ────────────────────────────────────────────────────

/// Types of expert modules available in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertType {
    /// Specialized in memory retrieval across all tiers
    Retrieval,
    /// Specialized in chain-of-thought reasoning
    Reasoning,
    /// Specialized in memory consolidation and management
    Consolidation,
    /// Specialized in system evolution and adaptation
    Evolution,
}

impl ExpertType {
    pub fn all() -> Vec<ExpertType> {
        vec![
            ExpertType::Retrieval,
            ExpertType::Reasoning,
            ExpertType::Consolidation,
            ExpertType::Evolution,
        ]
    }
}

impl std::fmt::Display for ExpertType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpertType::Retrieval => write!(f, "retrieval"),
            ExpertType::Reasoning => write!(f, "reasoning"),
            ExpertType::Consolidation => write!(f, "consolidation"),
            ExpertType::Evolution => write!(f, "evolution"),
        }
    }
}

/// An opinion or recommendation from an expert module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertOpinion {
    pub opinion_id: String,
    pub expert_type: ExpertType,
    pub target_record_id: Option<String>,
    pub recommendation: String,
    pub reasoning: String,
    pub confidence: f64,
    pub action_taken: Option<String>,
    pub created_at: String,
}

// ── Consolidation Types ────────────────────────────────────────────────────

/// Result of a consolidation cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationReport {
    pub cycle_id: String,
    pub started_at: String,
    pub completed_at: String,
    pub records_processed: u64,
    pub records_extracted: u64,
    pub records_deduplicated: u64,
    pub records_merged: u64,
    pub records_summarized: u64,
    pub records_promoted: u64,
    pub records_demoted: u64,
    pub records_evicted: u64,
    pub conflicts_detected: u64,
    pub conflicts_resolved: u64,
    pub insights_generated: Vec<String>,
    pub duration_ms: u64,
}

/// Context for computing importance scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceContext {
    pub access_count: u64,
    pub age_seconds: f64,
    pub has_embedding: bool,
    pub content_length: usize,
    pub content_type: String,
    pub tier: MemoryTier,
    pub graph_connections: usize,
    pub expert_endorsements: usize,
}

// ── Evolution Types ────────────────────────────────────────────────────────

/// An event recorded by the evolution system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionEvent {
    pub event_id: String,
    pub event_type: String, // e.g. "tier_tuned", "procedural_distilled", "stale_pruned"
    pub description: String,
    pub previous_value: Option<String>,
    pub new_value: Option<String>,
    pub confidence: f64,
    pub timestamp: String,
}

// ── Search Result ──────────────────────────────────────────────────────────

/// A search result from vector or keyword queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub record: MemoryRecord,
    pub score: f64,
    pub method: String,
}

// ── Storage Configuration ──────────────────────────────────────────────────

/// Configuration for the storage backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub db_path: String,
    pub max_ram_entries: usize,
    pub auto_embed: bool,
    /// Dimension for vector embeddings (used by sqlite-vec vec0 tables)
    #[serde(default = "default_vector_dim")]
    pub vector_dimension: usize,
}

fn default_vector_dim() -> usize {
    768
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: ":memory:".into(),
            max_ram_entries: 100,
            auto_embed: false,
            vector_dimension: default_vector_dim(),
        }
    }
}

// ── Text Chunk ─────────────────────────────────────────────────────────────

/// A chunk of text with its position in the original document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    pub index: usize,
    pub text: String,
    pub token_count: usize,
}
