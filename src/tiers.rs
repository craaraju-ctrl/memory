//! # Hierarchical Memory Tiers
//!
//! Implements a 4-tier memory architecture:
//! - **Working** — Ephemeral context buffer (in-memory, TTL-managed, LRU eviction)
//! - **Episodic** — Time-bound events and experiences
//! - **Semantic** — Facts, preferences, deduplicated knowledge
//! - **Procedural** — Learned tools, workflows, behavioral rules
//!
//! The `TieredMemory` orchestrator manages insertions, promotions, demotions,
//! and cross-tier search with automatic importance tracking.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::store::MemoryStore;
use crate::types::{MemoryRecord, MemoryTier, SearchResult, TierConfig, TieredRecord};

// ── Working Memory Buffer ──────────────────────────────────────────────────

/// An in-memory buffer for the Working tier with TTL and LRU eviction.
/// Records are automatically flushed to the Episodic tier when TTL expires
/// or the buffer reaches capacity.
pub struct WorkingMemory {
    /// Thread-safe in-memory buffer (hardened for direct library use)
    buffer: Mutex<HashMap<String, BufferEntry>>,
    max_size: usize,
    default_ttl: Duration,
}

struct BufferEntry {
    record: MemoryRecord,
    importance: f64,
    created_at: Instant,
    access_count: u64,
    ttl: Duration,
}

impl WorkingMemory {
    pub fn new(max_size: usize, default_ttl_secs: u64) -> Self {
        Self {
            buffer: Mutex::new(HashMap::with_capacity(max_size)),
            max_size,
            default_ttl: Duration::from_secs(default_ttl_secs),
        }
    }

    /// Safely acquires the lock, recovering from poisoning if necessary.
    /// This is the recommended way to access the buffer in all methods.
    fn lock_buffer(&self) -> std::sync::MutexGuard<'_, HashMap<String, BufferEntry>> {
        match self.buffer.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Insert a record into working memory.
    pub fn insert(&self, record: MemoryRecord, importance: f64, ttl_override: Option<Duration>) {
        let ttl = ttl_override.unwrap_or(self.default_ttl);
        let mut buffer = self.lock_buffer();

        // Evict if at capacity
        if buffer.len() >= self.max_size {
            self.evict_lru_locked(&mut buffer);
        }

        buffer.insert(
            record.id.clone(),
            BufferEntry {
                record,
                importance,
                created_at: Instant::now(),
                access_count: 0,
                ttl,
            },
        );
    }

    /// Get a record from working memory, updating access count.
    pub fn get(&self, id: &str) -> Option<MemoryRecord> {
        let mut buffer = self.lock_buffer();
        let entry = buffer.get_mut(id)?;

        entry.access_count += 1;

        if entry.created_at.elapsed() > entry.ttl {
            let _ = buffer.remove(id);
            return None;
        }

        Some(entry.record.clone())
    }

    /// Check if working memory contains an entry (read-only, no side effects).
    pub fn peek(&self, id: &str) -> bool {
        let buffer = self.lock_buffer();
        buffer
            .get(id)
            .is_some_and(|e| e.created_at.elapsed() <= e.ttl)
    }

    /// Check if a record exists and is not expired.
    pub fn contains(&self, id: &str) -> bool {
        let mut buffer = self.lock_buffer();
        match buffer.get_mut(id) {
            Some(entry) => {
                if entry.created_at.elapsed() > entry.ttl {
                    buffer.remove(id);
                    false
                } else {
                    entry.access_count += 1;
                    true
                }
            }
            None => false,
        }
    }

    /// Get all non-expired records ready for flushing to persistent storage.
    pub fn drain_expired(&self) -> Vec<(MemoryRecord, f64, u64)> {
        let now = Instant::now();
        let mut buffer = self.lock_buffer();

        let expired: Vec<String> = buffer
            .iter()
            .filter(|(_, e)| now.duration_since(e.created_at) > e.ttl)
            .map(|(k, _)| k.clone())
            .collect();

        let mut result = Vec::new();
        for key in expired {
            if let Some(entry) = buffer.remove(&key) {
                result.push((entry.record, entry.importance, entry.access_count));
            }
        }
        result
    }

    /// Drain all records (for forced flush).
    pub fn drain_all(&self) -> Vec<(MemoryRecord, f64, u64)> {
        let mut buffer = self.lock_buffer();
        buffer
            .drain()
            .map(|(_, entry)| (entry.record, entry.importance, entry.access_count))
            .collect()
    }

    /// Evict the least recently accessed entry (internal, assumes lock is held).
    fn evict_lru_locked(&self, buffer: &mut HashMap<String, BufferEntry>) {
        let oldest = buffer
            .iter()
            .min_by_key(|(_, e)| e.created_at)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest {
            buffer.remove(&key);
        }
    }

    /// Number of entries currently in working memory.
    pub fn len(&self) -> usize {
        let buffer = self.lock_buffer();
        buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        let buffer = self.lock_buffer();
        buffer.is_empty()
    }

    /// Purge all expired entries without returning them.
    pub fn purge_expired(&self) {
        let now = Instant::now();
        let mut buffer = self.lock_buffer();
        buffer.retain(|_, e| now.duration_since(e.created_at) <= e.ttl);
    }
}

// ── Promotion Engine ───────────────────────────────────────────────────────

/// Determines whether a record should be promoted, demoted, or left in place.
pub struct PromotionEngine {
    /// Per-tier configuration
    tier_configs: HashMap<MemoryTier, TierConfig>,
}

impl PromotionEngine {
    pub fn new() -> Self {
        let mut configs = HashMap::new();
        for tier in MemoryTier::all() {
            configs.insert(tier, TierConfig::for_tier(tier));
        }
        Self {
            tier_configs: configs,
        }
    }

    /// Check if a record should be promoted to the next tier.
    pub fn should_promote(&self, record: &TieredRecord) -> Option<MemoryTier> {
        let config = self.tier_configs.get(&record.tier)?;
        if !config.auto_promote {
            return None;
        }
        if record.importance >= config.promotion_threshold {
            record.tier.promote_to()
        } else {
            None
        }
    }

    /// Check if a record should be demoted to the previous tier.
    pub fn should_demote(&self, record: &TieredRecord) -> Option<MemoryTier> {
        let config = self.tier_configs.get(&record.tier)?;
        if record.importance < config.demotion_threshold {
            record.tier.demote_to()
        } else {
            None
        }
    }

    /// Get the configuration for a specific tier.
    pub fn get_config(&self, tier: MemoryTier) -> TierConfig {
        self.tier_configs
            .get(&tier)
            .cloned()
            .unwrap_or_else(|| TierConfig::for_tier(tier))
    }

    /// Update configuration for a tier.
    pub fn set_config(&mut self, tier: MemoryTier, config: TierConfig) {
        self.tier_configs.insert(tier, config);
    }
}

impl Default for PromotionEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tiered Memory Orchestrator ─────────────────────────────────────────────

/// High-level orchestrator for hierarchical memory operations.
pub struct TieredMemory {
    pub store: MemoryStore,
    pub working: WorkingMemory,
    pub promotion: PromotionEngine,
}

impl TieredMemory {
    /// Open a tiered memory system backed by the given store config.
    pub fn open(config: &crate::types::StorageConfig) -> rusqlite::Result<Self> {
        let store = MemoryStore::open(config)?;
        Ok(Self {
            store,
            working: WorkingMemory::new(100, 3600), // 100 items, 1 hour TTL
            promotion: PromotionEngine::new(),
        })
    }

    /// Insert a record into the appropriate tier.
    /// Working memory records go to the in-memory buffer; others go to SQLite.
    pub fn insert(
        &mut self,
        record: MemoryRecord,
        tier: MemoryTier,
        importance: f64,
    ) -> rusqlite::Result<()> {
        let ttl = self.promotion.get_config(tier).default_ttl_seconds;

        match tier {
            MemoryTier::Working => {
                let ttl_duration = ttl.map(Duration::from_secs);
                self.working.insert(record, importance, ttl_duration);
                Ok(())
            }
            _ => self
                .store
                .insert_into_tier(&record, tier, importance, ttl, None),
        }
    }

    /// Get a record from any tier, starting with working memory.
    pub fn get(&mut self, id: &str) -> rusqlite::Result<Option<TieredRecord>> {
        // Check working memory first (peek for existence, then get for the record)
        if self.working.peek(id) {
            if let Some(record) = self.working.get(id) {
                return Ok(Some(TieredRecord {
                    record,
                    tier: MemoryTier::Working,
                    importance: 0.5,
                    access_count: 1,
                    last_accessed: chrono::Utc::now().to_rfc3339(),
                    ttl_seconds: Some(3600),
                    parent_id: None,
                    tags: vec![],
                }));
            }
        }

        // Fall through to persistent store
        self.store.get_tiered(id)
    }

    /// Search across all tiers (working memory + persistent).
    pub fn search(&mut self, query: &str, limit: usize) -> rusqlite::Result<Vec<SearchResult>> {
        // Search FTS in persistent store
        let mut results = self.store.search_fts(query, limit)?;

        // Also search working memory
        let query_lower = query.to_lowercase();
        let buffer = self.working.lock_buffer();
        for entry in buffer.values() {
            if entry.record.content.to_lowercase().contains(&query_lower)
                || entry.record.id.to_lowercase().contains(&query_lower)
            {
                results.push((
                    entry.record.clone(),
                    0.5, // moderate score for working memory matches
                ));
            }
        }

        // Deduplicate and sort
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results
            .into_iter()
            .map(|(record, score)| SearchResult {
                record,
                score: (1.0 / (1.0 + score.abs())).clamp(0.0, 1.0),
                method: "fts".into(),
            })
            .collect())
    }

    /// Promote a record to the next tier.
    pub fn promote(&self, id: &str) -> rusqlite::Result<bool> {
        if let Some(tiered) = self.store.get_tiered(id)? {
            if let Some(next_tier) = tiered.tier.promote_to() {
                return self.store.promote(id, next_tier);
            }
        }
        Ok(false)
    }

    /// Demote a record to the previous tier.
    pub fn demote(&self, id: &str) -> rusqlite::Result<bool> {
        if let Some(tiered) = self.store.get_tiered(id)? {
            if let Some(prev_tier) = tiered.tier.demote_to() {
                return self.store.promote(id, prev_tier);
            }
        }
        Ok(false)
    }

    /// Flush expired working memory entries to the Episodic tier.
    pub fn flush_working_memory(&mut self) -> rusqlite::Result<u64> {
        let expired = self.working.drain_expired();
        let mut count = 0u64;

        for (record, importance, access_count) in &expired {
            // Boost importance slightly based on access count during working memory lifetime
            let adjusted_importance = (*importance + (*access_count as f64 * 0.05)).min(1.0);
            self.store.insert_into_tier(
                record,
                MemoryTier::Episodic,
                adjusted_importance,
                None,
                None,
            )?;
            count += 1;
        }

        Ok(count)
    }

    /// Flush ALL working memory entries to the Episodic tier.
    pub fn flush_all_working(&mut self) -> rusqlite::Result<u64> {
        let all = self.working.drain_all();
        let mut count = 0u64;

        for (record, importance, access_count) in &all {
            let adjusted_importance = (*importance + (*access_count as f64 * 0.05)).min(1.0);
            self.store.insert_into_tier(
                record,
                MemoryTier::Episodic,
                adjusted_importance,
                None,
                None,
            )?;
            count += 1;
        }

        Ok(count)
    }

    /// Run auto-promotion: evaluate all records in each tier and promote if eligible.
    pub fn run_auto_promotion(&self, tier: MemoryTier) -> rusqlite::Result<u64> {
        let records = self.store.list_by_tier(tier, 1000, 0)?;
        let mut promoted = 0u64;

        for tiered in &records {
            if let Some(next_tier) = self.promotion.should_promote(tiered) {
                if self
                    .store
                    .promote(&tiered.record.id, next_tier)
                    .unwrap_or(false)
                {
                    promoted += 1;
                }
            }
        }

        Ok(promoted)
    }

    /// Run auto-demotion: demote records that fall below the threshold.
    pub fn run_auto_demotion(&self, tier: MemoryTier) -> rusqlite::Result<u64> {
        let records = self.store.list_by_tier(tier, 1000, 0)?;
        let mut demoted = 0u64;

        for tiered in &records {
            if let Some(prev_tier) = self.promotion.should_demote(tiered) {
                if self
                    .store
                    .promote(&tiered.record.id, prev_tier)
                    .unwrap_or(false)
                {
                    demoted += 1;
                }
            }
        }

        Ok(demoted)
    }

    /// Evict low-importance records from a tier to stay within capacity.
    pub fn evict_from_tier(&self, tier: MemoryTier) -> rusqlite::Result<u64> {
        let config = self.promotion.get_config(tier);
        self.store.evict_from_tier(tier, config.max_records)
    }

    /// Get aggregate statistics across all tiers.
    pub fn stats(&self) -> rusqlite::Result<crate::types::MemoryStats> {
        let mut stats = self.store.stats()?;
        stats.total_records += self.working.len() as u64;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StorageConfig;

    #[test]
    fn test_working_memory_basic() {
        let wm = WorkingMemory::new(5, 3600);
        let record = MemoryRecord::new("w1".into(), "Working test".into(), "test".into());

        wm.insert(record.clone(), 0.8, None);
        assert!(wm.contains("w1"));
        assert_eq!(wm.len(), 1);
    }

    #[test]
    fn test_working_memory_lru_eviction() {
        let wm = WorkingMemory::new(2, 3600);
        wm.insert(
            MemoryRecord::new("a".into(), "A".into(), "test".into()),
            0.5,
            None,
        );
        wm.insert(
            MemoryRecord::new("b".into(), "B".into(), "test".into()),
            0.5,
            None,
        );
        wm.insert(
            MemoryRecord::new("c".into(), "C".into(), "test".into()),
            0.5,
            None,
        );

        assert_eq!(wm.len(), 2); // one was evicted
        assert!(!wm.contains("a")); // 'a' was oldest, evicted
    }

    #[test]
    fn test_tiered_memory_integration() {
        let config = StorageConfig::default();
        let mut tm = TieredMemory::open(&config).unwrap();

        let record = MemoryRecord::new(
            "tm1".into(),
            "Tiered integration test".into(),
            "test".into(),
        );
        tm.insert(record, MemoryTier::Episodic, 0.7).unwrap();

        let retrieved = tm.get("tm1").unwrap().expect("Should exist");
        assert_eq!(retrieved.tier, MemoryTier::Episodic);
    }

    #[test]
    fn test_promotion_engine() {
        let engine = PromotionEngine::new();

        let record = TieredRecord {
            record: MemoryRecord::new("p1".into(), "Test".into(), "test".into()),
            tier: MemoryTier::Episodic,
            importance: 0.9, // high importance, should promote
            access_count: 5,
            last_accessed: chrono::Utc::now().to_rfc3339(),
            ttl_seconds: None,
            parent_id: None,
            tags: vec![],
        };

        let promotion = engine.should_promote(&record);
        assert_eq!(promotion, Some(MemoryTier::Semantic));

        // Low importance should demote
        let low_record = TieredRecord {
            importance: 0.1,
            ..record
        };
        let demotion = engine.should_demote(&low_record);
        assert_eq!(demotion, Some(MemoryTier::Working));
    }

    #[test]
    fn test_tiered_insert_and_promote() {
        let config = StorageConfig::default();
        let tm = TieredMemory::open(&config).unwrap();

        let record = MemoryRecord::new("promo1".into(), "Promote me".into(), "test".into());
        tm.store
            .insert_into_tier(&record, MemoryTier::Episodic, 0.9, None, None)
            .unwrap();

        // Promote from episodic to semantic
        let promoted = tm.promote("promo1").unwrap();
        assert!(promoted);

        let tiered = tm
            .store
            .get_tiered("promo1")
            .unwrap()
            .expect("Should exist");
        assert_eq!(tiered.tier, MemoryTier::Semantic);
    }

    #[test]
    fn test_flush_working_memory() {
        let config = StorageConfig::default();
        let mut tm = TieredMemory::open(&config).unwrap();

        let record = MemoryRecord::new("flush1".into(), "Will be flushed".into(), "test".into());
        tm.insert(record, MemoryTier::Working, 0.6).unwrap();

        let flushed = tm.flush_all_working().unwrap();
        assert_eq!(flushed, 1);

        // Should now be in episodic tier
        let tiered = tm
            .store
            .get_tiered("flush1")
            .unwrap()
            .expect("Should exist in store");
        assert_eq!(tiered.tier, MemoryTier::Episodic);
    }
}
