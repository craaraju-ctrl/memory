//! # Memory Store — SQLite-Backed Hierarchical Memory Storage
//!
//! Extended with tier-aware storage, knowledge graph, reasoning chains,
//! expert opinions, evolution tracking, and bitemporal metadata.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use rusqlite::{params, Connection};

use crate::types::{
    GraphEdge, GraphTraversalResult, MemoryRecord, MemoryStats, MemoryTier,
    ReasoningChain, ReasoningStep, StorageConfig, TierConfig,
    TierStats, TieredRecord,
};

/// Initialize the sqlite-vec extension globally via sqlite3_auto_extension.
/// Safe to call multiple times — subsequent calls are no-ops at the SQLite level.
pub fn init_vector_search() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(
                Some(std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ())),
            );
        }
    });
}

/// SQLite-backed store with tier-aware, graph, reasoning, and vector search support.
#[derive(Debug, Clone)]
pub struct MemoryStore {
    pub(crate) conn: Arc<Mutex<Connection>>,
    vector_dimension: usize,
    vector_search_enabled: Arc<AtomicBool>,
}

impl MemoryStore {
    /// Acquire the database connection lock.
    /// Converts a poisoned mutex into a `rusqlite::Error` instead of panicking.
    ///
    /// **WARNING:** `std::sync::Mutex` is not re-entrant. Any method that holds
    /// the returned `MutexGuard` must NOT call other methods that acquire `lock_db()`.
    /// Use block scopes `{ let conn = self.lock_db()?; ... }` to drop the guard
    /// before calling lock-acquiring methods, or use `get_tier_config_with_conn()`
    /// variants that accept an already-held `&Connection`.
    fn lock_db(&self) -> rusqlite::Result<MutexGuard<'_, Connection>> {
        self.conn.lock().map_err(|e| {
            rusqlite::Error::InvalidParameterName(format!("Database mutex poisoned: {}", e))
        })
    }

    pub fn open(config: &StorageConfig) -> rusqlite::Result<Self> {
        init_vector_search();

        let conn = if config.db_path == ":memory:" {
            Connection::open_in_memory()?
        } else {
            Connection::open(&config.db_path)?
        };

        // Production SQLite pragmas: WAL for concurrent reads, busy_timeout to
        // prevent SQLITE_BUSY errors under contention, NORMAL sync for perf.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;
             PRAGMA foreign_keys=ON;
             PRAGMA temp_store=MEMORY;",
        )?;

        let dim = config.vector_dimension.clamp(64, 4096);

        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            vector_dimension: dim,
            vector_search_enabled: Arc::new(AtomicBool::new(false)),
        };
        store.initialize_tables()?;
        Ok(store)
    }

    fn initialize_tables(&self) -> rusqlite::Result<()> {
        // Use a block scope to drop the MutexGuard before calling run_migrations,
        // which also needs to acquire the same lock.
        {
            let conn = self.lock_db()?;
            conn.execute_batch(
                "
                CREATE TABLE IF NOT EXISTS records (
                    id            TEXT PRIMARY KEY,
                    content       TEXT NOT NULL,
                    content_type  TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding     BLOB,
                    timestamp     TEXT NOT NULL,

                    tier          TEXT NOT NULL DEFAULT 'working',
                    importance    REAL NOT NULL DEFAULT 0.5,
                    access_count  INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT,
                    ttl_seconds   INTEGER,
                    parent_id     TEXT,

                    valid_from    TEXT,
                    valid_to      TEXT,
                    sys_start     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    sys_end       TEXT,
                    tags_json     TEXT NOT NULL DEFAULT '[]'
                );

                CREATE INDEX IF NOT EXISTS idx_records_tier     ON records(tier);
                CREATE INDEX IF NOT EXISTS idx_records_type     ON records(content_type);
                CREATE INDEX IF NOT EXISTS idx_records_ts       ON records(timestamp);
                CREATE INDEX IF NOT EXISTS idx_records_parent   ON records(parent_id);
                CREATE INDEX IF NOT EXISTS idx_records_importance ON records(importance);

                CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
                    id UNINDEXED, content, content_type UNINDEXED, metadata_json UNINDEXED
                );

                CREATE TABLE IF NOT EXISTS tier_config (
                    tier              TEXT PRIMARY KEY,
                    max_records       INTEGER NOT NULL DEFAULT 1000,
                    default_ttl_secs  INTEGER,
                    promotion_threshold REAL NOT NULL DEFAULT 0.7,
                    demotion_threshold  REAL NOT NULL DEFAULT 0.2,
                    auto_promote      INTEGER NOT NULL DEFAULT 1
                );

                INSERT OR IGNORE INTO tier_config VALUES ('working',    100,   3600,        0.5, 0.1, 1);
                INSERT OR IGNORE INTO tier_config VALUES ('episodic',   10000, 2592000,     0.7, 0.2, 1);
                INSERT OR IGNORE INTO tier_config VALUES ('semantic',   100000,NULL,        0.85,0.15,0);
                INSERT OR IGNORE INTO tier_config VALUES ('procedural', 10000, NULL,        0.95,0.1, 0);

                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_id       TEXT PRIMARY KEY,
                    source_id     TEXT NOT NULL,
                    target_id     TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight        REAL NOT NULL DEFAULT 1.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    FOREIGN KEY (source_id) REFERENCES records(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES records(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id);
                CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id);
                CREATE INDEX IF NOT EXISTS idx_edges_relation ON graph_edges(relation_type);

                CREATE TABLE IF NOT EXISTS reasoning_chains (
                    chain_id          TEXT PRIMARY KEY,
                    goal              TEXT NOT NULL,
                    steps_json        TEXT NOT NULL DEFAULT '[]',
                    final_conclusion  TEXT,
                    overall_confidence REAL NOT NULL DEFAULT 0.0,
                    success           INTEGER NOT NULL DEFAULT 0,
                    consulted_records TEXT NOT NULL DEFAULT '[]',
                    tags_json         TEXT NOT NULL DEFAULT '[]',
                    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    duration_ms       INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_chains_goal ON reasoning_chains(goal);

                CREATE TABLE IF NOT EXISTS expert_opinions (
                    opinion_id        TEXT PRIMARY KEY,
                    expert_type       TEXT NOT NULL,
                    target_record_id  TEXT,
                    recommendation    TEXT NOT NULL,
                    reasoning         TEXT NOT NULL,
                    confidence        REAL NOT NULL DEFAULT 0.0,
                    action_taken      TEXT,
                    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    FOREIGN KEY (target_record_id) REFERENCES records(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_opinions_expert ON expert_opinions(expert_type);
                CREATE INDEX IF NOT EXISTS idx_opinions_target ON expert_opinions(target_record_id);

                CREATE TABLE IF NOT EXISTS evolution_events (
                    event_id       TEXT PRIMARY KEY,
                    event_type     TEXT NOT NULL,
                    description    TEXT NOT NULL,
                    previous_value TEXT,
                    new_value      TEXT,
                    confidence     REAL NOT NULL DEFAULT 0.0,
                    timestamp      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_evolution_type ON evolution_events(event_type);
                ",
            )?;

            let _ = self.init_vector_tables(&conn);
        } // MutexGuard dropped here

        // Run schema migrations — acquires its own lock
        self.run_migrations()?;

        Ok(())
    }

    /// Run database schema migrations.
    /// This ensures the database schema is always up-to-date when the application starts.
    fn run_migrations(&self) -> rusqlite::Result<()> {
        let conn = self.lock_db()?;

        // Create migration tracking table if it doesn't exist
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     INTEGER PRIMARY KEY,
                applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );
            "
        )?;

        // Get current migration version
        let current_version: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        // Migration v1: Add tags_json column to records table (for future tag support)
        if current_version < 1 {
            // Check if column already exists (defensive)
            let has_tags: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('records') WHERE name = 'tags_json'",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0) > 0;

            if !has_tags {
                conn.execute_batch(
                    "ALTER TABLE records ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]';"
                )?;
            }

            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version) VALUES (1)",
                [],
            )?;
        }

        // Future migrations can be added here easily:
        // if current_version < 2 { ... }

        Ok(())
    }

    fn init_vector_tables(&self, conn: &Connection) -> rusqlite::Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS vector_map (
                record_id TEXT PRIMARY KEY,
                vec_rowid INTEGER NOT NULL
            );"
        )?;

        let dim = self.vector_dimension;
        let sql = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vectors_ann USING vec0(
                embedding float[{}] distance_metric=cosine
            )",
            dim
        );

        match conn.execute_batch(&sql) {
            Ok(()) => {
                self.vector_search_enabled.store(true, Ordering::Relaxed);
                eprintln!("✅ sqlite-vec: vector search initialized (dim={})", dim);
                Ok(())
            }
            Err(e) => {
                eprintln!("ℹ️  sqlite-vec not available: {} — using fallback linear scan", e);
                Ok(())
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  TIER-AWARE CRUD
    // ══════════════════════════════════════════════════════════════════════

    pub fn insert_into_tier(
        &self,
        record: &MemoryRecord,
        tier: MemoryTier,
        importance: f64,
        ttl_seconds: Option<u64>,
        parent_id: Option<&str>,
    ) -> rusqlite::Result<()> {
        // Scope the lock so it's released before calling store_embedding,
        // which also acquires the lock.
        {
            let mut conn = self.lock_db()?;
            
            // Use a transaction for atomicity across records + FTS
            let tx = conn.transaction()?;

            let metadata_json = serde_json::to_string(&record.metadata).unwrap_or_default();
            let embedding_blob: Option<Vec<u8>> = record
                .embedding
                .as_ref()
                .map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect());

            let now = chrono::Utc::now().to_rfc3339();

            // 1. Insert into main records table
            tx.execute(
                "INSERT OR REPLACE INTO records
                 (id, content, content_type, metadata_json, embedding, timestamp,
                  tier, importance, access_count, last_accessed, ttl_seconds, parent_id,
                  valid_from, sys_start)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, ?10, ?11, ?9, ?9)",
                params![
                    record.id,
                    record.content,
                    record.content_type,
                    metadata_json,
                    embedding_blob,
                    record.timestamp,
                    tier.to_string(),
                    importance,
                    now,
                    ttl_seconds,
                    parent_id,
                ],
            )?;

            // 2. Insert into FTS index
            tx.execute(
                "INSERT OR REPLACE INTO records_fts (id, content, content_type, metadata_json)
                 VALUES (?1, ?2, ?3, ?4)",
                params![record.id, record.content, record.content_type, metadata_json],
            )?;

            // Commit the main transaction (records + FTS)
            tx.commit()?;
        } // MutexGuard dropped here — safe to call store_embedding now

        // 3. Handle vector embedding with compensation for data integrity
        if let Some(ref emb) = record.embedding {
            if let Err(e) = self.store_embedding(&record.id, emb) {
                tracing::error!("Vector embedding failed for {}. Attempting compensation delete.", record.id);
                let _ = self.delete(&record.id);
                return Err(e);
            }
        }

        Ok(())
    }

    pub fn insert(&self, record: &MemoryRecord) -> rusqlite::Result<()> {
        self.insert_into_tier(record, MemoryTier::Episodic, 0.5, None, None)
    }

    pub fn get_tiered(&self, id: &str) -> rusqlite::Result<Option<TieredRecord>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT id, content, content_type, metadata_json, embedding, timestamp,
                    tier, importance, access_count, last_accessed, ttl_seconds, parent_id
             FROM records WHERE id = ?1",
        )?;

        let mut rows = stmt.query_map(params![id], row_to_tiered_record)?;
        match rows.next() {
            Some(Ok(record)) => {
                // Best-effort update of access stats. Failure here should not fail the get.
                if let Err(e) = conn.execute(
                    "UPDATE records SET access_count = access_count + 1, last_accessed = ?1 WHERE id = ?2",
                    params![chrono::Utc::now().to_rfc3339(), id],
                ) {
                    tracing::warn!("Failed to update access stats for {}: {}", id, e);
                }
                Ok(Some(record))
            }
            _ => Ok(None),
        }
    }

    pub fn get(&self, id: &str) -> rusqlite::Result<Option<MemoryRecord>> {
        self.get_tiered(id).map(|opt| opt.map(|t| t.record))
    }

    pub fn delete(&self, id: &str) -> rusqlite::Result<bool> {
        let mut conn = self.lock_db()?;
        
        // Wrap multi-step delete in a transaction for data integrity
        let tx = conn.transaction()?;

        // 1. Get vector rowid if exists
        let rowid_opt: Option<i64> = tx.query_row(
            "SELECT vec_rowid FROM vector_map WHERE record_id = ?1",
            params![id],
            |r| r.get(0),
        ).ok();

        if let Some(vec_rowid) = rowid_opt {
            let _ = tx.execute("DELETE FROM vectors_ann WHERE rowid = ?1", params![vec_rowid]);
            let _ = tx.execute("DELETE FROM vector_map WHERE record_id = ?1", params![id]);
        }

        // 2. Delete from main tables
        let deleted = tx.execute("DELETE FROM records WHERE id = ?1", params![id])?;
        let _ = tx.execute("DELETE FROM records_fts WHERE id = ?1", params![id]);
        let _ = tx.execute("DELETE FROM graph_edges WHERE source_id = ?1 OR target_id = ?1", params![id]);

        tx.commit()?;
        Ok(deleted > 0)
    }

    pub fn list_by_type(&self, content_type: &str, limit: usize, offset: usize) -> rusqlite::Result<Vec<MemoryRecord>> {
        self.list_by_type_tiered(content_type, limit, offset)
            .map(|v| v.into_iter().map(|t| t.record).collect())
    }

    pub fn list_by_type_tiered(&self, content_type: &str, limit: usize, offset: usize) -> rusqlite::Result<Vec<TieredRecord>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT id, content, content_type, metadata_json, embedding, timestamp,
                    tier, importance, access_count, last_accessed, ttl_seconds, parent_id
             FROM records WHERE content_type = ?1
             ORDER BY importance DESC, timestamp DESC LIMIT ?2 OFFSET ?3",
        )?;

        let rows = stmt.query_map(params![content_type, limit as i64, offset as i64], row_to_tiered_record)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn list_by_tier(&self, tier: MemoryTier, limit: usize, offset: usize) -> rusqlite::Result<Vec<TieredRecord>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT id, content, content_type, metadata_json, embedding, timestamp,
                    tier, importance, access_count, last_accessed, ttl_seconds, parent_id
             FROM records WHERE tier = ?1
             ORDER BY importance DESC, timestamp DESC LIMIT ?2 OFFSET ?3",
        )?;

        let rows = stmt.query_map(params![tier.to_string(), limit as i64, offset as i64], row_to_tiered_record)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn all_with_embeddings(&self) -> rusqlite::Result<Vec<MemoryRecord>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT id, content, content_type, metadata_json, embedding, timestamp
             FROM records WHERE embedding IS NOT NULL
             ORDER BY timestamp DESC",
        )?;

        let rows = stmt.query_map([], row_to_record)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn all(&self, limit: usize, offset: usize) -> rusqlite::Result<Vec<MemoryRecord>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT id, content, content_type, metadata_json, embedding, timestamp
             FROM records ORDER BY timestamp DESC LIMIT ?1 OFFSET ?2",
        )?;

        let rows = stmt.query_map(params![limit as i64, offset as i64], row_to_record)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Promotion / Demotion ─────────────────────────────────────────────

    pub fn promote(&self, id: &str, to_tier: MemoryTier) -> rusqlite::Result<bool> {
        let conn = self.lock_db()?;
        let updated = conn.execute(
            "UPDATE records SET tier = ?1 WHERE id = ?2",
            params![to_tier.to_string(), id],
        )?;
        Ok(updated > 0)
    }

    pub fn update_importance(&self, id: &str, importance: f64) -> rusqlite::Result<bool> {
        let conn = self.lock_db()?;
        let updated = conn.execute(
            "UPDATE records SET importance = ?1 WHERE id = ?2",
            params![importance, id],
        )?;
        Ok(updated > 0)
    }

    pub fn get_eviction_candidates(&self, tier: MemoryTier, count: usize) -> rusqlite::Result<Vec<TieredRecord>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT id, content, content_type, metadata_json, embedding, timestamp,
                    tier, importance, access_count, last_accessed, ttl_seconds, parent_id
             FROM records WHERE tier = ?1
             ORDER BY importance ASC, access_count ASC, timestamp ASC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![tier.to_string(), count as i64], row_to_tiered_record)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn evict_from_tier(&self, tier: MemoryTier, max_to_keep: usize) -> rusqlite::Result<u64> {
        let conn = self.lock_db()?;
        let tier_str = tier.to_string();

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM records WHERE tier = ?1",
            params![tier_str],
            |r| r.get(0),
        )?;

        if count <= max_to_keep as i64 {
            return Ok(0);
        }

        let to_evict = (count - max_to_keep as i64) as u64;

        conn.execute(
            "DELETE FROM records WHERE id IN (
                SELECT id FROM records WHERE tier = ?1
                ORDER BY importance ASC, access_count ASC, timestamp ASC
                LIMIT ?2
            )",
            params![tier_str, to_evict as i64],
        )?;

        conn.execute("DELETE FROM records_fts WHERE id NOT IN (SELECT id FROM records)", [])?;
        Ok(to_evict)
    }

    // ── Full-Text Search ─────────────────────────────────────────────────

    pub fn search_fts(&self, query: &str, limit: usize) -> rusqlite::Result<Vec<(MemoryRecord, f64)>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT r.id, r.content, r.content_type, r.metadata_json, r.embedding, r.timestamp,
                    rank
             FROM records_fts f
             JOIN records r ON r.id = f.id
             WHERE records_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![query, limit as i64], |row| {
            let record = row_to_record(row)?;
            let rank: f64 = row.get(6)?;
            Ok((record, rank))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn search_fts_in_tier(
        &self,
        query: &str,
        tier: MemoryTier,
        limit: usize,
    ) -> rusqlite::Result<Vec<(MemoryRecord, f64)>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT r.id, r.content, r.content_type, r.metadata_json, r.embedding, r.timestamp,
                    rank
             FROM records_fts f
             JOIN records r ON r.id = f.id
             WHERE records_fts MATCH ?1 AND r.tier = ?2
             ORDER BY rank
             LIMIT ?3",
        )?;

        let rows = stmt.query_map(params![query, tier.to_string(), limit as i64], |row| {
            let record = row_to_record(row)?;
            let rank: f64 = row.get(6)?;
            Ok((record, rank))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Tier Configuration ───────────────────────────────────────────────

    /// Read tier config from an already-held connection (no extra lock).
    fn get_tier_config_with_conn(&self, conn: &Connection, tier: MemoryTier) -> TierConfig {
        conn.query_row(
            "SELECT max_records, default_ttl_secs, promotion_threshold, demotion_threshold, auto_promote
             FROM tier_config WHERE tier = ?1",
            params![tier.to_string()],
            |row| {
                Ok(TierConfig {
                    max_records: row.get(0)?,
                    default_ttl_seconds: row.get(1)?,
                    promotion_threshold: row.get(2)?,
                    demotion_threshold: row.get(3)?,
                    auto_promote: row.get::<_, i32>(4)? != 0,
                })
            },
        ).unwrap_or_else(|_| TierConfig::for_tier(tier))
    }

    pub fn get_tier_config(&self, tier: MemoryTier) -> rusqlite::Result<TierConfig> {
        let conn = self.lock_db()?;
        Ok(self.get_tier_config_with_conn(&conn, tier))
    }

    pub fn update_tier_config(&self, tier: MemoryTier, config: &TierConfig) -> rusqlite::Result<()> {
        let conn = self.lock_db()?;
        conn.execute(
            "INSERT OR REPLACE INTO tier_config
             (tier, max_records, default_ttl_secs, promotion_threshold, demotion_threshold, auto_promote)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                tier.to_string(),
                config.max_records as i64,
                config.default_ttl_seconds.map(|s| s as i64),
                config.promotion_threshold,
                config.demotion_threshold,
                config.auto_promote as i32,
            ],
        )?;
        Ok(())
    }

    // ══════════════════════════════════════════════════════════════════════
    //  KNOWLEDGE GRAPH OPERATIONS
    // ══════════════════════════════════════════════════════════════════════

    pub fn add_edge(&self, source_id: &str, target_id: &str, relation_type: &str, weight: f64) -> rusqlite::Result<String> {
        let conn = self.lock_db()?;
        let edge_id = format!("edge_{}", uuid_v4());
        conn.execute(
            "INSERT INTO graph_edges (edge_id, source_id, target_id, relation_type, weight)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![edge_id, source_id, target_id, relation_type, weight],
        )?;
        Ok(edge_id)
    }

    pub fn get_edges(&self, record_id: &str) -> rusqlite::Result<Vec<GraphEdge>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT edge_id, source_id, target_id, relation_type, weight, metadata_json, created_at
             FROM graph_edges WHERE source_id = ?1 OR target_id = ?1
             ORDER BY weight DESC",
        )?;

        let rows = stmt.query_map(params![record_id], row_to_edge)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn graph_bfs(&self, start_id: &str, max_depth: u32, relation_filter: Option<&str>) -> rusqlite::Result<Vec<GraphTraversalResult>> {
        let conn = self.lock_db()?;

        if max_depth == 0 {
            return Ok(vec![GraphTraversalResult {
                node_id: start_id.to_string(),
                depth: 0,
                path: vec![start_id.to_string()],
                cumulative_weight: 1.0,
            }]);
        }

        let relation_clause = match relation_filter {
            Some(rel) => format!("AND e.relation_type = '{}'", rel.replace('\'', "''")),
            None => String::new(),
        };

        let sql = format!(
            "WITH RECURSIVE traversal(node_id, depth, path, cum_weight) AS (
                SELECT ?1, 0, ?1, 1.0
                UNION
                SELECT e.target_id, t.depth + 1,
                       t.path || ',' || e.target_id,
                       t.cum_weight * e.weight
                FROM traversal t
                JOIN graph_edges e ON e.source_id = t.node_id
                WHERE t.depth < ?2 {}
            )
            SELECT DISTINCT node_id, depth, path, cum_weight FROM traversal
            ORDER BY depth, cum_weight DESC",
            relation_clause
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![start_id, max_depth], |row| {
            Ok(GraphTraversalResult {
                node_id: row.get(0)?,
                depth: row.get(1)?,
                path: row.get::<_, String>(2)?.split(',').map(|s| s.to_string()).collect(),
                cumulative_weight: row.get(3)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn get_related_records(
        &self,
        record_id: &str,
        relation_type: Option<&str>,
        max_depth: u32,
    ) -> rusqlite::Result<Vec<(MemoryRecord, u32, String, f64)>> {
        let traversal = self.graph_bfs(record_id, max_depth, relation_type)?;

        let mut results = Vec::new();
        for t in &traversal {
            if t.node_id == record_id { continue; }
            if let Some(record) = self.get(&t.node_id)? {
                let path_str = t.path.join(" → ");
                results.push((record, t.depth, path_str, t.cumulative_weight));
            }
        }
        Ok(results)
    }

    // ══════════════════════════════════════════════════════════════════════
    //  REASONING CHAINS
    // ══════════════════════════════════════════════════════════════════════

    pub fn store_reasoning_chain(&self, chain: &ReasoningChain) -> rusqlite::Result<()> {
        let conn = self.lock_db()?;
        let steps_json = serde_json::to_string(&chain.steps).unwrap_or_default();
        let consulted_json = serde_json::to_string(&chain.consulted_records).unwrap_or_default();
        let tags_json = serde_json::to_string(&chain.tags).unwrap_or_default();

        conn.execute(
            "INSERT OR REPLACE INTO reasoning_chains
             (chain_id, goal, steps_json, final_conclusion, overall_confidence, success,
              consulted_records, tags_json, created_at, duration_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                chain.chain_id,
                chain.goal,
                steps_json,
                chain.final_conclusion,
                chain.overall_confidence,
                chain.success as i32,
                consulted_json,
                tags_json,
                chain.created_at,
                chain.duration_ms as i64,
            ],
        )?;
        Ok(())
    }

    pub fn get_reasoning_chain(&self, chain_id: &str) -> rusqlite::Result<Option<ReasoningChain>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT chain_id, goal, steps_json, final_conclusion, overall_confidence, success,
                    consulted_records, tags_json, created_at, duration_ms
             FROM reasoning_chains WHERE chain_id = ?1",
        )?;

        let mut rows = stmt.query_map(params![chain_id], row_to_reasoning_chain)?;
        match rows.next() {
            Some(Ok(chain)) => Ok(Some(chain)),
            _ => Ok(None),
        }
    }

    pub fn search_reasoning_chains(&self, goal_query: &str, limit: usize) -> rusqlite::Result<Vec<ReasoningChain>> {
        let conn = self.lock_db()?;
        let pattern = format!("%{}%", goal_query);
        let mut stmt = conn.prepare(
            "SELECT chain_id, goal, steps_json, final_conclusion, overall_confidence, success,
                    consulted_records, tags_json, created_at, duration_ms
             FROM reasoning_chains
             WHERE goal LIKE ?1 OR final_conclusion LIKE ?1
             ORDER BY overall_confidence DESC, created_at DESC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![pattern, limit as i64], row_to_reasoning_chain)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ══════════════════════════════════════════════════════════════════════
    //  EXPERT OPINIONS
    // ══════════════════════════════════════════════════════════════════════

    pub fn store_opinion(
        &self,
        opinion_id: &str,
        expert_type: &str,
        target_record_id: Option<&str>,
        recommendation: &str,
        reasoning: &str,
        confidence: f64,
        action_taken: Option<&str>,
    ) -> rusqlite::Result<()> {
        let conn = self.lock_db()?;
        conn.execute(
            "INSERT INTO expert_opinions (opinion_id, expert_type, target_record_id, recommendation, reasoning, confidence, action_taken)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![opinion_id, expert_type, target_record_id, recommendation, reasoning, confidence, action_taken],
        )?;
        Ok(())
    }

    pub fn get_opinions_by_expert(&self, expert_type: &str, limit: usize) -> rusqlite::Result<Vec<(String, String, f64, String)>> {
        let conn = self.lock_db()?;
        let mut stmt = conn.prepare(
            "SELECT recommendation, reasoning, confidence, created_at
             FROM expert_opinions WHERE expert_type = ?1
             ORDER BY confidence DESC, created_at DESC LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![expert_type, limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ══════════════════════════════════════════════════════════════════════
    //  EVOLUTION EVENTS
    // ══════════════════════════════════════════════════════════════════════

    pub fn record_evolution_event(
        &self,
        event_id: &str,
        event_type: &str,
        description: &str,
        previous_value: Option<&str>,
        new_value: Option<&str>,
        confidence: f64,
    ) -> rusqlite::Result<()> {
        let conn = self.lock_db()?;
        conn.execute(
            "INSERT INTO evolution_events (event_id, event_type, description, previous_value, new_value, confidence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![event_id, event_type, description, previous_value, new_value, confidence],
        )?;
        Ok(())
    }

    pub fn get_evolution_events(&self, event_type: Option<&str>, limit: usize) -> rusqlite::Result<Vec<(String, String, f64, String)>> {
        let conn = self.lock_db()?;
        let (sql, type_param): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(et) = event_type {
            (
                "SELECT event_type, description, confidence, timestamp FROM evolution_events
                 WHERE event_type = ?1 ORDER BY timestamp DESC LIMIT ?2".to_string(),
                vec![Box::new(et.to_string()), Box::new(limit as i64)],
            )
        } else {
            (
                "SELECT event_type, description, confidence, timestamp FROM evolution_events
                 ORDER BY timestamp DESC LIMIT ?1".to_string(),
                vec![Box::new(limit as i64)],
            )
        };

        let mut stmt = conn.prepare(&sql)?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> = type_param.iter().map(|b| b.as_ref()).collect();
        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ══════════════════════════════════════════════════════════════════════
    //  STATISTICS
    // ══════════════════════════════════════════════════════════════════════

    pub fn stats(&self) -> rusqlite::Result<MemoryStats> {
        let conn = self.lock_db()?;

        let total: i64 = conn.query_row("SELECT COUNT(*) FROM records", [], |r| r.get(0)).unwrap_or(0);
        let with_embeddings: i64 = conn
            .query_row("SELECT COUNT(*) FROM records WHERE embedding IS NOT NULL", [], |r| r.get(0))
            .unwrap_or(0);

        let mut stmt = conn.prepare(
            "SELECT content_type, COUNT(*) FROM records GROUP BY content_type",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
        })?;
        let mut content_types = HashMap::new();
        for row in rows {
            let (ct, cnt) = row?;
            content_types.insert(ct, cnt);
        }

        let page_count: i64 = conn.query_row("PRAGMA page_count", [], |r| r.get(0)).unwrap_or(0);
        let page_size: i64 = conn.query_row("PRAGMA page_size", [], |r| r.get(0)).unwrap_or(0);

        // Batch-fetch all tier configs in one query (avoids N+1 pattern)
        let mut all_tier_configs: HashMap<String, TierConfig> = HashMap::new();
        {
            let mut cfg_stmt = conn.prepare(
                "SELECT tier, max_records, default_ttl_secs, promotion_threshold, demotion_threshold, auto_promote
                 FROM tier_config",
            )?;
            let cfg_rows = cfg_stmt.query_map([], |row| {
                let tier_name: String = row.get(0)?;
                Ok((tier_name, TierConfig {
                    max_records: row.get(1)?,
                    default_ttl_seconds: row.get(2)?,
                    promotion_threshold: row.get(3)?,
                    demotion_threshold: row.get(4)?,
                    auto_promote: row.get::<_, i32>(5)? != 0,
                }))
            })?;
            for row in cfg_rows {
                let (name, cfg) = row?;
                all_tier_configs.insert(name, cfg);
            }
        }

        let mut tier_breakdown = HashMap::new();
        for tier_str in ["working", "episodic", "semantic", "procedural"] {
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM records WHERE tier = ?1", params![tier_str], |r| r.get(0))
                .unwrap_or(0);
            let emb: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM records WHERE tier = ?1 AND embedding IS NOT NULL",
                    params![tier_str],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            let avg_imp: f64 = conn
                .query_row(
                    "SELECT COALESCE(AVG(importance), 0.0) FROM records WHERE tier = ?1",
                    params![tier_str],
                    |r| r.get(0),
                )
                .unwrap_or(0.0);
            let accesses: i64 = conn
                .query_row(
                    "SELECT COALESCE(SUM(access_count), 0) FROM records WHERE tier = ?1",
                    params![tier_str],
                    |r| r.get(0),
                )
                .unwrap_or(0);

            let tier = match tier_str {
                "working" => MemoryTier::Working,
                "episodic" => MemoryTier::Episodic,
                "semantic" => MemoryTier::Semantic,
                "procedural" => MemoryTier::Procedural,
                _ => unreachable!(),
            };

            let config = all_tier_configs
                .remove(tier_str)
                .unwrap_or_else(|| TierConfig::for_tier(tier));

            tier_breakdown.insert(
                tier_str.to_string(),
                TierStats {
                    tier,
                    total_records: count as u64,
                    total_with_embeddings: emb as u64,
                    average_importance: avg_imp,
                    total_accesses: accesses as u64,
                    storage_bytes: 0,
                    config,
                },
            );
        }

        Ok(MemoryStats {
            total_records: total as u64,
            total_with_embeddings: with_embeddings as u64,
            content_types,
            storage_bytes: (page_count * page_size) as u64,
            tier_breakdown,
        })
    }

    // ══════════════════════════════════════════════════════════════════════
    //  VECTOR SEARCH (sqlite-vec)
    // ══════════════════════════════════════════════════════════════════════

    pub fn store_embedding(&self, record_id: &str, embedding: &[f64]) -> rusqlite::Result<()> {
        if !self.vector_search_enabled.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mut conn = self.lock_db()?;
        
        // Wrap vector operations in a transaction
        let tx = conn.transaction()?;

        let old_rowid: Option<i64> = tx.query_row(
            "SELECT vec_rowid FROM vector_map WHERE record_id = ?1",
            params![record_id],
            |r| r.get::<_, i64>(0),
        ).ok();

        if let Some(old_rowid) = old_rowid {
            let _ = tx.execute("DELETE FROM vectors_ann WHERE rowid = ?1", params![old_rowid]);
            let _ = tx.execute("DELETE FROM vector_map WHERE record_id = ?1", params![record_id]);
        }

        let f32_bytes: Vec<u8> = embedding
            .iter()
            .map(|&v| v as f32)
            .flat_map(|f| f.to_le_bytes())
            .collect();

        tx.execute(
            "INSERT INTO vectors_ann(embedding) VALUES (?1)",
            params![f32_bytes],
        )?;

        let vec_rowid = tx.last_insert_rowid();

        tx.execute(
            "INSERT INTO vector_map(record_id, vec_rowid) VALUES (?1, ?2)",
            params![record_id, vec_rowid],
        )?;

        tx.commit()?;
        Ok(())
    }

    pub fn search_vectors(&self, query: &[f64], k: usize) -> rusqlite::Result<Vec<(MemoryRecord, f32)>> {
        if !self.vector_search_enabled.load(Ordering::Relaxed) {
            return Ok(Vec::new());
        }

        let conn = self.lock_db()?;

        let f32_bytes: Vec<u8> = query
            .iter()
            .map(|&v| v as f32)
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let mut stmt = conn.prepare(
            "SELECT r.id, r.content, r.content_type, r.metadata_json, r.embedding, r.timestamp,
                    v.distance
             FROM (
                 SELECT rowid, distance
                 FROM vectors_ann
                 WHERE embedding MATCH ?1
                 ORDER BY distance
                 LIMIT ?2
             ) v
             JOIN vector_map m ON m.vec_rowid = v.rowid
             JOIN records r ON r.id = m.record_id
             ORDER BY v.distance",
        )?;

        let rows = stmt.query_map(params![f32_bytes, k as i64], |row| {
            let record = row_to_record(row)?;
            let distance: f32 = row.get(6)?;
            Ok((record, distance))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }

        Ok(results)
    }

    pub fn search_vectors_hybrid(
        &self,
        query: &[f64],
        k: usize,
        top_n: usize,
    ) -> rusqlite::Result<Vec<(MemoryRecord, f32)>> {
        if !self.vector_search_enabled.load(Ordering::Relaxed) {
            return Ok(Vec::new());
        }
        let candidates = self.search_vectors(query, top_n)?;

        if candidates.is_empty() || candidates.len() <= k {
            return Ok(candidates.into_iter().take(k).collect());
        }

        let query_binary = crate::vector::quantize_binary(query);

        let mut reranked: Vec<(f64, MemoryRecord, f32)> = candidates
            .into_iter()
            .filter_map(|(record, cosine_dist)| {
                let emb = record.embedding.as_ref()?;
                let bin = crate::vector::quantize_binary(emb);
                let hamming_sim = crate::vector::hamming_similarity(&query_binary, &bin);
                let combined = hamming_sim - (cosine_dist as f64) / 2.0;
                Some((combined, record, cosine_dist))
            })
            .collect();

        reranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(reranked.into_iter().take(k).map(|(_, rec, dist)| (rec, dist)).collect())
    }

    pub fn clear(&self) -> rusqlite::Result<()> {
        let conn = self.lock_db()?;
        conn.execute_batch(
            "DELETE FROM records;
             DELETE FROM records_fts;
             DELETE FROM graph_edges;
             DELETE FROM reasoning_chains;
             DELETE FROM expert_opinions;
             DELETE FROM evolution_events;
             DELETE FROM vectors_ann;
             DELETE FROM vector_map;",
        )?;
        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  ROW MAPPING FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════

/// Convert a blob of f64 little-endian bytes into a Vec<f64>.
/// Returns None if the blob length is not a multiple of 8 (malformed data).
fn blob_to_f64_vec(blob: Vec<u8>) -> Option<Vec<f64>> {
    if !blob.len().is_multiple_of(8) {
        return None;
    }
    Some(
        blob.chunks_exact(8)
            .map(|chunk| {
                let arr: [u8; 8] = chunk.try_into().expect("chunks_exact(8) guarantees 8 bytes");
                f64::from_le_bytes(arr)
            })
            .collect(),
    )
}

fn row_to_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
    let embedding_blob: Option<Vec<u8>> = row.get(4)?;
    let embedding = embedding_blob.and_then(blob_to_f64_vec);

    let metadata_json: String = row.get(3)?;
    let metadata = serde_json::from_str(&metadata_json).unwrap_or_default();

    Ok(MemoryRecord {
        id: row.get(0)?,
        content: row.get(1)?,
        content_type: row.get(2)?,
        metadata,
        embedding,
        timestamp: row.get(5)?,
    })
}

fn row_to_tiered_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<TieredRecord> {
    let embedding_blob: Option<Vec<u8>> = row.get(4)?;
    let embedding = embedding_blob.and_then(blob_to_f64_vec);

    let metadata_json: String = row.get(3)?;
    let metadata = serde_json::from_str(&metadata_json).unwrap_or_default();

    let tier_str: String = row.get(6)?;
    let tier = tier_str.parse::<MemoryTier>().unwrap_or(MemoryTier::Episodic);

    Ok(TieredRecord {
        record: MemoryRecord {
            id: row.get(0)?,
            content: row.get(1)?,
            content_type: row.get(2)?,
            metadata,
            embedding,
            timestamp: row.get(5)?,
        },
        tier,
        importance: row.get(7)?,
        access_count: row.get(8)?,
        last_accessed: row.get(9)?,
        ttl_seconds: row.get(10)?,
        parent_id: row.get(11)?,
        tags: {
            // Robust tag deserialization with logging on failure
            match row.get::<_, Option<String>>(12) {
                Ok(Some(json)) if !json.is_empty() => {
                    match serde_json::from_str(&json) {
                        Ok(tags) => tags,
                        Err(e) => {
                            tracing::warn!("Failed to deserialize tags_json for record: {}", e);
                            vec![]
                        }
                    }
                }
                _ => vec![],
            }
        },
    })
}

fn row_to_edge(row: &rusqlite::Row<'_>) -> rusqlite::Result<GraphEdge> {
    let metadata_json: String = row.get(5)?;
    let metadata = serde_json::from_str(&metadata_json).unwrap_or_default();

    Ok(GraphEdge {
        edge_id: row.get(0)?,
        source_id: row.get(1)?,
        target_id: row.get(2)?,
        relation_type: row.get(3)?,
        weight: row.get(4)?,
        metadata,
        created_at: row.get(6)?,
    })
}

fn row_to_reasoning_chain(row: &rusqlite::Row<'_>) -> rusqlite::Result<ReasoningChain> {
    let steps_json: String = row.get(2)?;
    let steps: Vec<ReasoningStep> = serde_json::from_str(&steps_json).unwrap_or_default();

    let consulted_json: String = row.get(6)?;
    let consulted_records: Vec<String> = serde_json::from_str(&consulted_json).unwrap_or_default();

    let tags_json: String = row.get(7)?;
    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

    Ok(ReasoningChain {
        chain_id: row.get(0)?,
        goal: row.get(1)?,
        steps,
        final_conclusion: row.get(3)?,
        overall_confidence: row.get(4)?,
        success: row.get::<_, i32>(5)? != 0,
        consulted_records,
        tags,
        created_at: row.get(8)?,
        duration_ms: row.get(9)?,
    })
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        now.as_secs(),
        (now.as_nanos() & 0xffff) as u16,
        ((now.as_nanos() >> 16) & 0xfff) as u16,
        ((now.as_nanos() >> 28) & 0xffff) as u16,
        (now.as_nanos() >> 44) as u64 & 0xffffffffffff
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_insert_and_get() {
        let config = StorageConfig::default();
        let store = MemoryStore::open(&config).unwrap();
        let record = MemoryRecord::new("tier-test-1".into(), "Tiered content".into(), "test".into());
        store.insert_into_tier(&record, MemoryTier::Working, 0.8, Some(3600), None).unwrap();
        let tiered = store.get_tiered("tier-test-1").unwrap().expect("Should exist");
        assert_eq!(tiered.tier, MemoryTier::Working);
        assert!((tiered.importance - 0.8).abs() < 0.001);
        assert_eq!(tiered.ttl_seconds, Some(3600));
    }

    #[test]
    fn test_list_by_tier() {
        let config = StorageConfig::default();
        let store = MemoryStore::open(&config).unwrap();
        for i in 0..3 {
            let r = MemoryRecord::new(format!("e{}", i), format!("Episodic {}", i), "tier_test".into());
            store.insert_into_tier(&r, MemoryTier::Episodic, 0.5, None, None).unwrap();
        }
        for i in 0..2 {
            let r = MemoryRecord::new(format!("s{}", i), format!("Semantic {}", i), "tier_test".into());
            store.insert_into_tier(&r, MemoryTier::Semantic, 0.9, None, None).unwrap();
        }
        let episodic = store.list_by_tier(MemoryTier::Episodic, 10, 0).unwrap();
        assert_eq!(episodic.len(), 3);
        let semantic = store.list_by_tier(MemoryTier::Semantic, 10, 0).unwrap();
        assert_eq!(semantic.len(), 2);
    }

    #[test]
    fn test_graph_edges() {
        let config = StorageConfig::default();
        let store = MemoryStore::open(&config).unwrap();
        let r1 = MemoryRecord::new("g1".into(), "Node A".into(), "graph".into());
        let r2 = MemoryRecord::new("g2".into(), "Node B".into(), "graph".into());
        store.insert(&r1).unwrap();
        store.insert(&r2).unwrap();
        store.add_edge("g1", "g2", "related_to", 0.9).unwrap();
        let edges = store.get_edges("g1").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relation_type, "related_to");
        let related = store.get_related_records("g1", None, 1).unwrap();
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].0.id, "g2");
    }

    #[test]
    fn test_reasoning_chain() {
        let config = StorageConfig::default();
        let store = MemoryStore::open(&config).unwrap();
        let chain = ReasoningChain {
            chain_id: "chain-1".into(),
            goal: "Analyze market trend".into(),
            steps: vec![ReasoningStep {
                step_index: 0,
                premise: "Price increased 10%".into(),
                inference: "Check volume".into(),
                conclusion: "Volume confirms trend".into(),
                confidence: 0.85,
                tool_used: Some("volume_analyzer".into()),
                success: true,
                timestamp: chrono::Utc::now().to_rfc3339(),
            }],
            final_conclusion: Some("Bullish trend confirmed".into()),
            overall_confidence: 0.85,
            success: true,
            consulted_records: vec!["r1".into(), "r2".into()],
            tags: vec!["market".into(), "analysis".into()],
            created_at: chrono::Utc::now().to_rfc3339(),
            duration_ms: 1500,
        };
        store.store_reasoning_chain(&chain).unwrap();
        let retrieved = store.get_reasoning_chain("chain-1").unwrap().expect("Should exist");
        assert_eq!(retrieved.goal, "Analyze market trend");
        assert_eq!(retrieved.steps.len(), 1);
        assert_eq!(retrieved.tags.len(), 2);
    }

    #[test]
    fn test_eviction() {
        let config = StorageConfig::default();
        let store = MemoryStore::open(&config).unwrap();
        for i in 0..5 {
            let r = MemoryRecord::new(format!("evict{}", i), format!("Low importance {}", i), "evict".into());
            store.insert_into_tier(&r, MemoryTier::Episodic, 0.1 * (i as f64), None, None).unwrap();
        }
        let evicted = store.evict_from_tier(MemoryTier::Episodic, 3).unwrap();
        assert_eq!(evicted, 2);
        let remaining = store.list_by_tier(MemoryTier::Episodic, 10, 0).unwrap();
        assert_eq!(remaining.len(), 3);
    }

    #[test]
    fn test_stats_with_tiers() {
        let config = StorageConfig::default();
        let store = MemoryStore::open(&config).unwrap();
        let r = MemoryRecord::new("s1".into(), "Stats test".into(), "stats_test".into());
        store.insert_into_tier(&r, MemoryTier::Working, 0.5, Some(60), None).unwrap();
        let stats = store.stats().unwrap();
        assert!(stats.total_records >= 1);
        assert!(stats.tier_breakdown.contains_key("working"));
    }

    #[test]
    fn test_full_text_search() {
        let config = StorageConfig::default();
        let store = MemoryStore::open(&config).unwrap();
        store.insert(&MemoryRecord::new("fts1".into(), "Bitcoin hits all time high".into(), "news".into())).unwrap();
        store.insert(&MemoryRecord::new("fts2".into(), "Ethereum merge completed".into(), "news".into())).unwrap();
        let results = store.search_fts("bitcoin", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|(r, _)| r.id == "fts1"));
        let tier_results = store.search_fts_in_tier("bitcoin", MemoryTier::Episodic, 10).unwrap();
        assert!(!tier_results.is_empty());
    }
}
