//! # Expert Modules — Specialized Memory Operations
//!
//! Provides expert modules for retrieval, reasoning, consolidation, and evolution.

use crate::store::MemoryStore;
use crate::types::{MemoryRecord, SearchResult};
use crate::vector::cosine_similarity;

/// Expert specialized in hybrid memory retrieval (vector + graph + staleness).
pub struct RetrievalExpert {
    pub store: MemoryStore,
}

impl RetrievalExpert {
    pub fn new(store: MemoryStore) -> Self {
        Self { store }
    }

    // ==================== HYBRID RETRIEVAL ====================

    /// Hybrid retrieval: vector search + graph boosting.
    /// Combines sqlite-vec k-NN with graph-based relevance boosting.
    pub fn hybrid_retrieve(
        &self,
        query_vec: &[f64],
        k: usize,
        candidate_multiplier: usize,
    ) -> rusqlite::Result<Vec<SearchResult>> {
        if query_vec.is_empty() || k == 0 {
            return Ok(vec![]);
        }

        let num_candidates = k * candidate_multiplier.max(1);
        let candidates = self.store.search_vectors_hybrid(query_vec, num_candidates, num_candidates)?;

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let mut scored = self.dense_rerank(query_vec, &candidates);
        self.boost_with_graph_reasoning(&mut scored, 0.18)?;

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored)
    }

    /// Re-rank candidates using full-precision cosine similarity.
    fn dense_rerank(&self, query_vec: &[f64], candidates: &[(MemoryRecord, f32)]) -> Vec<SearchResult> {
        candidates
            .iter()
            .filter_map(|(record, _)| {
                let emb = record.embedding.as_ref()?;
                let score = (cosine_similarity(query_vec, emb) + 1.0) / 2.0;
                Some(SearchResult {
                    record: record.clone(),
                    score: score.clamp(0.0, 1.0),
                    method: "hybrid_binary_dense".to_string(),
                })
            })
            .collect()
    }

    /// Boost search results based on graph connectivity and relationship strength.
    fn boost_with_graph_reasoning(
        &self,
        results: &mut [SearchResult],
        boost_weight: f64,
    ) -> rusqlite::Result<()> {
        for result in results.iter_mut() {
            let edges = self.store.get_edges(&result.record.id).unwrap_or_default();
            if edges.is_empty() {
                continue;
            }

            let mut score = 0.0;
            for edge in &edges {
                let type_weight = match edge.relation_type.as_str() {
                    "causes" | "depends_on" | "leads_to" => 1.7,
                    "part_of" | "requires" => 1.4,
                    _ => 1.0,
                };
                score += edge.weight * type_weight;
            }

            let mut two_hop = 0.0;
            for edge in &edges {
                if let Ok(neighbors) = self.store.get_edges(&edge.target_id) {
                    two_hop += neighbors.len() as f64 * 0.3;
                }
            }

            let combined = (score + two_hop).min(35.0);
            let boost = (combined / 35.0) * boost_weight;
            result.score = (result.score + boost).clamp(0.0, 1.0);
        }
        Ok(())
    }
}
