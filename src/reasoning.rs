//! # Reasoning Engine — Chain-of-Thought / Chain-of-Reasoning
//!
//! Provides structured storage and retrieval of reasoning chains,
//! enabling agents to record, search, and reuse past reasoning patterns.
//!
//! ## Key Concepts
//!
//! - **ReasoningStep**: Atomic unit of reasoning (premise → inference → conclusion)
//! - **ReasoningChain**: Complete sequence of steps toward a goal
//! - **Retrieval**: Find past chains by goal similarity or tag matching
//! - **Insight Extraction**: Distill reusable patterns from successful chains

use chrono::Utc;

use crate::store::MemoryStore;
use crate::types::{ReasoningChain, ReasoningStep};

/// The reasoning engine manages chain-of-thought storage and retrieval.
pub struct ReasoningEngine {
    store: MemoryStore,
}

impl ReasoningEngine {
    pub fn new(store: MemoryStore) -> Self {
        Self { store }
    }

    /// Start a new reasoning chain for a goal.
    pub fn start_chain(&self, goal: &str, tags: Vec<String>) -> ReasoningChain {
        let now = Utc::now().to_rfc3339();
        ReasoningChain {
            chain_id: format!("chain_{}", uuid_v4()),
            goal: goal.to_string(),
            steps: Vec::new(),
            final_conclusion: None,
            overall_confidence: 0.0,
            success: false,
            consulted_records: Vec::new(),
            tags,
            created_at: now,
            duration_ms: 0,
        }
    }

    /// Add a step to a reasoning chain and save to store.
    pub fn add_step(
        &self,
        chain_id: &str,
        premise: &str,
        inference: &str,
        tool_used: Option<&str>,
    ) -> Result<ReasoningChain, String> {
        let mut chain = self
            .store
            .get_reasoning_chain(chain_id)
            .map_err(|e| format!("Failed to get chain: {}", e))?
            .ok_or_else(|| format!("Chain {} not found", chain_id))?;

        let now = Utc::now().to_rfc3339();
        let step = ReasoningStep {
            step_index: chain.steps.len() as u32,
            premise: premise.to_string(),
            inference: inference.to_string(),
            conclusion: String::new(), // to be filled after step completes
            confidence: 0.0,
            tool_used: tool_used.map(|s| s.to_string()),
            success: true,
            timestamp: now,
        };

        chain.steps.push(step);
        self.store
            .store_reasoning_chain(&chain)
            .map_err(|e| format!("Failed to save chain: {}", e))?;

        Ok(chain)
    }

    /// Complete a step with its conclusion and confidence.
    pub fn complete_step(
        &self,
        chain_id: &str,
        step_index: u32,
        conclusion: &str,
        confidence: f64,
        success: bool,
    ) -> Result<ReasoningChain, String> {
        let mut chain = self
            .store
            .get_reasoning_chain(chain_id)
            .map_err(|e| format!("Failed to get chain: {}", e))?
            .ok_or_else(|| format!("Chain {} not found", chain_id))?;

        if let Some(step) = chain.steps.get_mut(step_index as usize) {
            step.conclusion = conclusion.to_string();
            step.confidence = confidence;
            step.success = success;
        }

        self.store
            .store_reasoning_chain(&chain)
            .map_err(|e| format!("Failed to save chain: {}", e))?;

        Ok(chain)
    }

    /// Finalize a reasoning chain with the overall conclusion.
    pub fn finalize_chain(
        &self,
        chain_id: &str,
        final_conclusion: &str,
        overall_confidence: f64,
        success: bool,
        consulted_records: Vec<String>,
        duration_ms: u64,
    ) -> Result<ReasoningChain, String> {
        let mut chain = self
            .store
            .get_reasoning_chain(chain_id)
            .map_err(|e| format!("Failed to get chain: {}", e))?
            .ok_or_else(|| format!("Chain {} not found", chain_id))?;

        chain.final_conclusion = Some(final_conclusion.to_string());
        chain.overall_confidence = overall_confidence;
        chain.success = success;
        chain.consulted_records = consulted_records;
        chain.duration_ms = duration_ms;

        self.store
            .store_reasoning_chain(&chain)
            .map_err(|e| format!("Failed to save chain: {}", e))?;

        Ok(chain)
    }

    /// Get a reasoning chain by ID.
    pub fn get_chain(&self, chain_id: &str) -> rusqlite::Result<Option<ReasoningChain>> {
        self.store.get_reasoning_chain(chain_id)
    }

    /// Search reasoning chains by goal similarity.
    pub fn search_chains(
        &self,
        query: &str,
        limit: usize,
    ) -> rusqlite::Result<Vec<ReasoningChain>> {
        self.store.search_reasoning_chains(query, limit)
    }

    /// Find past chains that successfully achieved a similar goal.
    pub fn find_similar_successful_chains(
        &self,
        goal: &str,
        limit: usize,
    ) -> rusqlite::Result<Vec<ReasoningChain>> {
        let all = self.store.search_reasoning_chains(goal, limit)?;
        Ok(all.into_iter().filter(|c| c.success).collect())
    }

    /// Extract reusable insights from a reasoning chain.
    pub fn extract_insights(&self, chain_id: &str) -> Result<Vec<String>, String> {
        let chain = self
            .store
            .get_reasoning_chain(chain_id)
            .map_err(|e| format!("Failed to get chain: {}", e))?
            .ok_or_else(|| format!("Chain {} not found", chain_id))?;

        let mut insights = Vec::new();

        // Extract the goal as an insight if successful
        if chain.success {
            insights.push(format!(
                "Goal '{}' achieved with confidence {:.2}",
                chain.goal, chain.overall_confidence
            ));
        }

        // Extract tool usage patterns
        let tool_usage: Vec<&str> = chain
            .steps
            .iter()
            .filter_map(|s| s.tool_used.as_deref())
            .collect();
        if !tool_usage.is_empty() {
            insights.push(format!("Tools used: {}", tool_usage.join(", ")));
        }

        // Extract key reasoning pattern
        let successful_steps: Vec<&ReasoningStep> =
            chain.steps.iter().filter(|s| s.success).collect();
        if successful_steps.len() >= 2 {
            insights.push(format!(
                "Reasoning pattern: {} → ... → {} ({} steps)",
                successful_steps
                    .first()
                    .map(|s| &s.premise)
                    .unwrap_or(&"".to_string()),
                successful_steps
                    .last()
                    .map(|s| &s.conclusion)
                    .unwrap_or(&"".to_string()),
                successful_steps.len()
            ));
        }

        Ok(insights)
    }

    /// Build a prompt-friendly summary of a reasoning chain.
    pub fn format_chain_as_prompt(&self, chain: &ReasoningChain) -> String {
        let mut parts = Vec::new();

        parts.push(format!("Goal: {}", chain.goal));
        parts.push(format!("Confidence: {:.2}", chain.overall_confidence));
        parts.push(format!("Success: {}", chain.success));

        for step in &chain.steps {
            parts.push(format!(
                "\nStep {}:\n  Premise: {}\n  Inference: {}\n  Conclusion: {}\n  Confidence: {:.2}",
                step.step_index + 1,
                step.premise,
                step.inference,
                step.conclusion,
                step.confidence,
            ));
        }

        if let Some(ref conclusion) = chain.final_conclusion {
            parts.push(format!("\nFinal: {}", conclusion));
        }

        parts.join("\n")
    }

    /// Mark a reasoning chain as a lesson learned (store in semantic memory for reuse).
    pub fn distill_to_semantic(&self, chain_id: &str) -> Result<String, String> {
        let chain = self
            .store
            .get_reasoning_chain(chain_id)
            .map_err(|e| format!("Failed to get chain: {}", e))?
            .ok_or_else(|| format!("Chain {} not found", chain_id))?;

        if !chain.success {
            return Err("Cannot distill an unsuccessful chain".to_string());
        }

        let insights = self.extract_insights(chain_id)?;
        let content = format!(
            "Procedure: {}\n\nPattern: {}\n\nInsights:\n{}",
            chain.goal,
            self.format_chain_as_prompt(&chain),
            insights.join("\n"),
        );

        let record = crate::types::MemoryRecord::new(
            format!("procedure_{}", chain_id),
            content,
            "procedure".to_string(),
        )
        .with_metadata("source_chain", chain_id)
        .with_metadata("confidence", &format!("{:.2}", chain.overall_confidence));

        let record_id = record.id.clone();
        self.store
            .insert(&record)
            .map_err(|e| format!("Failed to store distilled procedure: {}", e))?;

        Ok(record_id)
    }
}

/// Simple UUID without external dependency.
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
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
    use crate::types::StorageConfig;

    fn setup() -> ReasoningEngine {
        let config = StorageConfig::default();
        let store = MemoryStore::open(&config).unwrap();
        ReasoningEngine::new(store)
    }

    #[test]
    fn test_start_chain() {
        let engine = setup();
        let chain = engine.start_chain("Test goal", vec!["test".into()]);
        assert_eq!(chain.goal, "Test goal");
        assert!(chain.steps.is_empty());
    }

    #[test]
    fn test_full_chain_workflow() {
        let engine = setup();

        // Start
        let chain = engine.start_chain("Analyze market", vec!["market".into(), "analysis".into()]);
        let chain_id = chain.chain_id.clone();
        engine.store.store_reasoning_chain(&chain).unwrap();

        // Add steps
        engine
            .add_step(
                &chain_id,
                "Price increased 10%",
                "Check volume",
                Some("volume_analyzer"),
            )
            .unwrap();
        engine
            .complete_step(&chain_id, 0, "Volume confirms trend", 0.85, true)
            .unwrap();

        engine
            .add_step(
                &chain_id,
                "Trend confirmed",
                "Check resistance levels",
                Some("resistance_checker"),
            )
            .unwrap();
        engine
            .complete_step(&chain_id, 1, "Resistance at $105", 0.75, true)
            .unwrap();

        // Finalize
        let finalized = engine
            .finalize_chain(
                &chain_id,
                "Bullish trend, target $105",
                0.85,
                true,
                vec!["r1".into()],
                2500,
            )
            .unwrap();
        assert_eq!(finalized.steps.len(), 2);
        assert!(finalized.success);

        // Retrieve
        let retrieved = engine.get_chain(&chain_id).unwrap().expect("Should exist");
        assert_eq!(
            retrieved.final_conclusion.unwrap(),
            "Bullish trend, target $105"
        );
    }

    #[test]
    fn test_search_chains() {
        let engine = setup();

        let chain1 = engine.start_chain("Bitcoin price analysis", vec!["crypto".into()]);
        engine.store.store_reasoning_chain(&chain1).unwrap();

        let chain2 = engine.start_chain("Stock market analysis", vec!["stocks".into()]);
        engine.store.store_reasoning_chain(&chain2).unwrap();

        let results = engine.search_chains("bitcoin", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].goal, "Bitcoin price analysis");
    }

    #[test]
    fn test_extract_insights() {
        let engine = setup();

        let chain = engine.start_chain("Test insight", vec![]);
        let chain_id = chain.chain_id.clone();
        engine.store.store_reasoning_chain(&chain).unwrap();

        engine
            .add_step(&chain_id, "Step 1", "Do thing", Some("tool_a"))
            .unwrap();
        engine
            .complete_step(&chain_id, 0, "Done", 0.9, true)
            .unwrap();
        engine
            .finalize_chain(&chain_id, "Success", 0.9, true, vec![], 100)
            .unwrap();

        let insights = engine.extract_insights(&chain_id).unwrap();
        assert!(!insights.is_empty());
    }
}
