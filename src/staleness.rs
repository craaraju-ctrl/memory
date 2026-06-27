//! Staleness Management System
//! Prevents outdated memories from dominating retrieval results through temporal decay.

use crate::types::TieredRecord;
use chrono::{DateTime, Utc};

/// Configuration for staleness behavior
#[derive(Debug, Clone)]
pub struct StalenessConfig {
    pub decay_rate_per_day: f64,
    pub max_age_days: u32,
    pub validation_threshold: f64,
    pub confidence_decay_per_day: f64,
}

impl Default for StalenessConfig {
    fn default() -> Self {
        Self {
            decay_rate_per_day: 0.028,
            max_age_days: 120,
            validation_threshold: 0.35,
            confidence_decay_per_day: 0.012,
        }
    }
}

/// Manages staleness calculations
pub struct StalenessManager {
    config: StalenessConfig,
}

impl StalenessManager {
    pub fn new(config: StalenessConfig) -> Self {
        Self { config }
    }

    /// Calculate effective score after applying temporal decay
    pub fn effective_score(&self, record: &TieredRecord) -> f64 {
        let age_days = self.calculate_age_days(&record.record.timestamp);

        let time_decay = (1.0 - self.config.decay_rate_per_day).powf(age_days);
        let confidence_decay = (1.0 - self.config.confidence_decay_per_day * age_days).max(0.15);

        let base = record.importance * time_decay * confidence_decay;

        if age_days > self.config.max_age_days as f64 {
            base * 0.22
        } else {
            base
        }
    }

    fn calculate_age_days(&self, timestamp: &str) -> f64 {
        if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp) {
            let now = Utc::now();
            (now - dt.with_timezone(&Utc)).num_days() as f64
        } else {
            60.0
        }
    }

    pub fn needs_validation(&self, record: &TieredRecord) -> bool {
        self.effective_score(record) < self.config.validation_threshold
    }

    pub fn apply_update(
        &self,
        record: &mut TieredRecord,
        new_importance: f64,
        new_timestamp: Option<String>,
    ) {
        record.importance = (record.importance * 0.55 + new_importance * 0.45).clamp(0.1, 1.0);
        if let Some(ts) = new_timestamp {
            record.record.timestamp = ts;
        }
    }
}

impl Default for StalenessManager {
    fn default() -> Self {
        Self::new(StalenessConfig::default())
    }
}
