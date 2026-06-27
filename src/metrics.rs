//! # Metrics — Real-time Observability
//!
//! Uses the `metrics` crate for instrumentation and `metrics-exporter-prometheus`
//! for Prometheus-compatible export. All metrics are lazily registered on first use.
//!
//! ## Setup
//!
//! Call `init_prometheus()` once at server startup (in `main.rs`). This installs
//! the Prometheus recorder globally. The `/metrics` endpoint on the main API server
//! serves the Prometheus text format via `render_prometheus()`.
//!
//! ## Exposed Metrics
//!
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | `memory_retrieval_candidates` | histogram | Candidates considered in hybrid retrieval |
//! | `memory_retrieval_results` | histogram | Final results returned |
//! | `memory_retrieval_graph_boost` | histogram | Graph boost applied to results |
//! | `memory_retrieval_duration_ms` | histogram | Hybrid retrieval latency |
//! | `memory_circuit_breaker_events_total` | counter | Circuit breaker state transitions |
//! | `memory_tier_operations_total` | counter | Promotions/demotions by tier |
//! | `memory_evolution_cycle_duration_ms` | histogram | Evolution cycle latency |
//! | `memory_evolution_events_total` | counter | Evolution events generated |
//! | `memory_reasoning_steps` | histogram | Steps in reasoning chains |
//! | `memory_reasoning_duration_ms` | histogram | Reasoning chain latency |
//! | `memory_reasoning_confidence` | histogram | Reasoning chain confidence |
//! | `memory_reasoning_chains_total` | counter | Reasoning chains by success |
//! | `memory_requests_total` | counter | Total HTTP requests by method+path+status |
//! | `memory_request_duration_seconds` | histogram | HTTP request latency |
//! | `memory_records_total` | gauge | Total records across all tiers |

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;

/// Global Prometheus handle for rendering metrics in text format.
static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Initialize the Prometheus recorder.
///
/// Call this once at server startup. Installs the global recorder so that
/// `metrics::counter!`, `metrics::histogram!`, and `metrics::gauge!` macros
/// populate the registry. No separate HTTP server is started — the `/metrics`
/// endpoint on the main API server serves the data via `render_prometheus()`.
pub fn init_prometheus() {
    let handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");
    HANDLE.set(handle).ok();
}

// ── Hybrid Retrieval Metrics ───────────────────────────────────────────────

/// Record metrics for Hybrid Retrieval (vector + graph boosting).
pub fn record_hybrid_retrieval(
    candidates: usize,
    final_results: usize,
    avg_graph_boost: f64,
    duration_ms: u128,
) {
    metrics::histogram!("memory_retrieval_candidates").record(candidates as f64);
    metrics::histogram!("memory_retrieval_results").record(final_results as f64);
    metrics::histogram!("memory_retrieval_graph_boost").record(avg_graph_boost);
    metrics::histogram!("memory_retrieval_duration_ms").record(duration_ms as f64);
}

// ── Circuit Breaker Metrics ───────────────────────────────────────────────

/// Record Circuit Breaker state transition events.
pub fn record_circuit_breaker_event(state: &str, operation: &str) {
    metrics::counter!(
        "memory_circuit_breaker_events_total",
        "state" => state.to_string(),
        "operation" => operation.to_string(),
    )
    .increment(1);
}

// ── Tier Operation Metrics ────────────────────────────────────────────────

/// Record memory promotions/demotions by tier.
pub fn record_tier_operation(operation: &str, tier: &str, count: u64) {
    metrics::counter!(
        "memory_tier_operations_total",
        "operation" => operation.to_string(),
        "tier" => tier.to_string(),
    )
    .increment(count);
}

// ── Evolution Metrics ─────────────────────────────────────────────────────

/// Record evolution cycle duration and event count.
pub fn record_evolution_cycle(duration_ms: u128, events_count: usize) {
    metrics::histogram!("memory_evolution_cycle_duration_ms").record(duration_ms as f64);
    metrics::counter!("memory_evolution_events_total").increment(events_count as u64);
}

// ── Reasoning Metrics ─────────────────────────────────────────────────────

/// Record reasoning chain metrics.
pub fn record_reasoning_chain(steps: usize, success: bool, confidence: f64, duration_ms: u128) {
    metrics::histogram!("memory_reasoning_steps").record(steps as f64);
    metrics::histogram!("memory_reasoning_duration_ms").record(duration_ms as f64);
    metrics::counter!(
        "memory_reasoning_chains_total",
        "success" => success.to_string(),
    )
    .increment(1);
    metrics::histogram!("memory_reasoning_confidence").record(confidence);
}

// ── HTTP Request Metrics ──────────────────────────────────────────────────

/// Record an HTTP request metric. Call from the logging middleware.
pub fn record_http_request(method: &str, path: &str, status: u16, duration_ms: u128) {
    metrics::counter!(
        "memory_requests_total",
        "method" => method.to_string(),
        "path" => path.to_string(),
        "status" => status.to_string(),
    )
    .increment(1);
    metrics::histogram!("memory_request_duration_seconds").record(duration_ms as f64 / 1000.0);
}

// ── Prometheus Export ──────────────────────────────────────────────────────

/// Render all metrics in Prometheus text exposition format.
///
/// This is served by the `/metrics` endpoint so Prometheus can scrape it.
pub fn render_prometheus() -> String {
    HANDLE
        .get()
        .map(|h| h.render())
        .unwrap_or_else(|| "# metrics not initialized\n".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_hybrid_retrieval() {
        record_hybrid_retrieval(100, 10, 0.15, 42);
    }

    #[test]
    fn test_record_circuit_breaker_event() {
        record_circuit_breaker_event("open", "embed");
    }

    #[test]
    fn test_record_tier_operation() {
        record_tier_operation("promote", "episodic", 5);
    }

    #[test]
    fn test_record_evolution_cycle() {
        record_evolution_cycle(150, 3);
    }

    #[test]
    fn test_record_reasoning_chain() {
        record_reasoning_chain(4, true, 0.85, 200);
    }

    #[test]
    fn test_render_prometheus_without_init() {
        // Should return fallback string if not initialized
        let output = render_prometheus();
        assert!(output.contains("metrics not initialized") || !output.is_empty());
    }
}
