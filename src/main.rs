use std::sync::Arc;
use std::time::Duration;

use agentic_memory::api::MemoryApi;
use agentic_memory::evolution::{EvolutionConfig, EvolutionEngine};
use agentic_memory::rag::{Embedder, OllamaEmbedder};
use agentic_memory::store::MemoryStore;
use agentic_memory::types::StorageConfig;
use tracing::{info, warn};

#[tokio::main]
async fn main() {
    // Initialize structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,agentic_memory=debug".to_string()),
        )
        .init();

    // Initialize Prometheus metrics recorder
    agentic_memory::metrics::init_prometheus();
    tracing::info!("Prometheus metrics recorder initialized (served at /metrics)");

    let db_path = std::env::var("MEMORY_DB_PATH").unwrap_or_else(|_| ":memory:".to_string());
    let addr = std::env::var("MEMORY_ADDR").unwrap_or_else(|_| "0.0.0.0:3111".to_string());

    info!("Starting agentic-memory server...");

    // Optional Ollama embedder
    let embedder = match (
        std::env::var("OLLAMA_BASE_URL").ok(),
        std::env::var("OLLAMA_MODEL").ok(),
    ) {
        (Some(url), Some(model)) if !url.is_empty() && !model.is_empty() => {
            match OllamaEmbedder::new(&url, &model) {
                Ok(e) => {
                    info!("Ollama embedder configured: {} @ {}", model, url);
                    Some(Arc::new(e) as Arc<dyn Embedder>)
                }
                Err(err) => {
                    warn!("Failed to create OllamaEmbedder: {}", err);
                    None
                }
            }
        }
        _ => {
            info!("No Ollama embedder configured (set OLLAMA_BASE_URL and OLLAMA_MODEL to enable)");
            None
        }
    };

    let api = MemoryApi::with_embedder(&db_path, &addr, embedder.clone())
        .expect("Failed to create MemoryApi");

    // ── Background Evolution Loop ─────────────────────────────────────────
    // Start self-evolution in the background (tier tuning, pruning, distillation)
    let evolution_store = MemoryStore::open(&StorageConfig {
        db_path: db_path.clone(),
        max_ram_entries: 100,
        auto_embed: false,
        vector_dimension: embedder.as_ref().map(|e| e.dimension()).unwrap_or(768),
    })
    .expect("Failed to open store for evolution");

    let evolution_config = EvolutionConfig::default();
    let mut evolution_engine = EvolutionEngine::new(evolution_store, evolution_config);

    let evolution_handle = tokio::spawn(async move {
        let interval = Duration::from_secs(3600); // Run every hour
        info!("Background evolution started (interval: 1 hour)");

        loop {
            tokio::time::sleep(interval).await;

            info!("Running scheduled evolution cycle...");
            let report = evolution_engine.run_sleep_cycle();

            if let Some(consolidation) = &report.consolidation_report {
                info!(
                    "Evolution cycle completed | records_processed: {} | promoted: {} | evicted: {}",
                    consolidation.records_processed,
                    consolidation.records_promoted,
                    consolidation.records_evicted
                );
            } else {
                info!("Evolution cycle completed (no consolidation report)");
            }
        }
    });

    info!("Server listening on http://{}", addr);

    // ── Graceful Shutdown ─────────────────────────────────────────────────
    // Listen for SIGTERM/SIGINT and shut down cleanly
    let server_handle = tokio::spawn(async move {
        api.serve().await.expect("Failed to serve");
    });

    let shutdown_signal = async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("Failed to install SIGTERM handler");
            tokio::select! {
                _ = ctrl_c => info!("Received SIGINT, shutting down..."),
                _ = sigterm.recv() => info!("Received SIGTERM, shutting down..."),
            }
        }
        #[cfg(not(unix))]
        {
            ctrl_c.await.ok();
            info!("Received SIGINT, shutting down...");
        }
    };

    tokio::select! {
        _ = server_handle => {},
        _ = shutdown_signal => {
            info!("Shutting down evolution background task...");
            evolution_handle.abort();
            info!("Server shut down gracefully.");
        }
    }
}
// ci-trigger
