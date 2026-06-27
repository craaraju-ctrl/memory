# Changelog

## [0.2.0] - 2026-06-25

### Added
- Full Staleness Management System (temporal decay + validation)
- Deep Reasoning + Web Search with tool-calling loop (Qwen3 / Nemotron ready)
- Hybrid Retrieval with Graph Boosting + Expansion
- Tags Support (full CRUD + filtering)
- Request body size limits and improved security
- Prometheus metrics + OpenTelemetry tracing
- Centralized UUIDv7 ID generation

### Improved
- Transaction safety and vector embedding consistency
- Mutex poison recovery in all hot paths
- Error handling unification (AppError everywhere)
- Code health and maintainability (reduced complexity)

### Fixed
- Vector write after transaction commit
- Inconsistent WorkingMemory side effects
- Binary quantization panic risk
- Remaining `.unwrap()` in production paths

## [0.1.0] - Initial Release
- Core hierarchical memory, Hybrid Retrieval, Graph, API