# agentic-memory v1.0.0

**Production-grade hierarchical memory layer for AI agents.**

A standalone, domain-agnostic memory server with SQLite persistence, vector search via sqlite-vec, knowledge graph traversal, chain-of-thought reasoning, memory consolidation, and self-evolution. Zero dependencies on any specific domain — works with any agent, chatbot, or AI system.

## Quick Start

```bash
cd Memory
cargo build --release
./target/release/agentic-memory
# Server starts on http://localhost:3111
```

### Custom configuration

```bash
MEMORY_DB_PATH=/tmp/memory.db \
MEMORY_ADDR=0.0.0.0:8080 \
OLLAMA_BASE_URL=http://localhost:11434 \
OLLAMA_MODEL=nomic-embed-text \
./target/release/agentic-memory
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| **4-Tier Memory** | ✅ | Working → Episodic → Semantic → Procedural with auto-promotion |
| **Vector Search** | ✅ | k-NN cosine via sqlite-vec, hybrid with BQ reranking, fallback |
| **Full-Text Search** | ✅ | FTS5 indexing across all record content |
| **Smart Search** | ✅ | Combined FTS + importance ranking |
| **Knowledge Graph** | ✅ | Directed edges, BFS/CTE traversal, community detection |
| **Chain-of-Thought** | ✅ | Reasoning chains with distillation to procedural memory |
| **Consolidation** | ✅ | Importance scoring, dedup, merge, conflict detection |
| **Self-Evolution** | ✅ | Tier tuning, stale pruning, procedural distillation |
| **REST API** | ✅ | 30+ endpoints via axum with CORS and auth |
| **Graceful Shutdown** | ✅ | SIGTERM/SIGINT handling with clean resource teardown |
| **Request Logging** | ✅ | Latency-tracked request/response logging |
| **Bitemporal Metadata** | ✅ | Valid time + transaction time on every record |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                HTTP API (port 3111)                        │
│            Graceful shutdown · Request logging             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌────────┐  ┌─────────┐  ┌───────────┐  │
│  │  Working  │  │Episodic│  │ Semantic │  │ Procedural │  │
│  │  (TTL)   │──▶│(events)│──▶│ (facts)  │──▶│ (rules)   │  │
│  └──────────┘  └────────┘  └─────────┘  └───────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │   SQLite Storage (single file)                      │ │
│  │   records · FTS5 · vec0 · graph · reasoning · tiers │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │   Engines: Consolidation · Evolution · Experts      │ │
│  │   (dedup, merge, tune tiers, prune, distill)        │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## API Reference

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health + recommendations |
| `GET` | `/stats` | Storage statistics with tier breakdown |
| `GET` | `/metrics` | Runtime metrics |
| `POST` | `/clear` | Clear all records |

### Records

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/records` | Insert a record |
| `GET` | `/records` | List records (`?type=&limit=&offset=`) |
| `GET` | `/records/:id` | Get record by ID |
| `DELETE` | `/records/:id` | Delete record by ID |

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/search?q=&tier=&limit=` | Full-text search |
| `POST` | `/search/semantic` | Semantic vector k-NN search |
| `GET` | `/search/smart?q=&limit=` | Smart search (FTS + importance) |

### Tier Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/tiers/:tier` | List records in a tier |
| `POST` | `/tiers/promote/:id/:tier` | Promote a record to a higher tier |
| `POST` | `/tiers/flush` | Flush working memory to episodic |
| `POST` | `/tiers/auto-promote/:tier` | Run auto-promotion |

### Knowledge Graph

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/graph/edges` | Add directed edge |
| `GET` | `/graph/edges/:id` | Get edges for a record |
| `POST` | `/graph/bfs` | BFS traversal |
| `GET` | `/graph/related/:id` | Get related records |
| `GET` | `/graph/hubs` | Most-connected records |

### Reasoning

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reason` | Run chain-of-thought reasoning |
| `GET` | `/reason/chains/:id` | Get reasoning chain |
| `GET` | `/reason/search?q=` | Search reasoning chains |
| `POST` | `/reason/distill/:id` | Distill to procedural memory |

### Consolidation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/consolidate` | Run consolidation cycle |
| `GET` | `/consolidate/analyze` | Analyze tier health |
| `POST` | `/consolidate/conflicts` | Detect conflicting records |

### Evolution

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/evolve` | Run evolution cycle |
| `GET` | `/evolution/events` | Get evolution events |

### Embeddings

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/embed` | Embed text via Ollama |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_DB_PATH` | `:memory:` | SQLite database path |
| `MEMORY_ADDR` | `0.0.0.0:3111` | Server bind address |
| `MEMORY_MAX_RECORDS` | `100000` | Maximum records |
| `VECTOR_DIMENSION` | `768` | Embedding dimension |
| `OLLAMA_BASE_URL` | (none) | Ollama URL for embeddings |
| `OLLAMA_MODEL` | (none) | Embedding model name |

## Examples

### Insert and search

```bash
# Insert a record
curl -s -X POST http://localhost:3111/records \
  -H 'Content-Type: application/json' \
  -d '{"id":"doc-1","content":"Bitcoin reached $100k","content_type":"news","tier":"episodic","importance":0.8}'

# Search
curl -s 'http://localhost:3111/search?q=bitcoin&limit=5'

# Smart search
curl -s 'http://localhost:3111/search/smart?q=cryptocurrency'

# Stats
curl -s http://localhost:3111/stats | python3 -m json.tool
```

### Python client

```python
import requests
BASE = "http://localhost:3111"

# Insert
requests.post(f"{BASE}/records", json={
    "id": "py-1", "content": "Test record", "content_type": "test", "tier": "episodic"
})

# Search
resp = requests.get(f"{BASE}/search", params={"q": "test"})
for r in resp.json():
    print(f"  [{r['score']:.2f}] {r['record']['content'][:80]}")
```

## Testing

```bash
# Run all 72 unit tests
cargo test --lib -- --test-threads=1

# Run specific module tests
cargo test --lib store::tests
cargo test --lib api::tests

# Clippy (0 warnings)
cargo clippy --all-targets
```

## Project Structure

```
Memory/src/
├── main.rs           # Entry point with graceful shutdown
├── lib.rs            # Public API, re-exports
├── api.rs            # HTTP REST handlers + tests (30+ endpoints)
├── store.rs          # SQLite storage (records, FTS5, graph, vec0)
├── types.rs          # All type definitions
├── tiers.rs          # WorkingMemory buffer, TieredMemory
├── vector.rs         # Cosine similarity, binary quantization
├── graph.rs          # KnowledgeGraph with BFS, hubs
├── rag.rs            # Embedder trait, OllamaEmbedder, chunking
├── reasoning.rs      # Chain-of-thought storage + distillation
├── consolidation.rs  # Importance scoring, dedup, merge
├── evolution.rs      # Tier tuning, stale pruning
├── experts.rs        # Retrieval/Reasoning/Consolidation/Evolution experts
├── metrics.rs        # Runtime metrics
├── resilience.rs     # Retry policies, timeouts
├── cache.rs          # In-memory policy cache with TTL
└── errors.rs         # Unified error types
```

## License

MIT
