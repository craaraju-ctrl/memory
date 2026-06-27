# =============================================================================
#  agentic-memory — Production Dockerfile
# =============================================================================
# Multi-stage build for minimal image size and security.
#
# Build:
#   docker build -t agentic-memory:0.2.0 .
#
# Run standalone:
#   docker run -p 3111:3111 -v memory_data:/data agentic-memory:0.2.0
#
# Recommended: Use docker-compose.yml for production (includes Ollama).
# =============================================================================

# ── Stage 1: Build ──────────────────────────────────────────────────────────
# Pin to a specific Rust version for reproducible builds.
# Update this periodically as the project tracks Rust stable.
FROM rust:1.81-slim-bookworm AS builder

# Build dependencies: C compiler needed by rusqlite (bundled) + sqlite-vec
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests first to cache dependency compilation
COPY Cargo.toml Cargo.lock ./

# Create dummy sources to pre-build dependencies
RUN mkdir src && \
    echo "fn main() { println!(\"dummy\"); }" > src/main.rs && \
    echo "" > src/lib.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Copy the real source code and rebuild (only project code, deps are cached)
COPY src/ src/
RUN touch src/main.rs && \
    cargo build --release

# ── Stage 2: Runtime ───────────────────────────────────────────────────────
FROM debian:bookworm-slim

# Add metadata labels
LABEL org.opencontainers.image.title="agentic-memory"
LABEL org.opencontainers.image.version="0.2.0"
LABEL org.opencontainers.image.description="Production-grade hierarchical agent memory system"
LABEL org.opencontainers.image.source="https://github.com/craaraju-ctrl/agentic-memory"

# Runtime dependencies: ca-certificates for reqwest HTTPS, wget for healthcheck
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the release binary
COPY --from=builder /app/target/release/agentic-memory /usr/local/bin/agentic-memory

# Persistent SQLite data directory
RUN mkdir -p /data
VOLUME ["/data"]

# Port for the HTTP API
EXPOSE 3111

# Health check (requires wget, installed above)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3111/health || exit 1

# Default environment variables
ENV MEMORY_DB_PATH=/data/memory.db
ENV MEMORY_ADDR=0.0.0.0:3111
ENV VECTOR_DIMENSION=768

# Run as non-root user for security
RUN useradd -m -u 1001 memory && \
    chown -R memory:memory /data
USER memory

ENTRYPOINT ["/usr/local/bin/agentic-memory"]
