//! # Memory Client SDK — Universal Agent Adapter
//!
//! A simple Rust client that wraps the Memory HTTP API.
//! Any agent can use this to store, search, and manage memories.
//!
//! ## Quick Start
//!
//! ```no_run
//! # async fn doc() {
//! use agentic_memory::client::MemoryClient;
//!
//! let client = MemoryClient::new("http://localhost:3111");
//!
//! // Insert a memory
//! let id = client.insert("I learned that Rust uses ownership", "fact", "semantic", 0.8).await.unwrap();
//!
//! // Search memories
//! let results = client.search("Rust ownership", 5).await.unwrap();
//!
//! // Get a specific memory
//! let record = client.get(&id).await.unwrap();
//! # }
//! ```

use serde::{Deserialize, Serialize};

/// A simple HTTP client for the Memory API.
pub struct MemoryClient {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

/// A search result returned by the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientSearchResult {
    pub id: String,
    pub content: String,
    pub content_type: String,
    pub tier: String,
    pub score: f64,
    pub method: String,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Record details returned by the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientRecord {
    pub id: String,
    pub content: String,
    pub content_type: String,
    pub tier: String,
    pub importance: f64,
    pub access_count: u64,
    pub metadata: std::collections::HashMap<String, String>,
    pub timestamp: String,
}

/// Health status returned by the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub total_records: u64,
    pub graph_edges: u64,
}

/// Stats returned by the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_records: u64,
    pub total_with_embeddings: u64,
    pub storage_bytes: u64,
}

impl MemoryClient {
    /// Create a new client pointing to the Memory API server.
    pub fn new(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: None,
        }
    }

    /// Set an API key for authenticated requests.
    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    /// Create from environment variables (MEMORY_API_URL, MEMORY_API_KEY).
    pub fn from_env() -> Self {
        let base_url =
            std::env::var("MEMORY_API_URL").unwrap_or_else(|_| "http://localhost:3111".to_string());
        let api_key = std::env::var("MEMORY_API_KEY").ok();
        let mut client = Self::new(&base_url);
        if let Some(key) = api_key {
            client = client.with_api_key(&key);
        }
        client
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    fn build_request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let mut req = self.client.request(method, self.url(path));
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        req
    }

    // ── Records CRUD ─────────────────────────────────────────────────────

    /// Insert a record into memory.
    pub async fn insert(
        &self,
        content: &str,
        content_type: &str,
        tier: &str,
        importance: f64,
    ) -> Result<String, String> {
        let body = serde_json::json!({
            "id": uuid::Uuid::now_v7().to_string(),
            "content": content,
            "content_type": content_type,
            "tier": tier,
            "importance": importance,
        });

        let resp: serde_json::Value = self
            .build_request(reqwest::Method::POST, "/records")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        resp["id"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| format!("No id in response: {}", resp))
    }

    /// Insert a record with a specific ID.
    pub async fn insert_with_id(
        &self,
        id: &str,
        content: &str,
        content_type: &str,
        tier: &str,
        importance: f64,
    ) -> Result<String, String> {
        let body = serde_json::json!({
            "id": id,
            "content": content,
            "content_type": content_type,
            "tier": tier,
            "importance": importance,
        });

        self.build_request(reqwest::Method::POST, "/records")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        Ok(id.to_string())
    }

    /// Get a record by ID.
    pub async fn get(&self, id: &str) -> Result<ClientRecord, String> {
        let resp: serde_json::Value = self
            .build_request(reqwest::Method::GET, &format!("/records/{}", id))
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        Ok(ClientRecord {
            id: resp["record"]["id"]
                .as_str()
                .or_else(|| resp["id"].as_str())
                .unwrap_or("")
                .to_string(),
            content: resp["record"]["content"]
                .as_str()
                .or_else(|| resp["content"].as_str())
                .unwrap_or("")
                .to_string(),
            content_type: resp["record"]["content_type"]
                .as_str()
                .or_else(|| resp["content_type"].as_str())
                .unwrap_or("")
                .to_string(),
            tier: resp["tier"].as_str().unwrap_or("episodic").to_string(),
            importance: resp["importance"].as_f64().unwrap_or(0.5),
            access_count: resp["access_count"].as_u64().unwrap_or(0),
            metadata: resp
                .get("record")
                .and_then(|r| r.get("metadata"))
                .or_else(|| resp.get("metadata"))
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default(),
            timestamp: resp["record"]["timestamp"]
                .as_str()
                .or_else(|| resp["timestamp"].as_str())
                .unwrap_or("")
                .to_string(),
        })
    }

    /// Delete a record by ID.
    pub async fn delete(&self, id: &str) -> Result<bool, String> {
        let resp = self
            .build_request(reqwest::Method::DELETE, &format!("/records/{}", id))
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        Ok(resp.status().is_success())
    }

    // ── Search ───────────────────────────────────────────────────────────

    /// Search records by full-text query.
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ClientSearchResult>, String> {
        let url = format!("/search?q={}&limit={}", urlencoding::encode(query), limit);
        let resp: Vec<serde_json::Value> = self
            .build_request(reqwest::Method::GET, &url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        Ok(resp
            .into_iter()
            .map(|r| ClientSearchResult {
                id: r["record"]["id"]
                    .as_str()
                    .or_else(|| r["id"].as_str())
                    .unwrap_or("")
                    .to_string(),
                content: r["record"]["content"]
                    .as_str()
                    .or_else(|| r["content"].as_str())
                    .unwrap_or("")
                    .to_string(),
                content_type: r["record"]["content_type"]
                    .as_str()
                    .or_else(|| r["content_type"].as_str())
                    .unwrap_or("")
                    .to_string(),
                tier: "episodic".to_string(),
                score: r["score"].as_f64().unwrap_or(0.0),
                method: r["method"].as_str().unwrap_or("fts").to_string(),
                metadata: r
                    .get("record")
                    .and_then(|rec| rec.get("metadata"))
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok())
                    .unwrap_or_default(),
            })
            .collect())
    }

    /// Search within a specific tier.
    pub async fn search_tier(
        &self,
        query: &str,
        tier: &str,
        limit: usize,
    ) -> Result<Vec<ClientSearchResult>, String> {
        let url = format!(
            "/search?q={}&tier={}&limit={}",
            urlencoding::encode(query),
            tier,
            limit
        );
        let resp: Vec<serde_json::Value> = self
            .build_request(reqwest::Method::GET, &url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        Ok(resp
            .into_iter()
            .map(|r| ClientSearchResult {
                id: r["record"]["id"]
                    .as_str()
                    .or_else(|| r["id"].as_str())
                    .unwrap_or("")
                    .to_string(),
                content: r["record"]["content"]
                    .as_str()
                    .or_else(|| r["content"].as_str())
                    .unwrap_or("")
                    .to_string(),
                content_type: r["record"]["content_type"]
                    .as_str()
                    .or_else(|| r["content_type"].as_str())
                    .unwrap_or("")
                    .to_string(),
                tier: tier.to_string(),
                score: r["score"].as_f64().unwrap_or(0.0),
                method: r["method"].as_str().unwrap_or("fts").to_string(),
                metadata: r
                    .get("record")
                    .and_then(|rec| rec.get("metadata"))
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok())
                    .unwrap_or_default(),
            })
            .collect())
    }

    // ── Tier Operations ──────────────────────────────────────────────────

    /// Promote a record to a different tier.
    pub async fn promote(&self, id: &str, tier: &str) -> Result<(), String> {
        self.build_request(
            reqwest::Method::POST,
            &format!("/tiers/promote/{}/{}", id, tier),
        )
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;
        Ok(())
    }

    /// Flush working memory to episodic tier.
    pub async fn flush_working(&self) -> Result<u64, String> {
        let resp: serde_json::Value = self
            .build_request(reqwest::Method::POST, "/tiers/flush")
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        Ok(resp["flushed"].as_u64().unwrap_or(0))
    }

    // ── Graph ────────────────────────────────────────────────────────────

    /// Add an edge between two records.
    pub async fn add_edge(
        &self,
        source_id: &str,
        target_id: &str,
        relation_type: &str,
        weight: f64,
    ) -> Result<String, String> {
        let body = serde_json::json!({
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "weight": weight,
        });

        let resp: serde_json::Value = self
            .build_request(reqwest::Method::POST, "/graph/edges")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        Ok(resp["edge_id"].as_str().unwrap_or("").to_string())
    }

    // ── System ───────────────────────────────────────────────────────────

    /// Check system health.
    pub async fn health(&self) -> Result<HealthStatus, String> {
        let resp: serde_json::Value = self
            .build_request(reqwest::Method::GET, "/health")
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        Ok(HealthStatus {
            status: resp["status"].as_str().unwrap_or("unknown").to_string(),
            total_records: resp["total_records"].as_u64().unwrap_or(0),
            graph_edges: resp["graph_edges"].as_u64().unwrap_or(0),
        })
    }

    /// Get storage statistics.
    pub async fn stats(&self) -> Result<MemoryStats, String> {
        let resp: serde_json::Value = self
            .build_request(reqwest::Method::GET, "/stats")
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        Ok(MemoryStats {
            total_records: resp["total_records"].as_u64().unwrap_or(0),
            total_with_embeddings: resp["total_with_embeddings"].as_u64().unwrap_or(0),
            storage_bytes: resp["storage_bytes"].as_u64().unwrap_or(0),
        })
    }

    /// Clear all records.
    pub async fn clear(&self) -> Result<(), String> {
        self.build_request(reqwest::Method::POST, "/clear")
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = MemoryClient::new("http://localhost:3111");
        assert_eq!(client.base_url, "http://localhost:3111");
    }

    #[test]
    fn test_client_with_api_key() {
        let client = MemoryClient::new("http://localhost:3111").with_api_key("test-key");
        assert_eq!(client.api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_url_construction() {
        let client = MemoryClient::new("http://localhost:3111");
        assert_eq!(client.url("/records"), "http://localhost:3111/records");
        assert_eq!(client.url("/health"), "http://localhost:3111/health");
    }
}
