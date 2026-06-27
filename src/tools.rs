// src/tools.rs
// Web Search Tool with Tavily + Fallback support

use crate::resilience::ResilientClient;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ExternalSearchResult {
    pub title: String,
    pub url: String,
    pub content: String,
    pub score: f64,
}

pub struct WebSearchTool {
    client: ResilientClient,
    api_key: Option<String>,
}

impl WebSearchTool {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            client: ResilientClient::new(3, 5, 45),
            api_key,
        }
    }

    pub async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<ExternalSearchResult>, String> {
        if self.api_key.is_some() {
            let payload = serde_json::json!({
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced"
            });

            let url = "https://api.tavily.com/search";
            let results: serde_json::Value = self.client.post_json(url, &payload).await?;

            let mut parsed = vec![];
            if let Some(arr) = results["results"].as_array() {
                for item in arr {
                    parsed.push(ExternalSearchResult {
                        title: item["title"].as_str().unwrap_or("").to_string(),
                        url: item["url"].as_str().unwrap_or("").to_string(),
                        content: item["content"].as_str().unwrap_or("").to_string(),
                        score: item["score"].as_f64().unwrap_or(0.0),
                    });
                }
            }
            Ok(parsed)
        } else {
            // Fallback when no API key
            Ok(vec![ExternalSearchResult {
                title: format!("Search result for: {}", query),
                url: "https://example.com".to_string(),
                content: "Configure TAVILY_API_KEY to enable real web search.".to_string(),
                score: 0.5,
            }])
        }
    }
}