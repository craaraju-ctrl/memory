// src/llm.rs
// LLM Client with Tool Calling Support (OpenAI-compatible for SGLang / vLLM / Ollama)

use crate::resilience::ResilientClient;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: Option<String>,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug)]
pub struct LLMResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

pub struct LLMClient {
    client: ResilientClient,
    base_url: String,
    model: String,
}

impl LLMClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        let client = ResilientClient::new(4, 6, 90);
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    pub fn new_from_env() -> Self {
        let base_url =
            std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());
        let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "qwen3-8b".to_string());
        Self::new(&base_url, &model)
    }

    pub async fn chat_completion(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<LLMResponse, String> {
        let payload = if let Some(tools) = tools {
            serde_json::json!({
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.6,
                "max_tokens": 2048,
            })
        } else {
            serde_json::json!({
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
            })
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let response: serde_json::Value = self.client.post_json(&url, &payload).await?;

        let choice = &response["choices"][0]["message"];
        let content = choice["content"].as_str().unwrap_or("").to_string();

        let tool_calls = if let Some(calls) = choice["tool_calls"].as_array() {
            calls
                .iter()
                .filter_map(|c| serde_json::from_value::<ToolCall>(c.clone()).ok())
                .collect()
        } else {
            vec![]
        };

        Ok(LLMResponse {
            content,
            tool_calls,
        })
    }
}
