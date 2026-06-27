//! # MCP Server — Model Context Protocol for Universal Agent Access
//!
//! Exposes the Memory module as an MCP server so any MCP-compatible agent
//! (Claude Desktop, Cursor, Zed, etc.) can use it as a memory tool.
//!
//! ## Available Tools
//!
//! | Tool | Description |
//! |------|-------------|
//! | `memory_insert` | Store a new memory with content, type, and tier |
//! | `memory_search` | Search memories by text query |
//! | `memory_get` | Retrieve a specific memory by ID |
//! | `memory_delete` | Remove a memory by ID |
//! | `memory_stats` | Get storage statistics |
//! | `memory_health` | Check system health |
//! | `memory_promote` | Promote a record to a different tier |
//! | `memory_add_edge` | Create a relationship between two memories |
//!
//! ## Running as MCP Server
//!
//! ```bash
//! # stdio mode (for Claude Desktop integration)
//! cargo run --bin agentic-memory -- mcp
//!
//! # HTTP mode (for remote access)
//! MEMORY_MCP_PORT=3112 cargo run --bin agentic-memory -- mcp --http
//! ```

use serde::{Deserialize, Serialize};

/// MCP tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// MCP tool call request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// MCP tool call response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResult {
    pub content: Vec<McpContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// MCP content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
}

/// Get all available MCP tools for the Memory module.
pub fn get_tools() -> Vec<McpTool> {
    vec![
        McpTool {
            name: "memory_insert".to_string(),
            description: "Store a new memory. Returns the record ID.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember"
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Type of content (e.g., fact, event, note, procedure)",
                        "default": "note"
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["working", "episodic", "semantic", "procedural"],
                        "description": "Memory tier to store in",
                        "default": "episodic"
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score 0.0-1.0",
                        "default": 0.5
                    }
                },
                "required": ["content"]
            }),
        },
        McpTool {
            name: "memory_search".to_string(),
            description: "Search memories by text query. Returns matching records with relevance scores.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 10
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["working", "episodic", "semantic", "procedural"],
                        "description": "Search within a specific tier (optional)"
                    }
                },
                "required": ["query"]
            }),
        },
        McpTool {
            name: "memory_get".to_string(),
            description: "Retrieve a specific memory by its ID.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The memory record ID"
                    }
                },
                "required": ["id"]
            }),
        },
        McpTool {
            name: "memory_delete".to_string(),
            description: "Remove a memory by its ID.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The memory record ID to delete"
                    }
                },
                "required": ["id"]
            }),
        },
        McpTool {
            name: "memory_stats".to_string(),
            description: "Get storage statistics (total records, embeddings, etc.).".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        McpTool {
            name: "memory_health".to_string(),
            description: "Check system health status.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        McpTool {
            name: "memory_promote".to_string(),
            description: "Promote a memory to a different tier (e.g., episodic → semantic).".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The memory record ID"
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["working", "episodic", "semantic", "procedural"],
                        "description": "Target tier"
                    }
                },
                "required": ["id", "tier"]
            }),
        },
        McpTool {
            name: "memory_add_edge".to_string(),
            description: "Create a relationship edge between two memories.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Source memory ID"
                    },
                    "target_id": {
                        "type": "string",
                        "description": "Target memory ID"
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Relationship type (e.g., related_to, causes, depends_on)"
                    },
                    "weight": {
                        "type": "number",
                        "description": "Relationship strength 0.0-1.0",
                        "default": 1.0
                    }
                },
                "required": ["source_id", "target_id", "relation_type"]
            }),
        },
    ]
}

/// Execute an MCP tool call against the Memory API.
pub async fn execute_tool(
    call: &McpToolCall,
    api_url: &str,
) -> McpToolResult {
    let client = reqwest::Client::new();
    let base = api_url.trim_end_matches('/');

    let result = match call.name.as_str() {
        "memory_insert" => {
            let content = call.arguments["content"].as_str().unwrap_or("");
            let content_type = call.arguments["content_type"].as_str().unwrap_or("note");
            let tier = call.arguments["tier"].as_str().unwrap_or("episodic");
            let importance = call.arguments["importance"].as_f64().unwrap_or(0.5);

            let body = serde_json::json!({
                "id": uuid::Uuid::now_v7().to_string(),
                "content": content,
                "content_type": content_type,
                "tier": tier,
                "importance": importance,
            });

            match client
                .post(format!("{}/records", base))
                .json(&body)
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<serde_json::Value>().await {
                            Ok(json) => Ok(format!("Memory stored with ID: {}", json["id"])),
                            Err(e) => Err(format!("Parse error: {}", e)),
                        }
                    } else {
                        Err(format!("HTTP {}: store failed", status))
                    }
                }
                Err(e) => Err(format!("Request failed: {}", e)),
            }
        }

        "memory_search" => {
            let query = call.arguments["query"].as_str().unwrap_or("");
            let limit = call.arguments["limit"].as_u64().unwrap_or(10);
            let tier = call.arguments["tier"].as_str();

            let url = if let Some(t) = tier {
                format!("{}/search?q={}&tier={}&limit={}", base, urlencoding::encode(query), t, limit)
            } else {
                format!("{}/search?q={}&limit={}", base, urlencoding::encode(query), limit)
            };

            match client.get(&url).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<Vec<serde_json::Value>>().await {
                            Ok(results) => {
                                if results.is_empty() {
                                    Ok("No matching memories found.".to_string())
                                } else {
                                    let formatted: Vec<String> = results
                                        .iter()
                                        .enumerate()
                                        .map(|(i, r)| {
                                            let id = r["record"]["id"]
                                                .as_str()
                                                .or_else(|| r["id"].as_str())
                                                .unwrap_or("?");
                                            let content = r["record"]["content"]
                                                .as_str()
                                                .or_else(|| r["content"].as_str())
                                                .unwrap_or("");
                                            let score = r["score"].as_f64().unwrap_or(0.0);
                                            format!("{}. [{}] (score: {:.2}) {}", i + 1, id, score, content)
                                        })
                                        .collect();
                                    Ok(formatted.join("\n"))
                                }
                            }
                            Err(e) => Err(format!("Parse error: {}", e)),
                        }
                    } else {
                        Err(format!("HTTP {}: search failed", status))
                    }
                }
                Err(e) => Err(format!("Request failed: {}", e)),
            }
        }

        "memory_get" => {
            let id = call.arguments["id"].as_str().unwrap_or("");
            match client.get(format!("{}/records/{}", base, id)).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<serde_json::Value>().await {
                            Ok(json) => Ok(serde_json::to_string_pretty(&json).unwrap_or_default()),
                            Err(e) => Err(format!("Parse error: {}", e)),
                        }
                    } else {
                        Err(format!("HTTP {}: record not found", status))
                    }
                }
                Err(e) => Err(format!("Request failed: {}", e)),
            }
        }

        "memory_delete" => {
            let id = call.arguments["id"].as_str().unwrap_or("");
            match client
                .delete(format!("{}/records/{}", base, id))
                .send()
                .await
            {
                Ok(resp) => {
                    if resp.status().is_success() {
                        Ok(format!("Memory '{}' deleted.", id))
                    } else {
                        Err(format!("Delete failed (status {})", resp.status()))
                    }
                }
                Err(e) => Err(format!("Request failed: {}", e)),
            }
        }

        "memory_stats" => {
            match client.get(format!("{}/stats", base)).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<serde_json::Value>().await {
                            Ok(json) => Ok(serde_json::to_string_pretty(&json).unwrap_or_default()),
                            Err(e) => Err(format!("Parse error: {}", e)),
                        }
                    } else {
                        Err(format!("HTTP {}: stats failed", status))
                    }
                }
                Err(e) => Err(format!("Request failed: {}", e)),
            }
        }

        "memory_health" => {
            match client.get(format!("{}/health", base)).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<serde_json::Value>().await {
                            Ok(json) => Ok(serde_json::to_string_pretty(&json).unwrap_or_default()),
                            Err(e) => Err(format!("Parse error: {}", e)),
                        }
                    } else {
                        Err(format!("HTTP {}: health check failed", status))
                    }
                }
                Err(e) => Err(format!("Request failed: {}", e)),
            }
        }

        "memory_promote" => {
            let id = call.arguments["id"].as_str().unwrap_or("");
            let tier = call.arguments["tier"].as_str().unwrap_or("semantic");
            match client
                .post(format!("{}/tiers/promote/{}/{}", base, id, tier))
                .send()
                .await
            {
                Ok(resp) => {
                    if resp.status().is_success() {
                        Ok(format!("Memory '{}' promoted to {}.", id, tier))
                    } else {
                        Err(format!("Promote failed (status {})", resp.status()))
                    }
                }
                Err(e) => Err(format!("Request failed: {}", e)),
            }
        }

        "memory_add_edge" => {
            let source = call.arguments["source_id"].as_str().unwrap_or("");
            let target = call.arguments["target_id"].as_str().unwrap_or("");
            let relation = call.arguments["relation_type"].as_str().unwrap_or("related_to");
            let weight = call.arguments["weight"].as_f64().unwrap_or(1.0);

            let body = serde_json::json!({
                "source_id": source,
                "target_id": target,
                "relation_type": relation,
                "weight": weight,
            });

            match client
                .post(format!("{}/graph/edges", base))
                .json(&body)
                .send()
                .await
            {
                Ok(resp) => {
                    if resp.status().is_success() {
                        Ok(format!("Edge created: {} --[{}]--> {}", source, relation, target))
                    } else {
                        Err(format!("Edge creation failed (status {})", resp.status()))
                    }
                }
                Err(e) => Err(format!("Request failed: {}", e)),
            }
        }

        _ => Err(format!("Unknown tool: {}", call.name)),
    };

    match result {
        Ok(text) => McpToolResult {
            content: vec![McpContent::Text { text }],
            is_error: None,
        },
        Err(error) => McpToolResult {
            content: vec![McpContent::Text { text: error }],
            is_error: Some(true),
        },
    }
}

/// Run the MCP server in stdio mode (for Claude Desktop integration).
pub async fn run_stdio(api_url: &str) {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    eprintln!("Memory MCP server running (stdio mode)...");
    eprintln!("API URL: {}", api_url);

    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(msg) => {
                let method = msg["method"].as_str().unwrap_or("");

                match method {
                    "initialize" => serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {}
                            },
                            "serverInfo": {
                                "name": "agentic-memory",
                                "version": "1.0.0"
                            }
                        }
                    }),

                    "tools/list" => {
                        let tools = get_tools();
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "result": {
                                "tools": tools
                            }
                        })
                    }

                    "tools/call" => {
                        let params = &msg["params"];
                        let tool_name = params["name"].as_str().unwrap_or("");
                        let arguments = params["arguments"].clone();

                        let call = McpToolCall {
                            name: tool_name.to_string(),
                            arguments,
                        };

                        let result = execute_tool(&call, api_url).await;

                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "result": result
                        })
                    }

                    _ => serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "error": {
                            "code": -32601,
                            "message": format!("Method not found: {}", method)
                        }
                    }),
                }
            }
            Err(e) => serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {
                    "code": -32700,
                    "message": format!("Parse error: {}", e)
                }
            }),
        };

        println!("{}", serde_json::to_string(&response).unwrap());
    }
}
