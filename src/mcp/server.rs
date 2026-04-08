//! MCP Server implementation
//!
//! This module provides the main server that handles JSON-RPC communication.

use crate::mcp::tools::{register_algorithm_tools, register_graph_tools, ToolRegistry};
use crate::mcp::types::{
    Capabilities, JsonRpcError, JsonRpcRequest, JsonRpcResponse, ServerInfo, ToolsCapability,
};
use serde_json::Value;
use std::io::{self, BufRead, Write};

/// MCP Server for God-Graph
pub struct McpServer {
    /// Server information
    info: ServerInfo,
    /// Tool registry
    registry: ToolRegistry,
    /// Capabilities
    capabilities: Capabilities,
    /// P2 OPTIMIZATION: Cached JSON-RPC version string to avoid repeated allocations
    jsonrpc_version: &'static str,
}

impl McpServer {
    /// Create a new MCP server
    /// P2 OPTIMIZATION: Use &'static str for JSON-RPC version
    pub fn new() -> Self {
        let mut registry = ToolRegistry::new();
        register_graph_tools(&mut registry);
        register_algorithm_tools(&mut registry);

        Self {
            info: ServerInfo::default(),
            registry,
            capabilities: Capabilities {
                tools: Some(ToolsCapability {
                    list_changed: Some(true),
                }),
                resources: None,
                prompts: None,
            },
            jsonrpc_version: crate::mcp::types::JSON_RPC_VERSION,
        }
    }

    /// Get server name
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Get server version
    pub fn version(&self) -> &str {
        &self.info.version
    }

    /// Register additional tools
    pub fn register_tool(&mut self, name: String, description: String, schema: crate::mcp::types::JsonSchema, handler: crate::mcp::tools::ToolFn) {
        let tool = crate::mcp::types::Tool::new(name, description, schema);
        self.registry.register_tool(tool, handler);
    }

    /// Handle a JSON-RPC request
    pub fn handle_request(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            // MCP protocol methods
            "initialize" => self.handle_initialize(request),
            "initialized" => self.handle_initialized(request),
            "ping" => self.handle_ping(request),

            // Tools methods
            "tools/list" => self.handle_tools_list(request),
            "tools/call" => self.handle_tools_call(request),

            // Unknown method
            _ => self.error_response(
                request.id.clone(),
                -32601,
                format!("Method not found: {}", request.method),
            ),
        }
    }

    /// Handle initialize request
    fn handle_initialize(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let result = serde_json::json!({
            "protocolVersion": self.info.protocol_version,
            "capabilities": self.capabilities,
            "serverInfo": self.info,
        });

        JsonRpcResponse {
            jsonrpc: self.jsonrpc_version.to_string(),
            id: request.id.clone(),
            result: Some(result),
            error: None,
        }
    }

    /// Handle initialized notification (no response needed)
    fn handle_initialized(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        // This is a notification, no response required
        // Return empty response to acknowledge
        JsonRpcResponse {
            jsonrpc: self.jsonrpc_version.to_string(),
            id: request.id.clone(),
            result: Some(Value::Null),
            error: None,
        }
    }

    /// Handle ping request
    fn handle_ping(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: self.jsonrpc_version.to_string(),
            id: request.id.clone(),
            result: Some(Value::Null),
            error: None,
        }
    }

    /// Handle tools/list request
    fn handle_tools_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let tools: Vec<Value> = self
            .registry
            .list_tools()
            .into_iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                })
            })
            .collect();

        let result = serde_json::json!({
            "tools": tools,
        });

        JsonRpcResponse {
            jsonrpc: self.jsonrpc_version.to_string(),
            id: request.id.clone(),
            result: Some(result),
            error: None,
        }
    }

    /// Handle tools/call request
    fn handle_tools_call(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params = match &request.params {
            Some(p) => p,
            None => {
                return self.error_response(request.id.clone(), -32602, "Missing params");
            }
        };

        let name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => {
                return self.error_response(request.id.clone(), -32602, "Missing tool name");
            }
        };

        let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);

        match self.registry.execute_tool(name, arguments) {
            Ok(result) => {
                let result_value = serde_json::to_value(&result).unwrap_or(Value::Null);
                JsonRpcResponse {
                    jsonrpc: self.jsonrpc_version.to_string(),
                    id: request.id.clone(),
                    result: Some(result_value),
                    error: None,
                }
            }
            Err(e) => self.error_response(request.id.clone(), -32000, e),
        }
    }

    /// Create an error response
    fn error_response(
        &self,
        id: Option<Value>,
        code: i32,
        message: impl Into<String>,
    ) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: self.jsonrpc_version.to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    /// Run the server using stdio transport
    pub fn run_stdio(&self) -> io::Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        for line in stdin.lock().lines() {
            let line = line?;

            // Parse JSON-RPC request
            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    let error_response = JsonRpcResponse {
                        jsonrpc: self.jsonrpc_version.to_string(),
                        id: None,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32700,
                            message: format!("Parse error: {}", e),
                            data: None,
                        }),
                    };
                    let response_str = serde_json::to_string(&error_response)?;
                    writeln!(stdout, "{}", response_str)?;
                    continue;
                }
            };

            // Handle request
            let response = self.handle_request(&request);

            // Send response (only if there's an id)
            if response.id.is_some() {
                let response_str = serde_json::to_string(&response)?;
                writeln!(stdout, "{}", response_str)?;
                stdout.flush()?;
            }
        }

        Ok(())
    }

    /// Process a single request (useful for testing)
    pub fn process_request(&self, request_str: &str) -> Result<String, serde_json::Error> {
        let request: JsonRpcRequest = serde_json::from_str(request_str)?;
        let response = self.handle_request(&request);
        serde_json::to_string(&response)
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = McpServer::new();
        assert_eq!(server.name(), "god-graph-mcp");
    }

    #[test]
    fn test_initialize_request() {
        let server = McpServer::new();
        let request = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0"}}"#;

        let response = server.process_request(request).unwrap();
        let response_value: Value = serde_json::from_str(&response).unwrap();

        assert!(response_value.get("result").is_some());
        assert!(response_value["result"]["serverInfo"]["name"].as_str() == Some("god-graph-mcp"));
    }

    #[test]
    fn test_tools_list_request() {
        let server = McpServer::new();
        let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#;

        let response = server.process_request(request).unwrap();
        let response_value: Value = serde_json::from_str(&response).unwrap();

        assert!(response_value["result"]["tools"].is_array());
        let tools = response_value["result"]["tools"].as_array().unwrap();
        assert!(tools.len() > 0);

        // Check for expected tools
        let tool_names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert!(tool_names.contains(&"create_graph"));
        assert!(tool_names.contains(&"add_node"));
    }

    #[test]
    fn test_create_graph_tool() {
        let server = McpServer::new();
        let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"create_graph","arguments":{"graph_id":"test","directed":true}}}"#;

        let response = server.process_request(request).unwrap();
        let response_value: Value = serde_json::from_str(&response).unwrap();

        assert!(response_value.get("result").is_some());
        assert!(response_value["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("test"));
    }

    #[test]
    fn test_unknown_method() {
        let server = McpServer::new();
        let request = r#"{"jsonrpc":"2.0","id":1,"method":"unknown/method","params":{}}"#;

        let response = server.process_request(request).unwrap();
        let response_value: Value = serde_json::from_str(&response).unwrap();

        assert!(response_value.get("error").is_some());
        assert_eq!(response_value["error"]["code"].as_i64(), Some(-32601));
    }
}
