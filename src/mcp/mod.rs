//! MCP (Model Context Protocol) Server for God-Graph
//!
//! This module provides an MCP server implementation that exposes God-Graph
//! operations as tools for AI assistants to interact with graph data.
//!
//! ## Features
//!
//! - **Graph Operations**: Create, modify, and query graphs
//! - **Algorithm Execution**: Run PageRank, shortest path, centrality algorithms
//! - **Tensor Operations**: GNN inference, tensor conversions
//! - **Visualization**: Export to DOT/Graphviz format
//!
//! ## Usage
//!
//! ```no_run
//! use god_graph::mcp::McpServer;
//!
//! let mut server = McpServer::new();
//! server.register_graph_tools();
//! server.register_algorithm_tools();
//!
//! // Run server (stdio transport)
//! server.run_stdio().expect("Failed to run server");
//! ```
//!
//! ## Available Tools
//!
//! ### Graph Management
//! - `create_graph`: Create a new graph (directed/undirected)
//! - `add_node`: Add a node to a graph
//! - `add_edge`: Add an edge to a graph
//! - `remove_node`: Remove a node from a graph
//! - `remove_edge`: Remove an edge from a graph
//! - `get_neighbors`: Get neighbors of a node
//! - `get_degree`: Get degree of a node
//!
//! ### Algorithms
//! - `page_rank`: Compute PageRank centrality
//! - `shortest_path`: Find shortest path between nodes
//! - `bfs`: Breadth-first search traversal
//! - `dfs`: Depth-first search traversal
//! - `connected_components`: Find connected components
//!
//! ### Tensor Operations (with `tensor` feature)
//! - `graph_to_tensor`: Convert graph to tensor representation
//! - `gnn_inference`: Run GNN inference on graph
//!
//! ## MCP Protocol
//!
//! The server implements the Model Context Protocol v1.0:
//! - JSON-RPC 2.0 message format
//! - Stdio transport (stdin/stdout)
//! - Tool discovery and execution
//! - Resource exposure (optional)

pub mod server;
pub mod tools;
pub mod types;

pub use server::McpServer;
pub use tools::ToolRegistry;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use serde_json::Value;

    #[test]
    fn test_mcp_server_creation() {
        let server = McpServer::new();
        assert_eq!(server.name(), "god-graph-mcp");
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register_tool(
            tools::create_graph_tool(),
            Arc::new(|_args: Value| Ok(ToolResult::text("ok"))),
        );
        assert!(registry.get_tool("create_graph").is_some());
    }
}
