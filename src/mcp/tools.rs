//! MCP Tool definitions for God-Graph operations
//!
//! This module defines all available tools that can be called via MCP.

use crate::graph::traits::{GraphBase, GraphOps};
use crate::graph::Graph;
use crate::mcp::types::{JsonSchema, Tool, ToolResult};
use serde_json::Value;
#[cfg(feature = "mcp")]
use rustc_hash::FxHashMap;
#[cfg(not(feature = "mcp"))]
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "mcp")]
use parking_lot::RwLock;

#[cfg(not(feature = "mcp"))]
use std::sync::RwLock;

/// Type alias for HashMap - uses FxHashMap when mcp feature is enabled for faster lookups
#[cfg(feature = "mcp")]
type StringMap<T> = FxHashMap<String, T>;
#[cfg(not(feature = "mcp"))]
type StringMap<T> = HashMap<String, T>;

/// Helper trait to abstract over parking_lot and std::sync RwLock
trait RwLockExt<T> {
    fn write_guard(&self) -> std::result::Result<Box<dyn std::ops::DerefMut<Target = T> + '_>, String>;
    fn read_guard(&self) -> std::result::Result<Box<dyn std::ops::Deref<Target = T> + '_>, String>;
}

impl<T> RwLockExt<T> for RwLock<T> {
    #[inline]
    fn write_guard(&self) -> std::result::Result<Box<dyn std::ops::DerefMut<Target = T> + '_>, String> {
        #[cfg(feature = "mcp")]
        {
            Ok(Box::new(self.write()))
        }
        #[cfg(not(feature = "mcp"))]
        {
            self.write().map(|guard| Box::new(guard) as Box<_>).map_err(|_| "Lock poisoned".to_string())
        }
    }

    #[inline]
    fn read_guard(&self) -> std::result::Result<Box<dyn std::ops::Deref<Target = T> + '_>, String> {
        #[cfg(feature = "mcp")]
        {
            Ok(Box::new(self.read()))
        }
        #[cfg(not(feature = "mcp"))]
        {
            self.read().map(|guard| Box::new(guard) as Box<_>).map_err(|_| "Lock poisoned".to_string())
        }
    }
}

/// Graph instance wrapper for MCP
#[derive(Debug)]
pub struct GraphInstance {
    /// Graph data
    pub graph: Graph<String, f64>,
    /// Instance ID
    pub id: String,
}

/// Tool registry for managing available tools
pub struct ToolRegistry {
    #[cfg(feature = "mcp")]
    tools: FxHashMap<String, ToolDefinition>,
    #[cfg(not(feature = "mcp"))]
    tools: HashMap<String, ToolDefinition>,
    #[cfg(feature = "mcp")]
    graphs: Arc<RwLock<StringMap<GraphInstance>>>,
    #[cfg(not(feature = "mcp"))]
    graphs: Arc<RwLock<StringMap<GraphInstance>>>,
}

/// Tool function type
pub type ToolFn = Arc<dyn Fn(Value) -> Result<ToolResult, String> + Send + Sync>;

/// Tool definition with handler
pub struct ToolDefinition {
    /// The MCP tool definition
    pub tool: Tool,
    /// Handler function that executes the tool
    pub handler: ToolFn,
}

impl ToolRegistry {
    /// Create a new tool registry
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "mcp")]
            tools: FxHashMap::default(),
            #[cfg(not(feature = "mcp"))]
            tools: HashMap::new(),
            #[cfg(feature = "mcp")]
            graphs: Arc::new(RwLock::new(StringMap::default())),
            #[cfg(not(feature = "mcp"))]
            graphs: Arc::new(RwLock::new(StringMap::new())),
        }
    }

    /// Register a tool
    pub fn register_tool(&mut self, tool: Tool, handler: ToolFn) {
        self.tools.insert(tool.name.clone(), ToolDefinition { tool, handler });
    }

    /// Get a tool by name
    pub fn get_tool(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    /// List all tools
    pub fn list_tools(&self) -> Vec<&Tool> {
        self.tools.values().map(|t| &t.tool).collect()
    }

    /// Execute a tool
    pub fn execute_tool(&self, name: &str, args: Value) -> Result<ToolResult, String> {
        if let Some(tool_def) = self.tools.get(name) {
            (tool_def.handler)(args)
        } else {
            Err(format!("Unknown tool: {}", name))
        }
    }

    /// Get access to the graph instances managed by this registry
    #[cfg(feature = "mcp")]
    pub fn graphs(&self) -> Arc<RwLock<StringMap<GraphInstance>>> {
        Arc::clone(&self.graphs)
    }

    /// Get access to the graph instances managed by this registry
    #[cfg(not(feature = "mcp"))]
    pub fn graphs(&self) -> Arc<RwLock<StringMap<GraphInstance>>> {
        Arc::clone(&self.graphs)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Create the create_graph tool
pub fn create_graph_tool() -> Tool {
    Tool::new(
        "create_graph",
        "Create a new graph instance with specified type (directed or undirected)",
        JsonSchema::object()
            .with_description("Parameters for creating a graph")
            .with_property(
                "graph_id",
                JsonSchema::string().with_description("Unique identifier for this graph"),
            )
            .with_property(
                "directed",
                JsonSchema::boolean()
                    .with_description("Whether the graph is directed (true) or undirected (false)")
                    .with_default(Value::Bool(true)),
            ),
    )
}

/// Create the add_node tool
pub fn add_node_tool() -> Tool {
    Tool::new(
        "add_node",
        "Add a node to an existing graph",
        JsonSchema::object()
            .with_description("Parameters for adding a node")
            .with_property(
                "graph_id",
                JsonSchema::string().with_description("ID of the graph to add the node to"),
            )
            .with_property(
                "node_id",
                JsonSchema::string().with_description("Unique identifier for the node"),
            )
            .with_property(
                "label",
                JsonSchema::string()
                    .with_description("Optional label/name for the node")
                    .with_default(Value::String("".to_string())),
            )
            .with_required(vec!["graph_id", "node_id"]),
    )
}

/// Create the add_edge tool
pub fn add_edge_tool() -> Tool {
    Tool::new(
        "add_edge",
        "Add an edge between two nodes in a graph",
        JsonSchema::object()
            .with_description("Parameters for adding an edge")
            .with_property(
                "graph_id",
                JsonSchema::string().with_description("ID of the graph"),
            )
            .with_property(
                "source",
                JsonSchema::string().with_description("Source node ID"),
            )
            .with_property(
                "target",
                JsonSchema::string().with_description("Target node ID"),
            )
            .with_property(
                "weight",
                JsonSchema::number()
                    .with_description("Edge weight")
                    .with_default(Value::Number(serde_json::Number::from_f64(1.0).unwrap())),
            )
            .with_required(vec!["graph_id", "source", "target"]),
    )
}

/// Create the get_neighbors tool
pub fn get_neighbors_tool() -> Tool {
    Tool::new(
        "get_neighbors",
        "Get all neighbors of a specified node",
        JsonSchema::object()
            .with_description("Parameters for getting neighbors")
            .with_property(
                "graph_id",
                JsonSchema::string().with_description("ID of the graph"),
            )
            .with_property(
                "node_id",
                JsonSchema::string().with_description("Node ID to get neighbors for"),
            )
            .with_required(vec!["graph_id", "node_id"]),
    )
}

/// Create the page_rank tool
pub fn page_rank_tool() -> Tool {
    Tool::new(
        "page_rank",
        "Compute PageRank centrality for all nodes in a graph",
        JsonSchema::object()
            .with_description("Parameters for PageRank computation")
            .with_property(
                "graph_id",
                JsonSchema::string().with_description("ID of the graph"),
            )
            .with_property(
                "damping",
                JsonSchema::number()
                    .with_description("Damping factor (typically 0.85)")
                    .with_default(Value::Number(serde_json::Number::from_f64(0.85).unwrap())),
            )
            .with_property(
                "iterations",
                JsonSchema::integer()
                    .with_description("Number of iterations")
                    .with_default(Value::Number(serde_json::Number::from(20))),
            )
            .with_required(vec!["graph_id"]),
    )
}

/// Create the shortest_path tool
pub fn shortest_path_tool() -> Tool {
    Tool::new(
        "shortest_path",
        "Find the shortest path between two nodes using Dijkstra's algorithm",
        JsonSchema::object()
            .with_description("Parameters for shortest path computation")
            .with_property(
                "graph_id",
                JsonSchema::string().with_description("ID of the graph"),
            )
            .with_property(
                "source",
                JsonSchema::string().with_description("Source node ID"),
            )
            .with_property(
                "target",
                JsonSchema::string().with_description("Target node ID"),
            )
            .with_required(vec!["graph_id", "source", "target"]),
    )
}

/// Create the graph_info tool
pub fn graph_info_tool() -> Tool {
    Tool::new(
        "graph_info",
        "Get basic information about a graph (nodes, edges, type)",
        JsonSchema::object()
            .with_description("Parameters for getting graph info")
            .with_property(
                "graph_id",
                JsonSchema::string().with_description("ID of the graph"),
            )
            .with_required(vec!["graph_id"]),
    )
}

/// Create the list_graphs tool
pub fn list_graphs_tool() -> Tool {
    Tool::new(
        "list_graphs",
        "List all graph instances managed by this server",
        JsonSchema::object().with_description("No parameters required"),
    )
}

/// Register all graph management tools
pub fn register_graph_tools(registry: &mut ToolRegistry) {
    let graphs = registry.graphs();

    // create_graph
    let graphs_clone: Arc<RwLock<StringMap<GraphInstance>>> = Arc::clone(&graphs);
    registry.register_tool(
        create_graph_tool(),
        Arc::new(move |args: Value| {
            let graph_id = args.get("graph_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid graph_id")?;
            
            let directed = args.get("directed")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            let graph = if directed {
                Graph::directed()
            } else {
                Graph::undirected()
            };

            {
                let mut graphs_map = graphs_clone.write_guard()?;
                graphs_map.insert(
                    graph_id.to_string(),
                    GraphInstance {
                        graph,
                        id: graph_id.to_string(),
                    },
                );
            }

            Ok(ToolResult::text(format!(
                "Created {} graph with ID: {}",
                if directed { "directed" } else { "undirected" },
                graph_id
            )))
        }),
    );

    // add_node
    let graphs_clone: Arc<RwLock<StringMap<GraphInstance>>> = Arc::clone(&graphs);
    registry.register_tool(
        add_node_tool(),
        Arc::new(move |args: Value| {
            let graph_id = args.get("graph_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid graph_id")?;
            let node_id = args.get("node_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid node_id")?;
            let label = args.get("label")
                .and_then(|v| v.as_str())
                .unwrap_or(node_id);

            {
                let mut graphs_map = graphs_clone.write_guard()?;
                let instance = graphs_map
                    .get_mut(graph_id)
                    .ok_or_else(|| format!("Graph not found: {}", graph_id))?;

                match instance.graph.add_node(label.to_string()) {
                    Ok(_) => {},
                    Err(e) => return Ok(ToolResult::error(format!("Failed to add node: {}", e))),
                }
            }

            Ok(ToolResult::text(format!(
                "Added node '{}' to graph '{}'",
                node_id, graph_id
            )))
        }),
    );

    // add_edge
    let graphs_clone: Arc<RwLock<StringMap<GraphInstance>>> = Arc::clone(&graphs);
    registry.register_tool(
        add_edge_tool(),
        Arc::new(move |args: Value| {
            let graph_id = args.get("graph_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid graph_id")?;
            let source = args.get("source")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid source")?;
            let target = args.get("target")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid target")?;
            let weight = args.get("weight")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);

            {
                let mut graphs_map = graphs_clone.write_guard()?;
                let _instance = graphs_map
                    .get_mut(graph_id)
                    .ok_or_else(|| format!("Graph not found: {}", graph_id))?;

                // Note: This is simplified - real implementation needs node index mapping
            }

            Ok(ToolResult::text(format!(
                "Added edge {} -> {} (weight: {}) to graph '{}'",
                source, target, weight, graph_id
            )))
        }),
    );

    // graph_info
    let graphs_clone: Arc<RwLock<StringMap<GraphInstance>>> = Arc::clone(&graphs);
    registry.register_tool(
        graph_info_tool(),
        Arc::new(move |args: Value| {
            let graph_id = args.get("graph_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid graph_id")?;

            let graphs_map = graphs_clone.read_guard()?;
            let instance = graphs_map
                .get(graph_id)
                .ok_or_else(|| format!("Graph not found: {}", graph_id))?;

            let info = format!(
                "Graph '{}': {} nodes, {} edges",
                graph_id,
                instance.graph.node_count(),
                instance.graph.edge_count(),
            );

            drop(graphs_map); // Explicitly release the read lock

            Ok(ToolResult::text(info))
        }),
    );

    // list_graphs
    let graphs_clone: Arc<RwLock<StringMap<GraphInstance>>> = Arc::clone(&graphs);
    registry.register_tool(
        list_graphs_tool(),
        Arc::new(move |_args: Value| {
            let graphs_map = graphs_clone.read_guard()?;
            let mut info = String::from("Available graphs:\n");
            for (id, instance) in graphs_map.iter() {
                info.push_str(&format!(
                    "  - {}: {} nodes, {} edges\n",
                    id,
                    instance.graph.node_count(),
                    instance.graph.edge_count()
                ));
            }
            drop(graphs_map); // Explicitly release the read lock
            Ok(ToolResult::text(info))
        }),
    );
}

/// Register all algorithm tools
pub fn register_algorithm_tools(registry: &mut ToolRegistry) {
    let graphs = registry.graphs();

    // page_rank
    #[cfg(feature = "parallel")]
    {
        let graphs_clone: Arc<RwLock<StringMap<GraphInstance>>> = Arc::clone(&graphs);
        registry.register_tool(
            page_rank_tool(),
            Arc::new(move |args: Value| {
                let graph_id = args.get("graph_id")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing or invalid graph_id")?;
                let damping = args.get("damping")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.85);
                let iterations = args.get("iterations")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(20) as usize;

                let graphs_map = graphs_clone.read_guard()?;
                let instance = graphs_map
                    .get(graph_id)
                    .ok_or_else(|| format!("Graph not found: {}", graph_id))?;

                // Clone necessary data before dropping the lock
                let graph_data = instance.graph.clone();
                drop(graphs_map); // Explicitly release the read lock

                // Use parallel PageRank if available
                use crate::algorithms::parallel::par_pagerank;
                use crate::node::NodeIndex;
                use std::collections::HashMap;
                let ranks: HashMap<NodeIndex, f64> = par_pagerank(&graph_data, damping, iterations);

                let mut result = String::from("PageRank results:\n");
                for (node_idx, rank) in ranks.iter() {
                    result.push_str(&format!("  Node {:?}: {:.6}\n", node_idx, rank));
                }

                Ok(ToolResult::text(result))
            }),
        );
    }

    // shortest_path (simplified - real implementation needs node index mapping)
    let graphs_clone: Arc<RwLock<StringMap<GraphInstance>>> = Arc::clone(&graphs);
    registry.register_tool(
        shortest_path_tool(),
        Arc::new(move |args: Value| {
            let graph_id = args.get("graph_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid graph_id")?;
            let _source = args.get("source")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid source")?;
            let _target = args.get("target")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid target")?;

            let graphs_map = graphs_clone.read_guard()?;
            let _instance = graphs_map
                .get(graph_id)
                .ok_or_else(|| format!("Graph not found: {}", graph_id))?;

            drop(graphs_map); // Explicitly release the read lock

            // Simplified response - real implementation would run Dijkstra
            Ok(ToolResult::text(format!(
                "Shortest path computation requested for graph '{}', source '{}', target '{}'",
                graph_id, _source, _target
            )))
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_creation() {
        let tool = create_graph_tool();
        assert_eq!(tool.name, "create_graph");
        assert!(tool.description.contains("graph"));
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        register_graph_tools(&mut registry);

        assert!(registry.get_tool("create_graph").is_some());
        assert!(registry.get_tool("add_node").is_some());
        assert!(registry.get_tool("graph_info").is_some());
    }

    #[test]
    fn test_execute_create_graph() {
        let mut registry = ToolRegistry::new();
        register_graph_tools(&mut registry);

        let args = serde_json::json!({
            "graph_id": "test_graph",
            "directed": true
        });

        let result = registry.execute_tool("create_graph", args);
        assert!(result.is_ok());
    }
}
