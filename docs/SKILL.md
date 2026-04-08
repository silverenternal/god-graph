# God-Graph Skill Definition for AI Agents

> **Version**: 0.6.0-alpha
> **Last Updated**: 2026-04-06
> **Target**: AI Agents (Claude, GPT, etc.)
>
> This document defines how AI agents should use God-Graph library.
> It maps natural language intents to Rust API calls.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Skill Definitions](#skill-definitions)
3. [Intent Mapping](#intent-mapping)
4. [Code Templates](#code-templates)
5. [Error Handling](#error-handling)
6. [Usage Examples](#usage-examples)
7. [Feature Flags](#feature-flags)

---

## Quick Reference

### Core Skills

| Skill | Trigger Phrases | Primary API |
|-------|----------------|-------------|
| **Create Graph** | "create a graph", "build a network" | `Graph::<T, E>::directed/undirected()` |
| **Add Nodes/Edges** | "add a node", "connect nodes" | `graph.add_node()`, `graph.add_edge()` |
| **Traverse Graph** | "visit all nodes", "walk through" | `bfs()`, `dfs()` |
| **Find Shortest Path** | "shortest path", "minimum distance" | `dijkstra()`, `astar()`, `bellman_ford()` |
| **Centrality Analysis** | "important nodes", "key people", "influencers" | `pagerank()`, `betweenness_centrality()` |
| **Community Detection** | "find communities", "clusters", "groups" | `connected_components()`, `louvain()` |
| **Graph Properties** | "is connected", "has cycle", "density" | `is_connected()`, `has_cycle()`, `density()` |
| **Export/Visualize** | "export graph", "visualize" | `to_dot()`, `to_adjacency_list()` |

### Advanced Skills (Tensor/LLM)

| Skill | Trigger Phrases | Primary API |
|-------|----------------|-------------|
| **Create Tensor** | "create tensor", "matrix" | `DenseTensor::from_vec()` |
| **Tensor Operations** | "matrix multiply", "transpose" | `tensor.matmul()`, `tensor.t()` |
| **GNN Layers** | "GCN layer", "attention" | `gcn_conv()`, `gat_conv()` |
| **Load LLM Model** | "load model", "import safetensors" | `ModelSwitch::load_from_safetensors()` |
| **Optimize LLM** | "optimize model", "prune attention" | `CadStyleEditor`, `TensorRingCompressor` |

---

## Skill Definitions

### Skill: Graph Creation

**Intent**: User wants to create a new graph structure

**Trigger Phrases**:
- "Create a graph"
- "Build a network"
- "Initialize a graph"
- "Make a directed/undirected graph"

**Parameters**:
- `graph_type`: `"directed"` | `"undirected"` (default: `"directed"`)
- `node_data_type`: Rust type name (e.g., `"String"`, `"i32"`, `"MyStruct"`)
- `edge_data_type`: Rust type name (e.g., `"f64"`, `"Weight"`, `"f32"`)

**Code Template**:
```rust
use god_graph::graph::Graph;

// For directed graph with String nodes and f64 edges
let mut graph = Graph::<String, f64>::directed();

// For undirected graph
let mut graph = Graph::<String, f64>::undirected();
```

**Example**:
- **User**: "创建一个有向图，节点是字符串，边是权重"
- **AI Action**:
```rust
use god_graph::graph::Graph;
let mut graph = Graph::<String, f64>::directed();
```

---

### Skill: Add Nodes

**Intent**: User wants to add nodes to a graph

**Trigger Phrases**:
- "Add a node"
- "Insert a vertex"
- "Create a node with data"

**Parameters**:
- `graph`: mutable reference to `Graph<T, E>`
- `node_data`: data to store in node

**Code Template**:
```rust
// Add single node
let node_a = graph.add_node("A".to_string()).unwrap();

// Add multiple nodes
let nodes: Vec<_> = (0..10)
    .map(|i| graph.add_node(format!("node_{}", i)).unwrap())
    .collect();
```

**Example**:
- **User**: "添加三个节点 A, B, C"
- **AI Action**:
```rust
let a = graph.add_node("A".to_string()).unwrap();
let b = graph.add_node("B".to_string()).unwrap();
let c = graph.add_node("C".to_string()).unwrap();
```

---

### Skill: Add Edges

**Intent**: User wants to connect nodes with edges

**Trigger Phrases**:
- "Add an edge"
- "Connect two nodes"
- "Link node A to B"
- "Add a relationship"

**Parameters**:
- `from`: source `NodeIndex`
- `to`: target `NodeIndex`
- `edge_data`: data to store in edge (e.g., weight)

**Code Template**:
```rust
// Add edge with weight
graph.add_edge(node_a, node_b, 1.0).unwrap();

// Add edge without data (use ())
graph.add_edge(node_a, node_b, ()).unwrap();
```

**Example**:
- **User**: "连接节点 A 到 B，权重为 2.5"
- **AI Action**:
```rust
graph.add_edge(a, b, 2.5).unwrap();
```

---

### Skill: Graph Traversal

**Intent**: User wants to visit all nodes in a specific order

**Trigger Phrases**:
- "Visit all nodes"
- "Traverse the graph"
- "Walk through"
- "BFS/DFS traversal"

**Parameters**:
- `graph`: reference to `Graph<T, E>`
- `start_node`: starting `NodeIndex`
- `algorithm`: `"bfs"` | `"dfs"` (default: `"bfs"`)

**Code Template**:
```rust
use god_graph::algorithms::traversal::{bfs, dfs};

// BFS traversal
bfs(&graph, start_node, |node, depth| {
    println!("Node: {} at depth {}", graph[node].data(), depth);
    true // continue traversal
});

// DFS traversal
dfs(&graph, start_node, |node, depth| {
    println!("Node: {} at depth {}", graph[node].data(), depth);
    true // continue traversal
});
```

**Example**:
- **User**: "从节点 A 开始 BFS 遍历"
- **AI Action**:
```rust
use god_graph::algorithms::traversal::bfs;

bfs(&graph, start_node, |node, _depth| {
    println!("Visited: {}", graph[node].data());
    true
});
```

---

### Skill: Shortest Path

**Intent**: User wants to find the shortest path between nodes

**Trigger Phrases**:
- "Shortest path"
- "Minimum distance"
- "Find path from A to B"
- "Calculate distance"

**Parameters**:
- `graph`: reference to `Graph<T, E>`
- `start`: source `NodeIndex`
- `end`: target `NodeIndex`
- `algorithm`: `"dijkstra"` | `"astar"` | `"bellman_ford"` (default: `"dijkstra"`)

**Code Template**:
```rust
use god_graph::algorithms::shortest_path::{dijkstra, astar, bellman_ford};

// Dijkstra (non-negative weights)
let distances = dijkstra(&graph, start_node);
let distance_to_b = distances.get(&end_node).copied();

// A* (with heuristic)
let path = astar(&graph, start_node, end_node, |n| heuristic(n));

// Bellman-Ford (handles negative weights)
let (distances, _) = bellman_ford(&graph, start_node);
```

**Decision Tree**:
- If weights are non-negative → use `dijkstra()`
- If negative weights exist → use `bellman_ford()`
- If heuristic function available → use `astar()`

**Example**:
- **User**: "找到从 A 到 B 的最短路径"
- **AI Action**:
```rust
use god_graph::algorithms::shortest_path::dijkstra;

let distances = dijkstra(&graph, a);
if let Some(dist) = distances.get(&b) {
    println!("Shortest distance: {}", dist);
}
```

---

### Skill: Centrality Analysis

**Intent**: User wants to find important/influential nodes

**Trigger Phrases**:
- "Important nodes"
- "Key people"
- "Influencers"
- "Central nodes"
- "Find hubs"
- "Rank nodes by importance"

**Parameters**:
- `graph`: reference to `Graph<T, E>`
- `metric`: `"pagerank"` | `"betweenness"` | `"closeness"` | `"degree"` (default: `"pagerank"`)

**Code Template**:
```rust
use god_graph::algorithms::centrality::{pagerank, betweenness_centrality, closeness_centrality, degree_centrality};

// PageRank (best for directed graphs, influence ranking)
let ranks = pagerank(&graph, 0.85, 20);
let top_node = ranks.iter().enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .map(|(i, _)| i);

// Betweenness Centrality (bridge nodes)
let centrality = betweenness_centrality(&graph);

// Closeness Centrality (nodes close to all others)
let closeness = closeness_centrality(&graph);

// Degree Centrality (most connections)
let degree = degree_centrality(&graph);
```

**Decision Tree**:
- If user mentions "influence" or "importance" → use `pagerank()`
- If user mentions "bridge" or "connector" → use `betweenness_centrality()`
- If user mentions "reachability" → use `closeness_centrality()`
- If user mentions "connections" or "degree" → use `degree_centrality()`

**Example**:
- **User**: "找出这个社交网络中最重要的人"
- **AI Action**:
```rust
use god_graph::algorithms::centrality::pagerank;

let ranks = pagerank(&graph, 0.85, 20);
let top_5: Vec<_> = ranks.iter().enumerate()
    .map(|(i, &rank)| (i, rank))
    .collect();

println!("Top 5 most important nodes: {:?}", top_5);
```

---

### Skill: Community Detection

**Intent**: User wants to find clusters or groups in the graph

**Trigger Phrases**:
- "Find communities"
- "Detect clusters"
- "Find groups"
- "Identify connected components"
- "Segment the network"

**Parameters**:
- `graph`: reference to `Graph<T, E>`
- `algorithm`: `"connected_components"` | `"louvain"` (default: `"connected_components"`)

**Code Template**:
```rust
use god_graph::algorithms::community::{connected_components, louvain};

// Connected Components (basic clustering)
let components = connected_components(&graph);
let num_components = components.len();

// Louvain (modularity-based, detects overlapping communities)
let communities = louvain(&graph);
```

**Decision Tree**:
- If user wants basic connectivity → use `connected_components()`
- If user wants modularity-based communities → use `louvain()`

**Example**:
- **User**: "找出图中的所有社区"
- **AI Action**:
```rust
use god_graph::algorithms::community::connected_components;

let components = connected_components(&graph);
println!("Found {} communities", components.len());

for (i, component) in components.iter().enumerate() {
    println!("Community {}: {:?} nodes", i, component.len());
}
```

---

### Skill: Graph Properties

**Intent**: User wants to check graph characteristics

**Trigger Phrases**:
- "Is connected"
- "Has cycle"
- "What's the density"
- "Is it a tree"
- "Is it a DAG"
- "What's the diameter"

**Parameters**:
- `graph`: reference to `Graph<T, E>`
- `property`: `"connected"` | `"cycle"` | `"density"` | `"tree"` | `"dag"` | `"diameter"`

**Code Template**:
```rust
use god_graph::algorithms::properties::{
    is_connected, has_cycle, density, is_tree, is_dag, diameter
};

// Check connectivity
let connected = is_connected(&graph);

// Check for cycles
let has_cycle = has_cycle(&graph);

// Calculate density
let dens = density(&graph);

// Check if tree
let is_tree = is_tree(&graph);

// Check if DAG (Directed Acyclic Graph)
let is_dag = is_dag(&graph);

// Calculate diameter
let diam = diameter(&graph);
```

**Example**:
- **User**: "这个图是连通的吗？有环吗？"
- **AI Action**:
```rust
use god_graph::algorithms::properties::{is_connected, has_cycle};

let connected = is_connected(&graph);
let cyclic = has_cycle(&graph);

println!("Connected: {}, Has cycle: {}", connected, cyclic);
```

---

### Skill: Export/Visualize

**Intent**: User wants to export or visualize the graph

**Trigger Phrases**:
- "Export graph"
- "Visualize the graph"
- "Save to file"
- "Generate DOT"
- "Draw the graph"

**Parameters**:
- `graph`: reference to `Graph<T, E>`
- `format`: `"dot"` | `"adjacency_list"` | `"edge_list"` (default: `"dot"`)

**Code Template**:
```rust
use god_graph::export::{to_dot, to_adjacency_list, to_edge_list};

// Export to DOT format (Graphviz)
let dot = to_dot(&graph);
std::fs::write("graph.dot", dot)?;

// Export to adjacency list
let adj_list = to_adjacency_list(&graph);

// Export to edge list
let edge_list = to_edge_list(&graph);
```

**Example**:
- **User**: "导出为 DOT 格式，我想可视化"
- **AI Action**:
```rust
use god_graph::export::to_dot;

let dot = to_dot(&graph);
std::fs::write("output.dot", dot)?;
println!("Graph exported to output.dot");
println!("Run: dot -Tpng output.dot -o output.png");
```

---

### Skill: Create Tensor (Tensor Feature)

**Intent**: User wants to create a tensor for numerical computation

**Trigger Phrases**:
- "Create a tensor"
- "Make a matrix"
- "N-dimensional array"

**Parameters**:
- `data`: vector of values
- `shape`: vector of dimensions
- `dtype`: `"f32"` | `"f64"` (default: `"f64"`)

**Code Template**:
```rust
use god_graph::tensor::{DenseTensor, TensorBase};

// Create from vector
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let tensor = DenseTensor::from_vec(data, vec![2, 3]);

// Create zeros
let zeros = DenseTensor::zeros(vec![3, 3]);

// Create ones
let ones = DenseTensor::ones(vec![2, 4]);
```

**Example**:
- **User**: "创建一个 2x3 的张量，数据是 1 到 6"
- **AI Action**:
```rust
use god_graph::tensor::DenseTensor;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let tensor = DenseTensor::from_vec(data, vec![2, 3]);
```

---

### Skill: Tensor Operations (Tensor Feature)

**Intent**: User wants to perform tensor/matrix operations

**Trigger Phrases**:
- "Matrix multiply"
- "Transpose"
- "Add tensors"
- "Element-wise multiply"
- "Sum/mean"

**Parameters**:
- `operation`: `"matmul"` | `"transpose"` | `"add"` | `"mul"` | `"sum"` | `"mean"`

**Code Template**:
```rust
use god_graph::tensor::{DenseTensor, TensorBase};

// Matrix multiplication
let c = a.matmul(&b);

// Transpose
let t = tensor.t();

// Element-wise operations
let sum = a.add(&b);
let prod = a.mul(&b);

// Reduction
let total = tensor.sum();
let avg = tensor.mean();
```

**Example**:
- **User**: "计算两个矩阵的乘积"
- **AI Action**:
```rust
let a = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = DenseTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

let c = a.matmul(&b);
println!("Result: {:?}", c.data());
```

---

### Skill: Load LLM Model (Transformer Feature)

**Intent**: User wants to load a model from Safetensors format

**Trigger Phrases**:
- "Load model"
- "Import Safetensors"
- "Load LLM weights"

**Parameters**:
- `path`: path to `.safetensors` file

**Code Template**:
```rust
use god_graph::transformer::optimization::ModelSwitch;

// Load model
let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// Validate topology
let report = ModelSwitch::validate_topology(&graph)?;
println!("Topology valid: {}", report.is_valid);
```

**Example**:
- **User**: "加载 model.safetensors 文件"
- **AI Action**:
```rust
use god_graph::transformer::optimization::ModelSwitch;

let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;
println!("Loaded model with {} nodes", graph.node_count());
```

---

### Skill: Optimize LLM (Transformer Feature)

**Intent**: User wants to optimize an LLM model's topology

**Trigger Phrases**:
- "Optimize model"
- "Prune attention"
- "Compress weights"
- "Detect defects"

**Parameters**:
- `optimization_type`: `"orthogonalize"` | `"compress"` | `"detect_defects"`

**Code Template**:
```rust
use god_graph::transformer::optimization::{
    CadStyleEditor, TensorRingCompressor, LieGroupOptimizer
};

// Detect topology defects
let mut editor = CadStyleEditor::new(&mut graph);
let defects = editor.detect_defects()?;
println!("Found {} defects", defects.len());

// Apply tensor ring compression
let compressor = TensorRingCompressor::default();
let compressed = compressor.compress_graph(&graph)?;

// Lie group orthogonalization
let optimizer = LieGroupOptimizer::default();
optimizer.orthogonalize_weights(&mut graph)?;
```

**Example**:
- **User**: "检测模型的拓扑缺陷"
- **AI Action**:
```rust
use god_graph::transformer::optimization::CadStyleEditor;

let mut editor = CadStyleEditor::new(&mut graph);
let defects = editor.detect_defects()?;

for defect in &defects {
    println!("Defect: {:?}", defect);
}
```

---

## Intent Mapping

### Graph Analysis Intents

| User Intent | Trigger Phrases | Mapped API |
|-------------|----------------|------------|
| **Find important nodes** | "key people", "influencers", "central nodes" | `pagerank()`, `betweenness_centrality()` |
| **Find communities** | "clusters", "groups", "segments" | `connected_components()`, `louvain()` |
| **Find shortest path** | "minimum distance", "path from A to B" | `dijkstra()`, `astar()` |
| **Check connectivity** | "is connected", "can reach" | `is_connected()`, `bfs()` |
| **Detect cycles** | "has cycle", "is acyclic" | `has_cycle()`, `is_dag()` |
| **Graph statistics** | "density", "diameter", "avg degree" | `density()`, `diameter()` |

### LLM Optimization Intents

| User Intent | Trigger Phrases | Mapped API |
|-------------|----------------|------------|
| **Load model** | "import model", "load weights" | `ModelSwitch::load_from_safetensors()` |
| **Validate model** | "check topology", "validate" | `ModelSwitch::validate_topology()` |
| **Compress model** | "reduce size", "compress" | `TensorRingCompressor` |
| **Detect issues** | "find problems", "defects" | `CadStyleEditor::detect_defects()` |
| **Stabilize weights** | "orthogonalize", "stabilize" | `LieGroupOptimizer` |

---

## Code Templates

### Template 1: Basic Graph Analysis

```rust
use god_graph::prelude::*;

fn analyze_graph<T, E>(graph: &Graph<T, E>) -> GraphResult<()>
where
    T: Clone,
    E: Clone,
{
    // 1. Basic properties
    println!("Nodes: {}", graph.node_count());
    println!("Edges: {}", graph.edge_count());
    println!("Density: {:.4}", density(graph));
    println!("Is connected: {}", is_connected(graph));
    println!("Has cycle: {}", has_cycle(graph));
    
    // 2. Centrality analysis
    let ranks = pagerank(graph, 0.85, 20);
    let top_node = ranks.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("Most central node: index {}", top_node.0);
    
    // 3. Community detection
    let components = connected_components(graph);
    println!("Connected components: {}", components.len());
    
    Ok(())
}
```

### Template 2: Shortest Path Workflow

```rust
use god_graph::prelude::*;

fn find_shortest_path(
    graph: &Graph<String, f64>,
    from: &str,
    to: &str,
) -> GraphResult<Option<f64>> {
    // Find start and end nodes
    let start = graph.nodes()
        .find(|n| n.data() == from)
        .map(|n| n.index());
    
    let end = graph.nodes()
        .find(|n| n.data() == to)
        .map(|n| n.index());
    
    let (Some(start), Some(end)) = (start, end) else {
        return Ok(None);
    };
    
    // Run Dijkstra
    let distances = dijkstra(graph, start);
    
    Ok(distances.get(&end).copied())
}
```

### Template 3: GNN Forward Pass

```rust
use god_graph::tensor::{DenseTensor, TensorBase};
use god_graph::tensor::gnn::{gcn_conv, gat_conv};

fn gnn_forward_pass(
    node_features: &DenseTensor,
    adjacency: &DenseTensor,
) -> DenseTensor {
    // GCN convolution
    let hidden = gcn_conv(node_features, adjacency);
    
    // ReLU activation
    let activated = hidden.relu();
    
    // Output projection
    let output = activated.matmul(&weight_matrix);
    
    output
}
```

---

## Error Handling

### Error Type Reference

#### VgiError (VGI Layer)

| Error Variant | When It Occurs | AI Response Template |
|---------------|----------------|---------------------|
| `UnsupportedCapability` | Backend doesn't support requested feature | "The current backend doesn't support {capability}. Try using a different backend or algorithm." |
| `PluginNotFound` | Requested plugin not registered | "The plugin '{name}' is not available. Make sure it's registered first." |
| `BackendInitializationFailed` | Backend failed to initialize | "Failed to initialize the backend. Check your configuration." |

#### GraphError (Core Layer)

| Error Variant | When It Occurs | AI Response Template |
|---------------|----------------|---------------------|
| `NodeNotFound` | Invalid node index | "Node {index} not found. It may have been deleted or never existed." |
| `EdgeNotFound` | Invalid edge index | "Edge {index} not found." |
| `NodeDeleted` | Using stale node index | "This node was deleted. The provided generation {provided} doesn't match current {current}." |
| `CycleDetected` | Operation requires DAG but graph has cycle | "This operation requires a DAG, but the graph contains a cycle." |

### Error Recovery Patterns

#### Pattern 1: Node Not Found Recovery

```rust
// When AI gets NodeNotFound error
match graph.get_node(index) {
    Ok(node) => { /* proceed */ }
    Err(GraphError::NodeNotFound { index }) => {
        // Recovery: list available nodes
        println!("Available nodes:");
        for node in graph.nodes() {
            println!("  Index: {:?}, Data: {}", node.index(), node.data());
        }
        // Ask user for correct index
    }
}
```

#### Pattern 2: Cycle Detected Recovery

```rust
// When AI gets CycleDetected error for topological sort
match topological_sort(&graph) {
    Ok(order) => { /* proceed */ }
    Err(GraphError::CycleDetected) => {
        // Recovery: find and report the cycle
        println!("Graph contains a cycle. Topological sort requires a DAG.");
        // Option: use Tarjan's SCC to find cycles
        let sccs = tarjan_scc(&graph);
        for scc in sccs.iter().filter(|c| c.len() > 1) {
            println!("Cycle involves nodes: {:?}", scc);
        }
    }
}
```

---

## Usage Examples

### Example 1: Social Network Analysis

**User**: "I have a social network. Find the most influential people and detect communities."

**AI Thought**: User wants centrality analysis (PageRank for influence) and community detection.

**AI Action**:
```rust
use god_graph::prelude::*;

// 1. Create graph
let mut graph = Graph::<String, f64>::undirected();

// 2. Add nodes (people)
let alice = graph.add_node("Alice".to_string()).unwrap();
let bob = graph.add_node("Bob".to_string()).unwrap();
let charlie = graph.add_node("Charlie".to_string()).unwrap();
let david = graph.add_node("David".to_string()).unwrap();

// 3. Add edges (relationships)
graph.add_edge(alice, bob, 1.0).unwrap();
graph.add_edge(alice, charlie, 1.0).unwrap();
graph.add_edge(bob, charlie, 1.0).unwrap();
graph.add_edge(charlie, david, 1.0).unwrap();

// 4. PageRank for influence
let ranks = pagerank(&graph, 0.85, 20);
println!("PageRank scores: {:?}", ranks);

// 5. Community detection
let components = connected_components(&graph);
println!("Found {} communities", components.len());
```

**AI Response**: "Analysis complete! Charlie has the highest PageRank score (most influential). The network has 1 connected component (everyone is connected)."

---

### Example 2: Route Planning

**User**: "Find the shortest path from city A to city B with distances as weights."

**AI Thought**: User wants shortest path with weighted edges. Use Dijkstra.

**AI Action**:
```rust
use god_graph::prelude::*;

// 1. Create graph with cities and distances
let mut graph = Graph::<String, f64>::directed();

let a = graph.add_node("City A".to_string()).unwrap();
let b = graph.add_node("City B".to_string()).unwrap();
let c = graph.add_node("City C".to_string()).unwrap();

// 2. Add edges with distances
graph.add_edge(a, c, 100.0).unwrap();  // A -> C: 100km
graph.add_edge(c, b, 50.0).unwrap();   // C -> B: 50km
graph.add_edge(a, b, 200.0).unwrap();  // A -> B: 200km (direct)

// 3. Find shortest path
let distances = dijkstra(&graph, a);

if let Some(dist) = distances.get(&b) {
    println!("Shortest distance from A to B: {:.1}km", dist);
    // Expected: 150km via C
}
```

**AI Response**: "The shortest path from City A to City B is 150km (via City C), which is shorter than the direct route of 200km."

---

### Example 3: LLM Model Validation

**User**: "Load this model and check if its topology is valid."

**AI Thought**: User wants to load a Safetensors model and validate its structure.

**AI Action**:
```rust
use god_graph::transformer::optimization::ModelSwitch;

// 1. Load model
let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// 2. Validate topology
let report = ModelSwitch::validate_topology(&graph)?;

println!("Topology Validation Report:");
println!("  Valid: {}", report.is_valid);
println!("  Is DAG: {}", report.is_dag);
println!("  Connected components: {}", report.connected_components);
println!("  Isolated nodes: {}", report.isolated_nodes);

// 3. Check for defects
if !report.is_valid {
    println!("\nIssues detected:");
    for issue in &report.issues {
        println!("  - {}", issue);
    }
}
```

**AI Response**: "Model loaded successfully. Topology is valid: it's a DAG with 1 connected component and no isolated nodes."

---

## Feature Flags

### Required Features for Each Skill

| Skill | Required Feature | Cargo Command |
|-------|------------------|---------------|
| **Basic Graph** | (none, always enabled) | `cargo build` |
| **Parallel Algorithms** | `parallel` | `cargo build --features parallel` |
| **Tensor Operations** | `tensor` | `cargo build --features tensor` |
| **Sparse Tensors** | `tensor-sparse` | `cargo build --features tensor-sparse` |
| **GNN Layers** | `tensor-gnn` | `cargo build --features tensor-gnn` |
| **Memory Pool** | `tensor-pool` | `cargo build --features tensor-pool` |
| **Transformer** | `transformer` | `cargo build --features transformer` |
| **Safetensors Loading** | `safetensors` | `cargo build --features safetensors` |
| **Full LLM Support** | `llm` | `cargo build --features llm` |
| **SIMD Optimization** | `simd` | `cargo build --features simd` |

### Recommended Feature Combinations

```toml
# For basic graph analysis
features = ["parallel"]

# For GNN development
features = ["parallel", "tensor", "tensor-sparse", "tensor-gnn", "tensor-pool"]

# For LLM optimization
features = ["parallel", "transformer", "safetensors", "tensor-pool"]

# Full features (for development)
features = ["parallel", "tensor-full", "llm", "simd"]
```

---

## Appendix: Complete API Reference

### Core Types

```rust
// Graph
Graph<T, E>                    // Main graph structure
NodeIndex                      // Node identifier
EdgeIndex                      // Edge identifier
NodeRef<'_, T>                 // Node reference
EdgeRef<'_, E>                 // Edge reference

// Traits
GraphBase                      // Base trait (node_count, edge_count)
GraphQuery                     // Query trait (get_node, neighbors)
GraphOps                       // Mutable operations (add_node, add_edge)
VirtualGraph                   // VGI trait (unified interface)
```

### Algorithm Functions

```rust
// Traversal
bfs(graph, start, visitor)     // Breadth-first search
dfs(graph, start, visitor)     // Depth-first search
topological_sort(graph)        // Topological ordering (DAG only)

// Shortest Path
dijkstra(graph, start)         // Dijkstra's algorithm
astar(graph, start, end, heuristic)  // A* search
bellman_ford(graph, start)     // Bellman-Ford (handles negative weights)

// Centrality
pagerank(graph, damping, iterations)  // PageRank
degree_centrality(graph)       // Degree centrality
betweenness_centrality(graph)  // Betweenness centrality
closeness_centrality(graph)    // Closeness centrality

// Community
connected_components(graph)    // Connected components
louvain(graph)                 // Louvain community detection

// Properties
is_connected(graph)            // Check connectivity
has_cycle(graph)               // Check for cycles
is_dag(graph)                  // Check if DAG
density(graph)                 // Calculate density
diameter(graph)                // Graph diameter
```

---

## Quick Copy-Paste Snippets

### Create and Analyze Graph
```rust
use god_graph::prelude::*;

let mut graph = Graph::<String, f64>::directed();
let a = graph.add_node("A".to_string()).unwrap();
let b = graph.add_node("B".to_string()).unwrap();
graph.add_edge(a, b, 1.0).unwrap();

println!("Nodes: {}", graph.node_count());
println!("Is connected: {}", is_connected(&graph));
```

### PageRank
```rust
use god_graph::algorithms::centrality::pagerank;

let ranks = pagerank(&graph, 0.85, 20);
```

### Shortest Path
```rust
use god_graph::algorithms::shortest_path::dijkstra;

let distances = dijkstra(&graph, start_node);
```

### Export to DOT
```rust
use god_graph::export::to_dot;

std::fs::write("graph.dot", to_dot(&graph))?;
```

---

**End of Skill Definition**

For more details, see:
- [API Documentation](https://docs.rs/god-graph)
- [User Guide](docs/user-guide/)
- [Performance Report](docs/reports/performance.md)
