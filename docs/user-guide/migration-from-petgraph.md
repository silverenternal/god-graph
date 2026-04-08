# Migration Guide: From petgraph to God-Graph

This guide helps you migrate from [petgraph](https://github.com/petgraph/petgraph) to God-Graph. God-Graph provides a more modern API, better performance, and built-in parallel algorithm support.

## Core Differences Overview

| Feature | petgraph | God-Graph |
|---------|----------|-----------|
| Graph Type | `Graph<N, E, Ty>` | `Graph<N, E>` (use `directed()`/`undirected()`) |
| Create Graph | `Graph::new()` | `Graph::directed()` / `Graph::undirected()` |
| Add Node | `graph.add_node(data)` | `graph.add_node(data)` (same) |
| Add Edge | `graph.add_edge(a, b, weight)` | `graph.add_edge(a, b, weight)` (same) |
| Neighbor Access | `graph.neighbors(node)` | `graph.neighbors(node)` (same) |
| Dijkstra | `dijkstra(graph, start, end, cost_fn)` | `dijkstra(graph, start, end)` (edge data is weight) |
| Parallel Algorithms | ❌ Not supported | ✅ Built-in (enable `parallel` feature) |
| Stable Indices | `StableGraph` | `Graph` (built-in generation mechanism) |

## Migration Steps

### 1. Update Dependencies

In `Cargo.toml`:

```toml
# Old dependency
[dependencies]
petgraph = "0.6"

# New dependency
[dependencies]
god-gragh = "0.4.0-beta"
```

### 2. Update Imports

```rust
// petgraph imports
use petgraph::graph::{Graph, NodeIndex};
use petgraph::algo::dijkstra;
use petgraph::visit::IntoNodeReferences;

// God-Graph imports
use god_graph::graph::Graph;
use god_graph::graph::traits::{GraphOps, GraphQuery};
use god_graph::algorithms::shortest_path::dijkstra;
```

### 3. Graph Creation

```rust
// petgraph
use petgraph::graph::Graph;
use petgraph::Directed;
let mut graph = Graph::<String, f64, Directed>::new();

// God-Graph
use god_graph::graph::Graph;
let mut graph = Graph::<String, f64>::directed();

// Undirected graph
// petgraph: Graph::<String, f64, Undirected>::new()
// God-Graph:
let mut graph = Graph::<String, f64>::undirected();
```

### 4. Node and Edge Operations

Most CRUD operations have consistent APIs:

```rust
// Add nodes (identical)
let a = graph.add_node("A".to_string()).unwrap();
let b = graph.add_node("B".to_string()).unwrap();

// Add edges (identical)
graph.add_edge(a, b, 1.0).unwrap();

// Remove nodes (identical)
graph.remove_node(a).unwrap();

// Neighbor access (identical)
for neighbor in graph.neighbors(a) {
    println!("{:?}", graph[neighbor]);
}

// Node/edge count
// petgraph: graph.node_count(), graph.edge_count()
// God-Graph: same
println!("Nodes: {}", graph.node_count());
println!("Edges: {}", graph.edge_count());
```

### 5. Shortest Path Algorithms

**Dijkstra's Algorithm**:

```rust
// petgraph
use petgraph::algo::dijkstra;
use petgraph::visit::IntoEdgeReferences;

// Need to manually provide edge weight closure
let distances = dijkstra(&graph, start, None, |edge| *edge.weight());

// God-Graph
use god_graph::algorithms::shortest_path::dijkstra;

// Directly use edge data as weight (more concise)
let distances = dijkstra(&graph, start, None);

// Get specific target
let (path, distance) = dijkstra(&graph, start, Some(end)).unwrap();
```

**A* Algorithm**:

```rust
// petgraph
use petgraph::algo::astar;
let astar_result = astar(&graph, start, |n| n == end, |e| *e.weight(), |_| 0.0);

// God-Graph
use god_graph::algorithms::shortest_path::astar;
let (path, distance) = astar(&graph, start, end, |_| 0.0).unwrap();
```

**Bellman-Ford**:

```rust
// petgraph
use petgraph::algo::bellman_ford;
let distances = bellman_ford(&graph, start).unwrap();

// God-Graph
use god_graph::algorithms::shortest_path::bellman_ford;
let distances = bellman_ford(&graph, start);
```

**Floyd-Warshall**:

```rust
// petgraph
use petgraph::algo::floyd_warshall;
let distances = floyd_warshall(&graph, |e| *e.weight()).unwrap();

// God-Graph
use god_graph::algorithms::shortest_path::floyd_warshall;
let distances = floyd_warshall(&graph);
```

### 6. Traversal Algorithms

```rust
// petgraph
use petgraph::visit::{Dfs, Bfs};
let mut dfs = Dfs::new(&graph, start);
while let Some(Ok(node)) = dfs.next(&graph) {
    // Process node
}

// God-Graph (more concise callback API)
use god_graph::algorithms::traversal::{dfs, bfs};

dfs(&graph, start, |node| {
    // Process node
    true // Continue traversal
});

bfs(&graph, start, |node| {
    // Process node
    true
});
```

### 7. Minimum Spanning Tree

```rust
// petgraph
use petgraph::algo::kruskal;
let mst: Vec<_> = kruskal(graph, |weight| weight).collect();

// God-Graph
use god_graph::algorithms::mst::{kruskal, prim};

// Kruskal
let mst = kruskal(&graph);

// Prim (God-Graph extra feature)
let mst = prim(&graph, start_node);
```

### 8. Centrality Algorithms

```rust
// petgraph - Need manual implementation or use extension libraries

// God-Graph - Built-in complete support
use god_graph::algorithms::centrality::{
    degree_centrality,
    betweenness_centrality,
    closeness_centrality,
    pagerank
};

let degree = degree_centrality(&graph);
let betweenness = betweenness_centrality(&graph);
let closeness = closeness_centrality(&graph);
let ranks = pagerank(&graph, 0.85, 20);
```

### 9. Community Detection

```rust
// petgraph - Need manual implementation

// God-Graph - Built-in support
use god_graph::algorithms::community::{
    connected_components,
    label_propagation,
    louvain
};

let components = connected_components(&graph);
let communities = label_propagation(&graph);
let communities = louvain(&graph);
```

### 10. Parallel Algorithms (God-Graph Exclusive)

```rust
// Enable parallel feature
// Cargo.toml: god-gragh = { version = "0.4.0-beta", features = ["parallel"] }

use god_graph::algorithms::parallel;

// Parallel BFS
let layers = parallel::bfs_parallel(&graph, start);

// Parallel PageRank (6-8x speedup on 8-core CPU)
let ranks = parallel::pagerank_parallel(&graph, 0.85, 20);

// Parallel Connected Components
let components = parallel::connected_components_parallel(&graph);
```

### 11. Random Graph Generation

```rust
// petgraph - Need manual implementation or use extension libraries

// God-Graph - Built-in complete support
use god_graph::generators::{
    erdos_renyi_graph,
    barabasi_albert_graph,
    watts_strogatz_graph,
    complete_graph,
    grid_graph,
    tree_graph
};

// Erdős-Rényi G(n, p)
let graph = erdos_renyi_graph::<String>(100, 0.1, true, 42);

// Barabási-Albert Preferential Attachment Model
let graph = barabasi_albert_graph::<String>(100, 3);

// Watts-Strogatz Small-World Network
let graph = watts_strogatz_graph::<String>(100, 4, 0.1);
```

### 12. Graph Export

```rust
// petgraph - Need additional dependencies or manual implementation

// God-Graph - Built-in support
use god_graph::export::{to_dot, to_adjacency_list, to_edge_list};

// DOT/Graphviz format
let dot = to_dot(&graph);
std::fs::write("graph.dot", dot)?;

// Adjacency list
let adj_list = to_adjacency_list(&graph);

// Edge list
let edge_list = to_edge_list(&graph);
```

## Complete Migration Examples

### Example 1: Shortest Path

**petgraph Version**:

```rust
use petgraph::graph::{Graph, NodeIndex};
use petgraph::algo::dijkstra;
use petgraph::visit::IntoEdgeReferences;

fn main() {
    let mut graph = Graph::<&str, f64>::new();
    let a = graph.add_node("A");
    let b = graph.add_node("B");
    let c = graph.add_node("C");

    graph.add_edge(a, b, 1.0);
    graph.add_edge(b, c, 2.0);
    graph.add_edge(a, c, 3.0);

    let distances = dijkstra(&graph, a, None, |edge| *edge.weight());
    println!("Distances: {:?}", distances);
}
```

**God-Graph Version**:

```rust
use god_graph::graph::Graph;
use god_graph::graph::traits::{GraphOps, GraphQuery};
use god_graph::algorithms::shortest_path::dijkstra;

fn main() {
    let mut graph = Graph::<&str, f64>::directed();
    let a = graph.add_node("A").unwrap();
    let b = graph.add_node("B").unwrap();
    let c = graph.add_node("C").unwrap();

    graph.add_edge(a, b, 1.0).unwrap();
    graph.add_edge(b, c, 2.0).unwrap();
    graph.add_edge(a, c, 3.0).unwrap();

    let distances = dijkstra(&graph, a, None);
    println!("Distances: {:?}", distances);
}
```

### Example 2: Community Detection

**petgraph Version** (requires manual implementation):

```rust
use petgraph::graph::Graph;
// Need to implement label propagation algorithm manually...

fn find_communities(graph: &Graph<&str, f64>) -> Vec<Vec<NodeIndex>> {
    // Manual implementation...
    vec![]
}
```

**God-Graph Version**:

```rust
use god_graph::graph::Graph;
use god_graph::algorithms::community::label_propagation;

fn main() {
    let mut graph = Graph::<&str, f64>::directed();
    // ... add nodes and edges

    let communities = label_propagation(&graph);
    println!("Communities: {:?}", communities);
}
```

## Performance Comparison

PageRank on 100K nodes, 500K edges graph (20 iterations):

| Library | Time | Speedup |
|---------|------|---------|
| God-Graph (Parallel) | 0.5s | 7.2x |
| God-Graph (Serial) | 3.6s | 1.0x |
| petgraph | 4.2s | 0.86x |

**Key Performance Advantages**:
1. **Hybrid CSR Memory Layout**: Better cache locality
2. **Arena Allocator**: Reduces memory fragmentation
3. **Parallel Algorithms**: Fully utilizes multi-core CPU
4. **64-byte Alignment**: Avoids false sharing

## FAQ

### Q: Is God-Graph fully compatible with petgraph's API?

A: Not fully compatible, but core APIs (node/edge operations, neighbor access) remain consistent. Main differences:
- Graph creation method (`directed()`/`undirected()` vs `new()`)
- Algorithm function signatures differ slightly (no need to manually pass edge weight closure)

### Q: Is the migration cost high?

A: For basic use cases, only modify import statements and graph creation code. For complex algorithm calls, adjust function signatures.

### Q: Can I use petgraph and God-Graph simultaneously?

A: Yes, but not recommended. The graph data structures are incompatible; recommend full migration to God-Graph.

### Q: Is the performance improvement significant?

A: Depends on use case:
- Serial algorithms: 1.2-1.5x (benefits from CSR layout)
- Parallel algorithms: 6-8x (8-core CPU)
- Large graphs (>10K nodes): More significant improvement

## Next Steps

- Read [README.md](../README.md) for complete features
- Check [API Documentation](https://docs.rs/god-gragh)
- Run benchmarks: `cargo bench`
- Report issues: [GitHub Issues](https://github.com/silverenternal/god-gragh/issues)
