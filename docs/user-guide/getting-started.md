# Getting Started with God-Graph

**Version**: 0.6.0-alpha  
**Last Updated**: 2026-04-01

This guide will help you get started with God-Graph in 5 minutes.

---

## 📋 Prerequisites

- **Rust**: 1.85 or later
- **Cargo**: Package manager (comes with Rust)
- **Optional**: Python 3.8+ (for model downloading)

---

## 🚀 Installation

### Basic Installation

Add God-Graph to your `Cargo.toml`:

```toml
[dependencies]
god-graph = "0.6.0-alpha"
```

### Feature Selection

God-Graph uses feature flags to enable optional functionality:

```toml
[dependencies]
# Basic graph algorithms (default)
god-graph = "0.6.0-alpha"

# With parallel algorithms
god-graph = { version = "0.6.0-alpha", features = ["parallel"] }

# With tensor/GNN support
god-graph = { version = "0.6.0-alpha", features = ["tensor", "tensor-gnn"] }

# With Transformer/LLM support
god-graph = { version = "0.6.0-alpha", features = ["transformer", "safetensors"] }

# Full-featured (recommended for exploration)
god-graph = { version = "0.6.0-alpha", features = [
    "parallel",
    "tensor-full",
    "transformer",
    "safetensors",
] }
```

### Available Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| `parallel` | Parallel algorithms (Rayon) | Large-scale graph processing |
| `tensor` | Tensor core (ndarray backend) | GNN, ML workflows |
| `tensor-sparse` | Sparse tensor formats | Memory-efficient operations |
| `tensor-gnn` | GNN layers (GCN, GAT, GraphSAGE) | Graph neural networks |
| `tensor-pool` | Memory pool optimization | Iterative algorithms |
| `transformer` | Transformer architecture | LLM inference |
| `safetensors` | Safetensors model loading | HuggingFace models |
| `simd` | SIMD vectorization | CPU acceleration |
| `dot` | Graphviz export | Visualization |

---

## 🎯 Quick Examples

### Example 1: Basic Graph Operations

Create and manipulate a simple graph:

```rust
use god_graph::graph::Graph;
use god_graph::graph::traits::{GraphBase, GraphOps, GraphQuery};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a directed graph
    let mut graph = Graph::<String, f64>::directed();
    
    // Add nodes
    let a = graph.add_node("A".to_string())?;
    let b = graph.add_node("B".to_string())?;
    let c = graph.add_node("C".to_string())?;
    
    // Add edges
    graph.add_edge(a, b, 1.0)?;
    graph.add_edge(b, c, 2.0)?;
    graph.add_edge(a, c, 3.0)?;
    
    // Query the graph
    println!("Nodes: {}", graph.node_count());
    println!("Edges: {}", graph.edge_count());
    
    // Iterate over neighbors
    for neighbor in graph.neighbors(a) {
        println!("Neighbor of A: {}", graph.node_data(neighbor)?);
    }
    
    Ok(())
}
```

**Run**: `cargo run --example basic_graph`

---

### Example 2: Graph Algorithms

Run common graph algorithms:

```rust
use god_graph::graph::Graph;
use god_graph::algorithms::traversal::{bfs, dfs, topological_sort};
use god_graph::algorithms::shortest_path::dijkstra;
use god_graph::algorithms::centrality::pagerank;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sample graph
    let mut graph = Graph::<String, f64>::directed();
    let nodes: Vec<_> = (0..5)
        .map(|i| graph.add_node(format!("Node {}", i)))
        .collect::<Result<_, _>>()?;
    
    // Add edges
    graph.add_edge(nodes[0], nodes[1], 1.0)?;
    graph.add_edge(nodes[0], nodes[2], 2.0)?;
    graph.add_edge(nodes[1], nodes[3], 3.0)?;
    graph.add_edge(nodes[2], nodes[3], 4.0)?;
    graph.add_edge(nodes[3], nodes[4], 5.0)?;
    
    // BFS traversal
    println!("\nBFS from Node 0:");
    bfs(&graph, nodes[0], |node, depth| {
        println!("  Depth {}: {}", depth, node.data());
        true
    });
    
    // Topological sort
    println!("\nTopological order:");
    let order = topological_sort(&graph)?;
    for (i, node_idx) in order.iter().enumerate() {
        println!("  {}: {}", i, graph.node_data(*node_idx)?);
    }
    
    // PageRank
    println!("\nPageRank scores:");
    let ranks = pagerank(&graph, 0.85, 20);
    for (node_idx, rank) in ranks.iter().enumerate() {
        println!("  {}: {:.4}", graph.node_data(node_idx)?, rank);
    }
    
    Ok(())
}
```

**Run**: `cargo run --example graph_algorithms --features parallel`

---

### Example 3: Differentiable Graph (Core Innovation)

Optimize graph structure using gradient descent:

```rust
use god_graph::tensor::differentiable::{
    DifferentiableGraph, GradientConfig, ThresholdEditPolicy
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build a simple transformer-like graph
    let mut graph = build_mini_transformer();
    
    // 2. Convert to differentiable form
    let config = GradientConfig::default()
        .with_sparsity(0.1)  // Target 10% sparsity
        .with_learning_rate(0.01);
    
    let mut diff_graph = DifferentiableGraph::from_graph(graph, config);
    
    // 3. Gradient-based structure optimization
    for step in 0..100 {
        // Define loss: entropy + sparsity regularization
        let loss = diff_graph.entropy_loss() + 0.1 * diff_graph.sparsity_loss();
        
        // Compute gradients w.r.t. structure
        let grads = diff_graph.compute_structure_gradients(loss);
        
        // Update structure
        diff_graph.update_structure(&grads, 0.01);
        
        if step % 10 == 0 {
            println!("Step {}: loss = {:.4}", step, loss);
        }
    }
    
    // 4. Discretize (prune weak edges)
    let policy = ThresholdEditPolicy::new(0.5);
    let pruned_graph = diff_graph.discretize(&policy);
    
    println!("\nPruning complete!");
    println!("  Original edges: {}", pruned_graph.original_edge_count());
    println!("  Pruned edges: {}", pruned_graph.num_pruned_edges());
    println!("  Pruning ratio: {:.2}%", pruned_graph.pruned_ratio() * 100.0);
    
    Ok(())
}

fn build_mini_transformer() -> god_graph::graph::Graph<String, f64> {
    // Simplified transformer graph for demonstration
    let mut graph = god_graph::graph::Graph::directed();
    
    // Add layers
    let input = graph.add_node("input".to_string()).unwrap();
    let attention = graph.add_node("attention".to_string()).unwrap();
    let ffn = graph.add_node("ffn".to_string()).unwrap();
    let output = graph.add_node("output".to_string()).unwrap();
    
    // Add connections
    graph.add_edge(input, attention, 1.0).unwrap();
    graph.add_edge(attention, ffn, 1.0).unwrap();
    graph.add_edge(ffn, output, 1.0).unwrap();
    graph.add_edge(input, output, 0.3).unwrap(); // Skip connection
    
    graph
}
```

**Run**: `cargo run --example differentiable_graph --features tensor`

**Learn More**: See [DifferentiableGraph Tutorial](differentiable-graph.md)

---

### Example 4: LLM Model Loading (Safetensors)

Load and validate a real LLM model:

```rust
use god_graph::transformer::optimization::{
    ModelSwitch, CadStyleEditor, LieGroupOptimizer
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load model from Safetensors
    let mut graph = ModelSwitch::load_from_safetensors("model.safetensors")?;
    
    println!("Model loaded successfully!");
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}", graph.edge_count());
    
    // 2. Validate topology
    let topology_report = ModelSwitch::validate_topology(&graph)?;
    println!("\nTopology validation:");
    println!("  Valid: {}", topology_report.is_valid);
    println!("  Connected components: {}", topology_report.connected_components);
    println!("  Is DAG: {}", topology_report.is_dag);
    
    // 3. Detect topological defects
    let mut editor = CadStyleEditor::new(&mut graph);
    let defects = editor.detect_defects()?;
    
    if !defects.is_empty() {
        println!("\n⚠ Found {} defects:", defects.len());
        for defect in defects {
            println!("  - {}", defect.description());
        }
    } else {
        println!("\n✓ No topological defects detected");
    }
    
    // 4. Apply Lie group orthogonalization
    let config = god_graph::transformer::optimization::LieGroupConfig::default();
    let optimizer = LieGroupOptimizer::new(config);
    optimizer.orthogonalize_weights(&mut graph)?;
    
    println!("\n✓ Weights orthogonalized (SO(n) guarantee)");
    
    // 5. Export optimized model
    ModelSwitch::save_to_safetensors(&graph, "optimized.safetensors")?;
    println!("✓ Exported to optimized.safetensors");
    
    Ok(())
}
```

**Run**: `cargo run --example cad_llm_switch --features "safetensors tensor"`

**Download Model**: Use TinyLlama-1.1B for testing:
```bash
python scripts/download_tinyllama.py
```

---

### Example 5: Graph Neural Networks

Build and run a simple GNN:

```rust
use god_graph::tensor::{DenseTensor, GraphAdjacencyMatrix};
use god_graph::tensor::gnn::{GCNConv, MessagePassingLayer};
use god_graph::graph::Graph;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create a graph with node features
    let mut graph = Graph::<Vec<f64>, f64>::directed();
    
    // Add nodes with 4-dimensional features
    let n0 = graph.add_node(vec![1.0, 0.0, 0.0, 0.0])?;
    let n1 = graph.add_node(vec![0.0, 1.0, 0.0, 0.0])?;
    let n2 = graph.add_node(vec![0.0, 0.0, 1.0, 0.0])?;
    let n3 = graph.add_node(vec![0.0, 0.0, 0.0, 1.0])?;
    
    // Add edges (chain structure)
    graph.add_edge(n0, n1, 1.0)?;
    graph.add_edge(n1, n2, 1.0)?;
    graph.add_edge(n2, n3, 1.0)?;
    
    // 2. Convert to tensor representation
    let (features, adjacency) = graph.to_tensor_representation()?;
    println!("Feature shape: {:?}", features.shape());
    
    // 3. Create GCN layer
    let gcn = GCNConv::new(4, 8); // 4 input features, 8 output
    
    // 4. Forward pass
    let hidden = gcn.forward(&features, &adjacency);
    println!("Hidden state shape: {:?}", hidden.shape());
    
    // 5. Add another layer
    let gcn2 = GCNConv::new(8, 4);
    let output = gcn2.forward(&hidden, &adjacency);
    println!("Output shape: {:?}", output.shape());
    
    Ok(())
}
```

**Run**: `cargo run --example gnn_inference --features "tensor tensor-gnn"`

---

## 📚 Next Steps

### Choose Your Path

#### Path 1: Graph Algorithms
- **Next**: [Migration from Petgraph](migration-from-petgraph.md)
- **Learn**: BFS, DFS, shortest paths, centrality measures
- **Build**: Social network analysis, recommendation systems

#### Path 2: Differentiable Graph
- **Next**: [DifferentiableGraph Tutorial](differentiable-graph.md)
- **Learn**: Gradient-based structure optimization
- **Build**: Neural architecture search, dynamic pruning

#### Path 3: LLM Optimization
- **Next**: [Transformer Guide](transformer-guide.md)
- **Learn**: Model loading, topology validation, weight optimization
- **Build**: LLM white-box analysis, model compression

#### Path 4: Graph Neural Networks
- **Next**: [Tensor Core Documentation](../api-reference/tensor.md)
- **Learn**: GCN, GAT, GraphSAGE layers
- **Build**: Node classification, graph classification

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: Feature Not Enabled

**Error**: `cannot find type 'DenseTensor' in this scope`

**Solution**: Enable the `tensor` feature:
```toml
god-graph = { version = "0.6.0-alpha", features = ["tensor"] }
```

#### Issue 2: Parallel Algorithms Not Working

**Error**: `use of undeclared crate or module 'rayon'`

**Solution**: Enable the `parallel` feature:
```toml
god-graph = { version = "0.6.0-alpha", features = ["parallel"] }
```

#### Issue 3: Safetensors Loading Fails

**Error**: `InvalidFormat("Tensor model.layers.0.weight not found")`

**Solution**: Ensure the model file is valid Safetensors format and contains expected tensors. Use HuggingFace's `transformers` library to convert if needed.

#### Issue 4: Compilation Slow

**Solution**: Use release mode for benchmarks:
```bash
cargo run --release --example your_example
```

---

## 📖 Documentation Navigation

| Document | Description |
|----------|-------------|
| [DifferentiableGraph Tutorial](differentiable-graph.md) | Complete guide to differentiable graph structures |
| [VGI Architecture Guide](../VGI_GUIDE.md) | Virtual Graph Interface design and usage |
| [Transformer Guide](transformer-guide.md) | LLM model loading and optimization |
| [Migration from Petgraph](migration-from-petgraph.md) | API comparison and migration guide |
| [Performance Report](../reports/performance.md) | Benchmark data and optimization tips |
| [Implementation Status](../reports/implementation-status.md) | Feature completeness and roadmap |

---

## 🤝 Getting Help

- **API Documentation**: [docs.rs/god-graph](https://docs.rs/god-graph)
- **GitHub Issues**: [Report bugs](https://github.com/silverenternal/god-graph/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/silverenternal/god-graph/discussions)
- **Examples**: See `examples/` directory for complete code samples

---

## ✅ Verification Checklist

Before moving to advanced topics, ensure you can:

- [ ] Create a graph and add nodes/edges
- [ ] Run BFS/DFS traversal
- [ ] Execute at least one algorithm (PageRank, Dijkstra, etc.)
- [ ] Enable and use optional features
- [ ] Run an example successfully

**Congratulations!** You're ready to explore advanced God-Graph features! 🎉
