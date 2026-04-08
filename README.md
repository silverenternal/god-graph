# God-Graph

[![Crates.io](https://img.shields.io/crates/v/god-graph.svg)](https://crates.io/crates/god-graph)
[![Documentation](https://docs.rs/god-graph/badge.svg)](https://docs.rs/god-graph)
[![License](https://img.shields.io/crates/l/god-graph.svg)](https://github.com/silverenternal/god-graph?tab=License-1-ov-file#readme)
[![Build Status](https://github.com/silverenternal/god-graph/workflows/CI/badge.svg)](https://github.com/silverenternal/god-graph/actions)
[![Coverage Status](https://codecov.io/gh/silverenternal/god-graph/branch/main/graph/badge.svg)](https://codecov.io/gh/silverenternal/god-graph)
[![Tests](https://img.shields.io/badge/tests-512%20passing-green)](https://github.com/silverenternal/god-graph/actions)
[![Clippy](https://img.shields.io/badge/clippy-0%20warnings-blue)](https://github.com/silverenternal/god-graph/actions)

> **Transform LLMs from Black Boxes into Editable White Boxes**
>
> **God-Graph** is a graph-based LLM white-box analysis toolbox featuring:
> - **VGI Architecture** (Virtual Graph Interface) — unified graph backend interface like Linux VFS
> - **DifferentiableGraph** — differentiable graph structures for gradient-based neural architecture search
> - **CAD-LLM Paradigm** — mechanical CAD design philosophy applied to LLM topology debugging
> - **ModelSwitch** — bidirectional Safetensors ↔ GodGraph conversion with L2 verification (< 1e-5 loss)

---

## 🎯 Core Positioning

**What God-Graph is NOT:**
- ❌ LLM inference engine (can't beat `llama.cpp`)
- ❌ GNN training framework (can't beat DGL/PyG)
- ❌ General graph library (`petgraph` is more mature)

**What God-Graph IS:**
- ✅ **LLM White-Box Analyzer** — inspect and modify model topology structure
- ✅ **Differentiable Graph Engine** — optimize neural architectures via gradient descent on graph structure
- ✅ **Topological Defect Detector** — find gradient blocking, isolated nodes, missing residual connections
- ✅ **Mathematical Optimizer** — Lie group orthogonalization, tensor ring compression
- ✅ **VGI Architecture** — unified graph backend interface (single-machine, distributed, GPU pluggable)

**One-Sentence Summary:** God-Graph applies CAD software design philosophy to LLMs — checking for "surface cracks" (isolated nodes), "non-manifold geometry" (gradient blocking), "dimensional constraints" (attention head balance), and pioneered **DifferentiableGraph** for gradient-guided architecture search.

---

## 🏗️ VGI Architecture: Linux VFS for Graph Computing

**VGI (Virtual Graph Interface)** is God-Graph's core abstraction layer, similar to Linux VFS (Virtual File System). It provides unified graph operations so algorithm code doesn't need to care about underlying storage details.

### Architecture Layers

```
┌─────────────────────────────────────────┐
│  Application (Your Code)                │
├─────────────────────────────────────────┤
│  Plugin System (GraphAlgorithm)         │
├─────────────────────────────────────────┤
│  VGI (VirtualGraph Trait) ← Core        │
├─────────────────────────────────────────┤
│  Backend (SingleMachine/Parallel/...)   │
└─────────────────────────────────────────┘
```

### Why VGI?

| Feature | Description | Example |
|---------|-------------|---------|
| **Pluggable Backend** | Algorithm code doesn't care about storage | Same algorithm runs on single-machine/parallel backends |
| **Capability Discovery** | Runtime query of backend capabilities | `graph.has_capability(Capability::Parallel)` |
| **Plugin Ecosystem** | Third-party algorithm development | `registry.register_algorithm("pagerank", plugin)` |
| **Backward Compatible** | Existing `Graph<T,E>` integrates seamlessly | `impl VirtualGraph for Graph<T,E>` |

### VGI Quick Example

```rust
use god_graph::vgi::{VirtualGraph, Capability};
use god_graph::graph::Graph;

// 1. Generic algorithm function (works with any backend)
fn average_degree<G>(graph: &G) -> f64
where
    G: VirtualGraph,
{
    let total = graph.nodes()
        .map(|n| graph.out_degree(n.index()).unwrap_or(0))
        .sum::<usize>();
    total as f64 / graph.node_count() as f64
}

// 2. Usage example
let mut graph = Graph::<String, f64>::directed();
let a = graph.add_node("A".to_string())?;
let b = graph.add_node("B".to_string())?;
let _ = graph.add_edge(
    a,
    b,
    1.0
)?;
```

### Core Capability Enum

```rust
pub enum Capability {
    Parallel,              // Parallel execution support
    Distributed,           // Distributed execution (planned)
    IncrementalUpdate,     // Incremental update support
    DynamicMode,           // Dynamic mode support
    WeightedEdges,         // Weighted edge support
    DifferentiableStructure, // Structure gradient computation (DifferentiableGraph)
    LieGroupOrthogonalization, // Lie group orthogonalization
    TensorRingCompression, // Tensor ring compression
    // ... more
}
```

---

## 📚 Documentation

**Complete Documentation**: [docs/README.md](docs/README.md)

### Quick Links

| Document | Description |
|----------|-------------|
| [**Quick Start**](docs/user-guide/getting-started.md) | 5-minute God-Graph introduction |
| [**DifferentiableGraph Tutorial**](docs/user-guide/differentiable-graph.md) | Complete differentiable graph guide |
| [**VGI Architecture Guide**](docs/VGI_GUIDE.md) | Virtual Graph Interface design and usage |
| [**Design Philosophy**](docs/internals/cad-design.md) | Why CAD-LLM paradigm shift is needed |
| [**Architecture Guide**](docs/internals/architecture.md) | Module responsibilities and workflows |
| [**Performance Report**](docs/reports/performance.md) | Parallel algorithms and SIMD benchmarks |
| [**Implementation Status**](docs/reports/implementation-status.md) | Feature completion and roadmap |
| [**TinyLlama Validation**](docs/reports/validation.md) | Real model end-to-end validation |

---

## ⚡ DifferentiableGraph: 5-Minute Quick Start

**DifferentiableGraph is God-Graph's core innovation** — transforming graph structure from "static container" to "differentiable computation itself", enabling gradient descent optimization of neural network architectures.

### Core Application Scenarios

1. **Dynamic Attention Pruning** — gradient-guided removal of weak attention edges (30-50% reduction)
2. **Topological Defect Detection** — automatic discovery of isolated nodes, gradient blocking, missing residual connections
3. **Neural Architecture Search** — let models learn optimal residual connections and attention patterns
4. **Weight Editing** — Lie group orthogonalization ensures numerical stability for precise weight modifications

### 5-Minute Example

```rust
use god_graph::tensor::differentiable::{
    DifferentiableGraph, GradientConfig, ThresholdEditPolicy
};

// 1. Build differentiable graph from standard Transformer
let mut graph = build_mini_transformer();
let config = GradientConfig::default().with_sparsity(0.1);
let mut diff_graph = DifferentiableGraph::from_graph(graph, config);

// 2. Define objective (attention entropy + sparsity regularization)
let loss_fn = |g: &DifferentiableGraph| {
    g.entropy_loss() + 0.1 * g.sparsity_loss()
};

// 3. Gradient descent on structure
for step in 0..100 {
    let loss = loss_fn(&diff_graph);
    let grads = diff_graph.compute_structure_gradients(loss);
    diff_graph.update_structure(&grads, 0.01);

    if step % 10 == 0 {
        println!("Step {}: loss={:.4}", step, loss);
    }
}

// 4. Export pruned graph
let policy = ThresholdEditPolicy::new(0.5);
let pruned_graph = diff_graph.discretize(&policy);
println!("Pruned {} weak attention edges", pruned_graph.num_pruned_edges());
```

### Complete Examples

| Example | Description | Command |
|---------|-------------|---------|
| [Differentiable Attention Pruning](examples/differentiable_graph.rs) | Gradient-guided edge pruning | `cargo run --example differentiable_graph --features tensor` |
| [Topological Defect Detection](examples/cad_llm_validate_1b.rs) | Detect model topology issues | `cargo run --example cad_llm_validate_1b --features transformer` |
| [Lie Group Orthogonalization](examples/cad_llm_orthogonalize.rs) | Weight orthogonalization stability | `cargo run --example cad_llm_orthogonalize --features transformer` |
| [Tensor Ring Compression](examples/cad_llm_tensor_ring.rs) | Model compression workflow | `cargo run --example cad_llm_tensor_ring --features transformer` |

See [DifferentiableGraph Complete Tutorial](docs/user-guide/differentiable-graph.md) for full guide.

---

## 🚀 Quick Start

### Installation

```toml
[dependencies]
god-graph = "0.6.0-alpha"
```

### Basic Usage: Graph Data Structure & Algorithms

```rust
use god_graph::graph::Graph;
use god_graph::algorithms::traversal::{bfs, dfs};

// Create a graph
let mut graph = Graph::<String, f64>::directed();
let a = graph.add_node("A".to_string())?;
let b = graph.add_node("B".to_string())?;
let _ = graph.add_edge(a, b, 1.0)?;

// BFS traversal
bfs(&graph, a, |node, _depth| {
    println!("Visit: {}", node.data());
    true
});
```

### Advanced Usage: LLM Topology Optimization

```rust
use god_graph::transformer::optimization::{
    ModelSwitch, CadStyleEditor, TensorRingCompressor
};

// 1. Load model from Safetensors
let mut graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// 2. Detect topological defects
let mut editor = CadStyleEditor::new(&mut graph);
let defects = editor.detect_defects()?;
println!("Found {} defects", defects.len());

// 3. Tensor ring compression
let compressor = TensorRingCompressor::default();
let report = compressor.compress_graph(&graph)?;
println!("Compression ratio: {:.2}x", report.compression_ratio);

// 4. Export optimized model to Safetensors
ModelSwitch::save_to_safetensors(&graph, "optimized.safetensors")?;
```

### ModelSwitch Bidirectional Conversion

**ModelSwitch** provides lossless bidirectional conversion between HuggingFace Safetensors and GodGraph:

```rust
use god_graph::transformer::optimization::ModelSwitch;

// Load: Safetensors → GodGraph
let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// Validate topology
let topology_report = ModelSwitch::validate_topology(&graph)?;
println!("Topology valid: {}", topology_report.is_valid);

// Verify weights (compare weight differences between two graphs)
let weight_diff = ModelSwitch::verify_weights(&original_graph, &modified_graph)?;
println!("Max L2 difference: {:.6e}", weight_diff.max_l2_diff);

// Export: GodGraph → Safetensors
ModelSwitch::save_to_safetensors(&graph, "optimized.safetensors")?;
```

**Features:**
- ✅ F32/F64/F16 data type support
- ✅ Weight precision verification (L2 norm comparison)
- ✅ Topology integrity check
- ✅ Operator inference (automatically infers operator type from weight name: Attention, MLP, Norm, etc.)

See [ModelSwitch Example](examples/cad_llm_switch.rs) for complete workflow.

---

## 🔬 Core Features

### 1. ModelSwitch Bidirectional Conversion ⭐ Core Feature

**ModelSwitch** implements lossless bidirectional conversion between HuggingFace Safetensors and GodGraph, forming the workflow foundation for LLM white-box analysis.

```rust
use god_graph::transformer::optimization::ModelSwitch;

// 1. Load: Safetensors → GodGraph
let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;

// 2. Validate topology integrity
let topology_report = ModelSwitch::validate_topology(&graph)?;
println!("Topology valid: {}", topology_report.is_valid);
println!("Connected components: {}", topology_report.connected_components);
println!("Is DAG: {}", topology_report.is_dag);

// 3. Verify weight precision (compare weight differences)
let weight_diff = ModelSwitch::verify_weights(&original_graph, &modified_graph)?;
println!("Max L2 difference: {:.6e}", weight_diff.max_l2_diff);
println!("Average L2 difference: {:.6e}", weight_diff.avg_l2_diff);

// 4. Export: GodGraph → Safetensors
ModelSwitch::save_to_safetensors(&graph, "optimized.safetensors")?;
```

**Key Features:**
- **Bidirectional Conversion:** Safetensors ↔ GodGraph lossless conversion
- **Data Type Support:** F32, F64, F16 automatic conversion
- **Topology Validation:** Check connectivity, cycles, isolated nodes
- **Weight Verification:** L2 norm comparison, precision loss < 1e-5
- **Operator Inference:** Auto-infer operator type from weight name (Attention, MLP, Norm, etc.)

**Run Example:**
```bash
cargo run --example cad_llm_switch --features safetensors
```

See [ModelSwitch Example](examples/cad_llm_switch.rs) for complete workflow.

---

### 2. DifferentiableGraph (Original Innovation) ⭐ Core Innovation

**This is God-Graph's original contribution** — transforming graph structure from "static container" to "differentiable computation itself".

```rust
use god_graph::tensor::differentiable::{DifferentiableGraph, GradientConfig};

// 1. Build differentiable graph from standard Transformer
let mut graph = build_transformer();
let config = GradientConfig::default().with_sparsity(0.1);
let mut diff_graph = DifferentiableGraph::from_graph(graph, config);

// 2. Gradient descent on structure
for step in 0..100 {
    let loss = diff_graph.entropy_loss() + 0.1 * diff_graph.sparsity_loss();
    let grads = diff_graph.compute_structure_gradients(loss);
    diff_graph.update_structure(&grads, 0.01);
}

// 3. Export pruned graph
let pruned = diff_graph.discretize(&ThresholdEditPolicy::new(0.5));
println!("Pruning ratio: {:.2}%", pruned.pruned_ratio() * 100.0);
```

**Core Techniques:**
- **Continuous Relaxation:** Convert discrete edge existence to continuous probabilities (0 to 1)
- **Straight-Through Estimator (STE):** Discrete-continuous bidirectional conversion with gradient backpropagation
- **Gumbel-Softmax:** Differentiable sampling supporting gradient backpropagation
- **Lie Group Orthogonalization:** Ensure numerical stability of weight matrices

**Application Scenarios:**
- Dynamic attention pruning (30-50% redundant edge reduction)
- Neural architecture search (auto-discover optimal residual connections)
- Topological defect detection (isolated nodes, gradient blocking)

See [DifferentiableGraph Tutorial](docs/differentiable_graph.md) for complete guide.

---

### 3. Lie Group Orthogonalization

Use Lie group theory to guarantee orthogonality of weight matrices, improving numerical stability.

```rust
use god_graph::tensor::decomposition::{lie_exponential, is_orthogonal};

// so(n) Lie algebra → SO(n) Lie group
let algebra = DenseTensor::from_vec(
    vec![0.0, -0.1, 0.1, 0.0],
    vec![2, 2],
);

let rotation = lie_exponential(&algebra)?;
assert!(is_orthogonal(&rotation, 1e-5));
```

**Mathematical Principle:** Exponential map `exp: so(n) → SO(n)` implemented via Padé approximation + scaling-squaring algorithm.

---

### 4. Tensor Ring Compression

Represent high-dimensional tensors as rings of 3D core tensors, reducing parameter count.

```rust
use god_graph::transformer::optimization::TensorRingCompressor;

let compressor = TensorRingCompressor::default();
let ring = compressor.decompose(&weight_tensor)?;

println!("Compression ratio: {:.2}x", ring.compression_ratio());
```

**Compression Ratio Formula:** `(m × n) / (r₀×m×r₁ + r₁×n×r₂)`

---

### 5. Topology Constraint Solving

Check LLM "geometric integrity" like CAD software.

```rust
use god_graph::transformer::optimization::{CadStyleEditor, TopologyConstraint};

let mut editor = CadStyleEditor::new(&mut graph);

// Detect defects
let defects = editor.detect_defects()?;

// Add constraints
editor.add_constraint(TopologyConstraint::ResidualConnection {
    from_layer: "attention".to_string(),
    to_layer: "output".to_string(),
})?;

// Solve constraints (automatic repair)
editor.solve_constraints()?;
```

**Defect Types:** Isolated nodes, disconnected components, gradient blocking, missing residual connections.

---

### 6. GraphTransformer Explicit Attention Analysis

**Positioning:** GraphTransformer is primarily for **visualizing attention topology**, **dynamic pruning of weak edges**, and **adding custom connections**. For high-performance inference, recommend converting to standard LlamaModel.

```rust
use god_graph::transformer::graph_transformer::GraphTransformer;

let mut transformer = GraphTransformer::new(12, 12, 768);
transformer.build_graph(&input_ids);

// Visualize attention topology
let dot = transformer.to_dot();
std::fs::write("attention_graph.dot", dot)?;

// Prune weak attention edges (threshold=0.01)
let pruned = transformer.prune_weak_edges(0.01);
println!("Pruned {} edges", pruned);

// Add custom long-range connections
transformer.add_skip_connection(layer_0, layer_11);
```

**Core Advantages:**
- Each attention edge individually accessible/modifiable (black-box inference engines can't do this)
- Dynamic topology editing (traditional static graphs can't do this)
- DOT/Graphviz export for visualization

---

## 📊 Performance Benchmarks

> **⚠️ Performance Disclaimer**: Benchmarks are run in controlled environments with specific workloads. Actual performance may vary based on:
> - CPU core count and clock speed (tested on AMD Ryzen 9 8945HS, 16 cores)
> - Graph structure (density, connectivity, sparsity pattern)
> - Memory bandwidth and cache size (61GB RAM in test environment)
> - Concurrent system load and OS scheduling
> - Rust compiler version and optimization flags
>
> Results shown represent **best-case scenarios** for demonstrating algorithmic improvements. Production workloads should run their own benchmarks for accurate performance expectations. See [docs/reports/performance.md](docs/reports/performance.md) for detailed methodology.

### Parallel Algorithm Speedup

**Test Environment**: Linux, AMD Ryzen 9 8945HS (16 cores @ 5.26 GHz), 61GB RAM  
**Compile Flags**: `-C opt-level=3 -C lto=thin -C codegen-units=1`  
**Features**: `parallel` (Rayon-based parallelization)

| Algorithm | Scale | Sequential | Parallel | Speedup | Notes |
|-----------|-------|------------|----------|---------|-------|
| PageRank | 1,000 nodes | 53.9ms | 668µs | **80.7×** | damping=0.85, iterations=20, avg_degree=5 |
| DFS | 50K nodes | 9.7ms | 1.3ms | **7.5×** | Sparse graph traversal |
| Connected Components | 2,000 nodes | - | 357.8µs | - | 4 components, ring-like structure |
| Degree Centrality | 5,000 nodes | - | 68µs | - | avg_degree=10 |

**Why PageRank shows 80.7× speedup:**
1. **Embarrassingly parallel**: Each node's rank update is independent
2. **Small graph fits in L3 cache**: 1000 nodes × avg_degree=5 = ~5K edges, minimal memory bandwidth pressure
3. **Rayon work-stealing**: Automatic load balancing across 16 cores
4. **Reversed adjacency list**: O(E) per iteration instead of O(V²)
5. **Fine-grained locking**: Mutex-protected per-node updates, no global contention

> Note: Speedup > core count is possible due to cache effects and measurement variance. Sequential baseline may include cold-start overhead not present in parallel version.

### SIMD Optimization

**Features**: `simd` (uses `wide::f64x4` for 4-way FP parallelism)

| Graph Scale | Sequential | Parallel | SIMD | Speedup vs Sequential |
|-------------|------------|----------|------|----------------------|
| 100 nodes | 2.1ms | 280µs | ~150µs | 14× |
| 1,000 nodes | 210ms | 2.8ms | ~1.5ms | 140× |

**CPU Feature Detection**: Runtime AVX-512 detection via `has_avx512()`. Falls back to `wide::f64x4` (SSE/AVX) if unavailable. See [System Requirements](#system-requirements) for AVX-512 configuration.

### Memory Pool Optimization

**Benchmark Source**: `benches/memory_pool_reduction.rs`  
**Test Pattern**: 50 iterations of acquire/drop cycle (simulates GNN/Transformer forward pass)

| Benchmark | Without Pool | With Pool | Pool Hit Rate | Allocation Reduction |
|-----------|--------------|-----------|---------------|---------------------|
| Iterative (128×128 tensors) | 850.84 µs | 127.76 µs | **98-100%** | **98-99.9%** |
| GNN Iteration (100×64 tensors) | - | 31.93 µs | **96-99%** | **96-99%** |
| MatMul Temporaries (64×64 tensors) | - | 42.15 µs | **95-98%** | **95-98%** |
| Small Tensors (16×16) | - | 6.89 µs | **98%+** | **98%+** |
| Large Tensors (512×512) | - | 17.36 µs | **95%+** | **95%+** |

**Key Findings:**
- **98-99.9% allocation reduction** for iterative workloads (50+ iterations of same-size tensors)
- **6.7× speedup** for iterative allocation patterns (850.84 µs → 127.76 µs)
- **Automatic recycling** via `PooledTensor` Drop trait
- **GradientCheckpoint** reduces backprop memory by **40-60%** (not shown in table)

**How memory pool works:**
```rust
// Without pool: 50 new allocations
for _ in 0..50 {
    let tensor = DenseTensor::zeros(vec![128, 128]); // 50 × malloc
}

// With pool: 1 allocation, 49 reuses
let mut pool = TensorPool::new(config);
for _ in 0..50 {
    let tensor = pool.acquire(vec![128, 128]); // 1st alloc, 49 reuses from pool
    drop(tensor); // Returns to pool, not system
}
// Allocation reduction = 49/50 = 98%
```

> Note: Pool effectiveness depends on allocation pattern. Best results for iterative algorithms (GNN, PageRank, Transformer inference) with repeated same-size allocations. One-off allocations see minimal benefit.

---

## 🏗️ Architecture Design

### CAD-LLM Paradigm Mapping

| CAD Concept | LLM Equivalent | GodGraph Implementation |
|-------------|----------------|------------------------|
| Surface crack check | Isolated attention head detection | `connected_components` |
| Non-manifold geometry check | Gradient blocking detection | `topological_sort + path_analysis` |
| Dimensional constraint | Attention head weight balance | `AttentionHeadBalance` constraint |
| Parallel constraint | Residual connection enforcement | `ResidualConnection` constraint |
| Assembly constraint | Module interface matching | `validate_assembly` |
| Part replacement | Module extraction/replacement | `extract_module` / `replace_module` |

See [Design Philosophy](docs/CAD_LLM_DESIGN_PHILOSOPHY.md) for details.

---

## 📦 Feature Flags

### Base Features

| Feature | Description |
|---------|-------------|
| `parallel` | Parallel algorithms (Rayon) |
| `simd` | SIMD vectorization (`wide::f64x4`) |
| `tensor` | Tensor core support (ndarray) |
| `tensor-sparse` | Sparse tensor formats (COO/CSR) |
| `tensor-gnn` | GNN layers (GCN/GAT/GraphSAGE) |

### LLM Optimization Features

| Feature | Description |
|---------|-------------|
| `transformer` | Transformer base architecture |
| `safetensors` | Safetensors model loading |
| `cad-llm` | CAD-LLM topology optimization (experimental) |

### Meta-Features (Recommended)

| Meta-Feature | Includes |
|--------------|----------|
| `tensor-full` | All tensor features |
| `tensor-inference` | GNN inference only |
| `llm` | Complete LLM support |

---

## 🔮 Roadmap

| Version | Status | Key Features |
|---------|--------|--------------|
| v0.4.3-beta | ✅ Released | Lie group orthogonalization, tensor ring compression, topology constraints |
| v0.5.0-alpha | ✅ Released | **DifferentiableGraph differentiable structure**, complete model loading, real model validation |
| **v0.6.0-alpha** | 🔥 **Current** | **VGI Architecture**, **HashMap→Vec Performance Optimizations**, Distributed Algorithms, Fault Tolerance |
| v0.7.0-beta | 📅 Planned | GPU backend completion, memory pool benchmarks, GraphTransformer execution engine |
| v1.0.0-rc | 📅 Planned | API stabilization, production-ready release |

### v0.6.0-alpha Key Features

- **VGI Architecture:** Complete Virtual Graph Interface with plugin ecosystem
- **Performance Optimizations:** HashMap→Vec replacements (2-3x DFS speedup, 1.5-2x community detection)
- **Distributed Algorithms:** DFS, Connected Components, Dijkstra, PageRank, BFS
- **Fault Tolerance:** RetryPolicy, CircuitBreaker, HealthChecker, CheckpointRecovery
- **508 Tests Passing:** Full test suite with all features enabled

### v0.5.0-alpha Key Features

- **DifferentiableGraph:** 1421 lines of core code enabling gradient-guided architecture search
- **Real Model Validation:** TinyLlama-1.1B end-to-end optimization workflow
- **Graph-level Orthogonalization:** In-place orthogonalization interface (zero-copy), error < 1e-8
- **Complete Examples:** 5 end-to-end DifferentiableGraph examples

See [Implementation Status](docs/reports/implementation-status.md) and [todo.json](todo.json) for details.

---

## 🎯 Target Users

### Ideal for God-Graph

✅ **LLM Researchers** — want to inspect and modify model topology
✅ **Model Compression Engineers** — want tensor ring/orthogonalization compression
✅ **QA Teams** — want to validate model integrity and numerical stability
✅ **Algorithm Explorers** — want to experiment with dynamic pruning, sparse attention, NAS
✅ **White-Box Analysis Needs** — want to understand LLM internal mechanisms

### NOT for God-Graph

❌ **Application Developers** — just want LLM inference (use `llama.cpp`)
❌ **Training Engineers** — want to train new models (use PyTorch/JAX)
❌ **GPU Acceleration Needs** — need CUDA inference (use `candle` or `vllm`)

---

## 🌟 God-Graph's Unique Advantages

### 1. Bucket Adjacency List + Generation Indexing

- **O(1) Incremental Updates:** Better than static CSR for dynamic graph editing scenarios
- **Prevents ABA Problem:** Reused indices after node deletion don't confuse (type safety petgraph lacks)
- **64-byte Alignment:** Prevents CPU cache false sharing, foundation for inference performance

### 2. DifferentiableGraph (Original Innovation)

- **Differentiable Graph Structure:** Converts discrete graph structures to continuous, differentiable form
- **Gradient-Guided Search:** Uses gradient descent to auto-discover optimal neural architectures
- **STE + Gumbel-Softmax:** Supports discrete-continuous bidirectional conversion with gradient backpropagation

### 3. GraphTransformer Explicit Attention

- **Per-Edge Access/Modification:** Black-box inference engines (llama.cpp) can't do this
- **Dynamic Topology Editing:** Traditional static graphs (petgraph) can't do this
- **Visualization Support:** Export to DOT/Graphviz format for intuitive attention pattern understanding

### 4. ModelSwitch Bidirectional Conversion Workflow

- **Safetensors ↔ GodGraph:** HuggingFace format bidirectional conversion
- **Weight Precision Verification:** L2 norm comparison, round-trip loss < 1e-5
- **Topology Integrity Check:** Automatic detection of isolated nodes, gradient blocking
- **Operator Type Inference:** Identifies Attention, MLP, Norm, etc. from weight names

### 5. Lie Group Orthogonalization + Tensor Ring Compression

- **Mathematical Guarantee:** Lie group theory ensures weight matrix orthogonality, numerical stability
- **Compression Ratio:** Tensor ring decomposition reduces parameters 2-10×
- **End-to-End Workflow:** Safetensors ↔ GodGraph ↔ Safetensors

---

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code passes `cargo clippy` and `cargo fmt`
- Add appropriate tests
- Update documentation

---

## 📄 License

Dual-licensed: MIT or Apache-2.0 (your choice)

---

## 🙏 Acknowledgments

- [petgraph](https://github.com/petgraph/petgraph) - Pioneer Rust graph algorithm library
- [ndarray](https://crates.io/crates/ndarray) - N-dimensional arrays
- [wide](https://crates.io/crates/wide) - SIMD math library
- [HuggingFace](https://huggingface.co/) - Safetensors format

---

**Contact**: silverenternal <3147264070@qq.com>  
**Project**: https://github.com/silverenternal/god-graph

---

## Quick Start (English)

### Installation

Add dependency to `Cargo.toml`:

```toml
[dependencies]
god-graph = "0.6.0-alpha"
```

### Basic Usage

```rust
use god_graph::graph::Graph;
use god_graph::graph::traits::{GraphOps, GraphQuery};

// Create a directed graph
let mut graph = Graph::<String, f64>::directed();

// Add nodes
let a = graph.add_node("A".to_string()).unwrap();
let b = graph.add_node("B".to_string()).unwrap();
let c = graph.add_node("C".to_string()).unwrap();

// Add edges
graph.add_edge(a, b, 1.0).unwrap();
graph.add_edge(b, c, 2.0).unwrap();
graph.add_edge(a, c, 3.0).unwrap();

// Query
println!("Nodes: {}", graph.node_count());
println!("Edges: {}", graph.edge_count());

// Iterate over neighbors
for neighbor in graph.neighbors(a) {
    println!("Neighbor: {}", neighbor.data());
}
```

### Using Graph Builder

```rust
use god_graph::graph::builders::GraphBuilder;

let graph = GraphBuilder::directed()
    .with_nodes(vec!["A", "B", "C", "D"])
    .with_edges(vec![
        (0, 1, 1.0),
        (0, 2, 2.0),
        (1, 3, 3.0),
        (2, 3, 4.0),
    ])
    .build()
    .unwrap();
```

## Algorithms

### Traversal Algorithms

```rust
use god_graph::algorithms::traversal::{dfs, bfs, topological_sort, tarjan_scc};

// Depth-First Search
dfs(&graph, start_node, |node| {
    println!("Visit: {}", node.data());
    true // Continue traversal
});

// Breadth-First Search
bfs(&graph, start_node, |node| {
    println!("Visit: {}", node.data());
    true
});

// Topological Sort (DAG only)
let order = topological_sort(&graph);

// Tarjan's Strongly Connected Components
let sccs = tarjan_scc(&graph);
```

### Shortest Path Algorithms

```rust
use god_graph::algorithms::shortest_path::{dijkstra, bellman_ford, floyd_warshall, astar};

// Dijkstra's Algorithm (non-negative weights)
let (path, distance) = dijkstra(&graph, start, Some(end)).unwrap();

// A* Search
let heuristic = |node: NodeIndex| -> f64 { /* Heuristic function */ 0.0 };
let (path, distance) = astar(&graph, start, end, heuristic).unwrap();

// Bellman-Ford (handles negative weights)
let distances = bellman_ford(&graph, start);

// Floyd-Warshall (all-pairs shortest paths)
let distances = floyd_warshall(&graph);
```

### Minimum Spanning Tree

```rust
use god_graph::algorithms::mst::{kruskal, prim};

// Kruskal's Algorithm
let mst = kruskal(&graph);

// Prim's Algorithm
let mst = prim(&graph, start_node);
```

### Centrality Algorithms

```rust
use god_graph::algorithms::centrality::{
    degree_centrality, betweenness_centrality, closeness_centrality, pagerank
};

// Degree Centrality
let centrality = degree_centrality(&graph);

// Betweenness Centrality
let centrality = betweenness_centrality(&graph);

// Closeness Centrality
let centrality = closeness_centrality(&graph);

// PageRank
let ranks = pagerank(&graph, 0.85, 20);
```

### Community Detection

```rust
use god_graph::algorithms::community::{connected_components, label_propagation};

// Connected Components
let components = connected_components(&graph);

// Label Propagation Algorithm
let communities = label_propagation(&graph);
```

### Flow Algorithms

```rust
use god_graph::algorithms::flow::{edmonds_karp, dinic, push_relabel};

// Edmonds-Karp Maximum Flow
let (flow, residual_graph) = edmonds_karp(&graph, source, sink);

// Dinic's Algorithm
let flow = dinic(&graph, source, sink);

// Push-Relabel Algorithm
let flow = push_relabel(&graph, source, sink);
```

## Parallel Algorithms

Enable `parallel` feature to use parallel algorithms:

```toml
[dependencies]
god-graph = { version = "0.6.0-alpha", features = ["parallel"] }
```

```rust
use god_graph::algorithms::parallel;

// Parallel BFS
let layers = parallel::bfs_parallel(&graph, start_node);

// Parallel PageRank
let ranks = parallel::pagerank_parallel(&graph, 0.85, 20);

// Parallel Connected Components
let components = parallel::connected_components_parallel(&graph);
```

### SIMD Optimization

Enable `simd` feature for SIMD vectorization (stable Rust support):

```toml
[dependencies]
god-graph = { version = "0.6.0-alpha", features = ["simd"] }
```

```rust
use god_graph::algorithms::parallel;

// SIMD-accelerated PageRank
#[cfg(feature = "simd")]
let ranks = parallel::par_pagerank_simd(&graph, 0.85, 20);

// SIMD-accelerated Degree Centrality
#[cfg(feature = "simd")]
let centrality = parallel::par_degree_centrality_simd(&graph);
```

**Implementation Details:** Uses `wide::f64x4` type for 4-way parallel floating-point operations, automatically leveraging CPU SIMD instruction sets (SSE/AVX/AVX-512).

## Tensor & GNN Support

Enable tensor features for Graph Neural Network workflows:

```toml
[dependencies]
god-graph = { version = "0.6.0-alpha", features = ["tensor", "tensor-gnn"] }
```

### Basic Tensor Operations

```rust
use god_graph::tensor::{DenseTensor, TensorBase, TensorOps};

// Create tensors
let a = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = DenseTensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

// Matrix multiplication
let c = a.matmul(&b);

// Transpose
let t = a.transpose(None);

// Normalize
let norm = a.normalize();
```

### Graph-Tensor Conversion

```rust
use god_graph::graph::Graph;
use god_graph::tensor::GraphTensorExt;

// Create a graph with vector node features
let mut graph = Graph::<Vec<f64>, f64>::directed();

let n0 = graph.add_node(vec![1.0, 0.0]).unwrap();
let n1 = graph.add_node(vec![0.0, 1.0]).unwrap();
let n2 = graph.add_node(vec![1.0, 1.0]).unwrap();

let _ = graph.add_edge(n0, n1, 1.0);
let _ = graph.add_edge(n1, n2, 1.0);
let _ = graph.add_edge(n2, n0, 1.0);

// Convert to tensor representation
let (features, adjacency) = graph.to_tensor_representation().unwrap();

assert_eq!(features.shape(), &[3, 2]);
assert_eq!(adjacency.num_nodes, 3);
```

### GNN Layers

> **Important**: God-Graph GNN modules are **inference-only** (forward pass only).
> For training workflows, integrate with external autograd libraries:
> - **[dfdx](https://crates.io/crates/dfdx)**: Deep learning framework with CUDA support
> - **[Candle](https://github.com/huggingface/candle)**: HuggingFace's lightweight tensor library
> - **[tch-rs](https://crates.io/crates/tch-rs)**: Rust bindings for PyTorch

#### Inference Example (Recommended Use Case)

```rust
use god_graph::tensor::gnn::{GCNConv, GATConv, GraphSAGE, MessagePassingLayer};

// Create GCN layer
let gcn = GCNConv::new(64, 64);

// Create GAT layer (multi-head attention)
let gat = GATConv::new(
    64,  // in_features
    64,  // out_features
    4,   // num_heads
);

// Create GraphSAGE layer
let graphsage = GraphSAGE::new(
    64,  // in_features
    32,  // out_features
    10,  // num_samples
);

// Forward pass (inference only)
let h1 = gcn.forward(&features, &adjacency);
let h2 = gat.forward(&h1, &edge_index);
let output = graphsage.forward(&h2, &edge_index);
```

#### Training Integration Example (with dfdx)

For complete GNN training, integrate with dfdx:

```rust
// Pseudo-code: Integrate god-graph GNN with dfdx autograd
use dfdx::prelude::*;
use god_graph::tensor::gnn::GCNConv;

// 1. Use god-graph for graph structure and forward pass
let gcn = GCNConv::new(64, 64);
let output = gcn.forward(&features, &adjacency);

// 2. Convert to dfdx tensor for autograd
// let dfdx_tensor = Tensor1D::from(output.data());

// 3. Define loss and optimizer (dfdx)
// let loss = cross_entropy_loss(&dfdx_tensor, &labels);
// let mut optimizer = Adam::new(model.parameters(), lr=0.001);

// 4. Training loop
// for epoch in 0..num_epochs {
//     optimizer.zero_grad();
//     let loss = forward_pass(&graph, &labels);
//     optimizer.backward(&loss);
//     optimizer.step();
// }
```

**See**: [examples/differentiable_graph.rs](examples/differentiable_graph.rs) for an example of differentiable graph structures and gradient-based optimization.

### Memory Pool Optimization

```rust
use god_graph::tensor::{TensorPool, PoolConfig};

// Create a tensor pool
let config = PoolConfig::new(16, 128).with_preallocate(true);
let mut pool = TensorPool::new(config);

// Acquire tensor from pool (automatically zeroed)
let tensor = pool.acquire(vec![100, 100]);

// Automatically returned to pool when dropped
drop(tensor);
```

**Benefits:**
- **Memory Reuse:** Reduces allocation overhead in iterative algorithms (PageRank, GNN training) by **80-90%**
- **Automatic Recycling:** `PooledTensor` automatically returns to pool on Drop
- **Gradient Checkpointing:** `GradientCheckpoint` reduces memory usage during backpropagation by **40-60%**

> **Note:** See [Performance Benchmarks](#-performance-benchmarks) for detailed memory pool benchmark results and methodology.

---

## Performance Optimizations Summary

**Test Environment**: Linux, AMD Ryzen 9 8945HS (16 cores @ 5.26 GHz), 61GB RAM
**Rust Version**: 1.85, 2021 edition
**Compile Flags**: `-C opt-level=3 -C lto=thin -C codegen-units=1`
**Test Suite**: 508 tests passing (11.5s runtime)
**Release Build**: ~17-20s (clean build with all features)

### HashMap → Vec Optimizations (v0.6.0-alpha)

| Optimization | Files Modified | Expected Speedup | Algorithm Impact |
|--------------|----------------|------------------|------------------|
| **DFS HashMap→Vec** | `distributed/algorithms/dfs.rs` | 2-3x | Eliminates ~10-50ns hash overhead per node visit |
| **Tarjan SCC Vec** | `distributed/algorithms/dfs.rs` | 2-3x | O(1) direct access for lowlinks/index arrays |
| **Community Detection** | `algorithms/community.rs` | 1.5-2x | Faster label propagation iterations |
| **Matrix Operations** | `utils/matrix.rs` | 1.2-1.5x | Adjacency/Laplacian matrix construction |
| **Matching Algorithm** | `algorithms/matching.rs` | 1.2-1.8x | Sorted Vec + dedup for edge deduplication |
| **GraphTransformer** | `transformer/graph_transformer/execution.rs` | 1.1-1.3x | Vec<bool> for visited tracking |
| **Constraint Validation** | `transformer/optimization/constraints.rs` | 1.1-1.2x | Vec<bool> for gradient flow validation |

**Optimization Pattern**: Replace `HashMap<usize, T>` or `HashMap<NodeIndex, T>` with `Vec<T>` for dense integer keys, using `usize::MAX` as sentinel for invalid/unvisited entries.

### Parallel & SIMD Optimizations

| Optimization | Implementation | Measured Speedup | Test Conditions |
|--------------|----------------|------------------|-----------------|
| **Parallel Algorithms** | Rayon-based parallelization | PageRank: 80.7×, DFS: 7.5× | 1K nodes, damping=0.85, 20 iterations |
| **SIMD Vectorization** | `wide::f64x4` for 4-way FP ops | 14-140× | 100-1K node graphs |
| **Memory Pool** | `TensorPool` with automatic recycling | 98-99.9% alloc reduction, 6.7× speedup | 50 iterations of 128×128 tensors |
| **Bucket Adjacency** | O(1) incremental updates | N/A (algorithmic improvement) | Better than CSR for dynamic edits |
| **64-byte Alignment** | Prevents CPU cache false sharing | N/A (foundational optimization) | Inference performance baseline |
| **AVX-512 Support** | Runtime CPU feature detection | 2× matmul and layer_norm | Requires AVX-512F, BW, CD, VL |
| **Flash Attention** | True single-pass algorithm | 60-90% memory reduction | Eliminates exp_scores allocation |
| **Register Blocking** | 4-row SPMV blocking | 1.3-1.8× sparse matmul | Sparse matrix-vector multiply |

> **Note**: Speedup values represent measured improvements over baseline implementations. Actual performance gains depend on workload characteristics, hardware configuration, and data patterns. See individual benchmark sources for detailed methodology:
> - Parallel algorithms: `benches/parallel.rs`
> - Memory pool: `benches/memory_pool_reduction.rs`
> - Transformer ops: `benches/transformer.rs`
> - Full report: [docs/reports/performance.md](docs/reports/performance.md)

---

## System Requirements

### Minimum Requirements

- **Rust Version**: 1.85 or later (2021 edition)
- **CPU**: x86_64 with SSE2 support (all modern x86_64 CPUs)
- **OS**: Linux, macOS, Windows
- **Memory**: Varies by workload (1GB+ recommended for large graphs)

### Optional Dependencies

| Feature | Requires | Purpose |
|---------|----------|---------|
| `tensor-gpu` | CUDA-capable NVIDIA GPU | GPU-accelerated tensor operations via dfdx |
| `tensor-blas` | OpenBLAS/MKL system library | BLAS-accelerated large matrix operations |
| `simd` | SSE2 (baseline), AVX/AVX-512 recommended | SIMD vectorization for numeric ops |
| `parallel` | Multi-core CPU | Parallel algorithm execution |

### Recommended Configuration

For **best performance** on modern hardware, enable these features:

```toml
[dependencies]
god-graph = {
    version = "0.6.0-alpha",
    features = ["parallel", "simd", "tensor-full", "transformer", "safetensors"]
}
```

### AVX-512 Acceleration (Optional)

If your CPU supports AVX-512 (e.g., AMD Ryzen 9 7940HS/8945HS, Intel Xeon Scalable), you can enable compile-time AVX-512 optimization:

**Option 1: Project-wide (`.cargo/config.toml`)**
```toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+avx512f,+avx512vl,+avx512bw,+avx512cd"]
```

**Option 2: Per-build (environment variable)**
```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512vl,+avx512bw,+avx512cd" cargo build --release
```

> **⚠️ Warning**: Binaries compiled with AVX-512 target features will **not run on CPUs without AVX-512 support**. The runtime detection (`has_avx512()`) allows fallback to SIMD paths, but compile-time features require the target CPU to support the instructions.

### Verify AVX-512 Support

Check if your CPU supports AVX-512:

```bash
# Linux
grep -o 'avx512f' /proc/cpuinfo

# Or use CPUID tools
cargo install raw-cpuid
raw-cpuid | grep -i avx512
```

If `avx512f` appears in output, your CPU supports AVX-512. God-Graph's runtime detection will automatically use AVX-512 paths when available (no compile-time flags required).

### CUDA Support (Optional)

For GPU-accelerated tensor operations:

1. Install CUDA Toolkit 11.0+
2. Enable `tensor-gpu` feature:
   ```toml
   [dependencies]
   god-graph = { version = "0.6.0-alpha", features = ["tensor-gpu"] }
   ```
3. Ensure `nvcc` is in your PATH

> **Note**: `tensor-gpu` feature requires dfdx crate and CUDA-capable NVIDIA GPU. See [dfdx documentation](https://docs.rs/dfdx) for detailed setup instructions.
