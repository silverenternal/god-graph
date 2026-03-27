# Differentiable Graph Structure Transformation

## Overview

This module implements **gradient computation for graph structure transformation operations**, enabling gradient descent optimization of graph structures. This is cutting-edge technology in graph neural networks, graph generation, and graph optimization.

## Core Concepts

### 1. Continuous Relaxation

Traditional graph structures are discrete: edges either exist (1) or don't exist (0). To support gradient computation, we use **continuous relaxation**:

```
A_soft = σ(logits / τ)

Where:
- logits: Edge log-odds (learnable parameters)
- τ: Temperature parameter (controls discreteness)
- σ: Sigmoid function
```

As τ → 0, soft relaxation approaches discrete values.

### 2. Straight-Through Estimator (STE)

For scenarios requiring discrete output, use STE:
- **Forward pass**: Hard threshold (0 or 1)
- **Backward pass**: Soft gradient (through sigmoid)

```rust
// Discretization
exists = probability > 0.5

// Gradient propagation
∂L/∂logits = ∂L/∂A · A·(1-A)/τ
```

### 3. Gumbel-Softmax Sampling

For differentiable discrete sampling:

```
y_i = exp((log(π_i) + g_i) / τ) / Σ_j exp((log(π_j) + g_j) / τ)

Where:
- π: Category probabilities
- g_i ~ Gumbel(0,1): Noise
- τ: Temperature
```

## Usage

### Basic Example

```rust
use god_gragh::tensor::differentiable::{
    DifferentiableGraph, GradientConfig
};

// Create differentiable graph
let mut graph = DifferentiableGraph::<Vec<f64>>::new(4);

// Add learnable edges (initial probability indicates edge existence likelihood)
graph.add_learnable_edge(0, 1, 0.5);
graph.add_learnable_edge(1, 2, 0.8);
graph.add_learnable_edge(2, 3, 0.3);

// Discretize (get current structure)
graph.discretize();

// Check if edge exists
let edge_01_exists = graph.get_edge_exists(0, 1).unwrap();
```

### Gradient Computation and Optimization

```rust
use std::collections::HashMap;

// Assume gradients from downstream tasks (e.g., GNN classification)
// These gradients indicate: if an edge exists, will loss increase/decrease
let mut loss_gradients = HashMap::new();
loss_gradients.insert((0, 1), 0.5);   // Positive gradient: encourage removal
loss_gradients.insert((1, 2), -0.8);  // Negative gradient: encourage retention

// Compute structure gradients
let structure_gradients = graph.compute_structure_gradients(&loss_gradients);

// Update structure based on gradients
graph.update_structure(&structure_gradients);
```

### Complete Optimization Loop

```rust
// Configure optimizer
let config = GradientConfig::new(
    1.0,    // Initial temperature
    true,   // Use STE
    0.05,   // Edge learning rate
    0.01,   // Node learning rate
)
.with_sparsity(0.001)      // L1 sparse regularization
.with_smoothness(0.0001);  // Smooth regularization

let mut graph = DifferentiableGraph::with_config(5, config);

// Initialize edges
graph.add_learnable_edge(0, 1, 0.5);
graph.add_learnable_edge(1, 2, 0.5);
// ...

// Optimization loop
for step in 0..100 {
    // 1. Get gradients from downstream tasks
    let loss_gradients = compute_loss_gradients(&graph);

    // 2. One-step optimization (discretize → compute gradients → update → annealing)
    graph.optimization_step(loss_gradients);

    // 3. Periodic checking
    if step % 10 == 0 {
        println!("Step {}: T={:.4}", step, graph.temperature());
    }
}
```

### Using Edit Policies

```rust
use god_gragh::tensor::differentiable::{
    GraphTransformer, ThresholdEditPolicy
};

// Define edit policy
let policy = Box::new(ThresholdEditPolicy {
    add_threshold: 0.1,      // Add edge when gradient > 0.1
    remove_threshold: -0.1,  // Remove edge when gradient < -0.1
    min_prob: 0.01,
    max_prob: 0.99,
});

let mut transformer = GraphTransformer::new(policy);

// Record gradients
transformer.record_gradients(&gradients);

// Execute structure transformation
let edits = transformer.transform(&mut graph);

// View edit history
for edit in &edits {
    println!("Edit operation: {:?}, Gradient: {:.4}", edit.operation, edit.gradient);
}
```

### Gumbel-Softmax Sampling

```rust
use god_gragh::tensor::differentiable::GumbelSoftmaxSampler;

let sampler = GumbelSoftmaxSampler::new(1.0);
let logits = vec![0.5, 1.0, -0.5, 2.0];

// Soft sampling (differentiable, for training)
let soft = sampler.sample_soft(&logits);

// Hard sampling (non-differentiable, for inference)
let hard = sampler.sample_hard(&logits);

// STE sampling (hard forward, soft backward)
let (hard_ste, soft_ste) = sampler.sample_ste(&logits);
```

## Configuration Options

### GradientConfig

```rust
pub struct GradientConfig {
    pub temperature: f64,           // Temperature parameter (default 1.0)
    pub use_ste: bool,              // Whether to use STE (default true)
    pub edge_learning_rate: f64,    // Edge learning rate (default 0.01)
    pub node_learning_rate: f64,    // Node learning rate (default 0.001)
    pub sparsity_weight: f64,       // L1 sparse weight (default 0.0)
    pub smoothness_weight: f64,     // Smooth weight (default 0.0)
}
```

### Regularization

**Sparse Regularization (L1)**:
```rust
let config = GradientConfig::default().with_sparsity(0.01);
```
Encourages edge probabilities toward 0, producing sparser graph structures.

**Smooth Regularization**:
```rust
let config = GradientConfig::default().with_smoothness(0.001);
```
Encourages connected edges to have similar probabilities.

### Temperature Annealing

```rust
let mut graph = DifferentiableGraph::with_config(5, config)
    .with_temperature_annealing(100);  // 100-step annealing

// Automatic annealing per optimization step
for _ in 0..100 {
    graph.optimization_step(gradients);
    // Temperature exponentially decays from 1.0 to ~0.1
}
```

## Mathematical Principles

### Gradient Computation

For edge (i, j), gradient of loss L with respect to logits:

```
∂L/∂logits_ij = ∂L/∂A_ij · ∂A_ij/∂logits_ij
              = ∂L/∂A_ij · A_ij·(1-A_ij)/τ
```

Where:
- `∂L/∂A_ij`: Gradient from downstream tasks
- `A_ij·(1-A_ij)/τ`: Sigmoid derivative

### Sparse Regularization Gradient

```
L_sparse = λ · ||logits||_1
∂L_sparse/∂logits = λ · sign(logits)
```

### Smooth Regularization Gradient

```
L_smooth = μ · Σ_{(i,j),(i,k)∈E} (A_ij - A_ik)²
∂L_smooth/∂A_ij = 2μ · Σ_k (A_ij - A_ik)
```

## Application Scenarios

### 1. Graph Structure Learning

Learn optimal graph structure from data:
```rust
// Initial graph may be incomplete or noisy
let mut graph = create_initial_graph(data);

// Learn structure by optimizing task loss
for epoch in 0..num_epochs {
    // Forward: GNN trains on current structure
    let predictions = gnn.forward(&graph, features);
    let loss = compute_loss(predictions, labels);

    // Backward: Compute structure gradients
    let structure_gradients = compute_structure_gradients(loss);

    // Update graph structure
    graph.optimization_step(structure_gradients);
}
```

### 2. Graph Optimization

Optimize graph to meet specific objectives:
```rust
// Objective: Learn graph structure beneficial for classification
fn compute_loss_gradients(graph: &DifferentiableGraph) -> HashMap<(usize, usize), f64> {
    let mut gradients = HashMap::new();

    for ((src, dst), edge) in graph.get_learnable_edges() {
        // Simulated gradient: if edge connects same-class nodes, give negative gradient (encourage)
        // If connects different-class nodes, give positive gradient (suppress)
        let same_class = check_same_class(*src, *dst);
        let grad = if same_class { -0.5 } else { 0.5 };
        gradients.insert((*src, *dst), grad);
    }

    gradients
}
```

### 3. Graph Generation

Generate graphs conforming to specific distributions:
```rust
// Learn edge probability distribution
let mut graph = DifferentiableGraph::new(num_nodes);

// Learn via adversarial training or maximum likelihood
for step in 0..num_steps {
    // Sample current structure
    graph.discretize();

    // Compute difference between generated and target distributions
    let gradients = compute_distribution_gradients(&graph);

    // Update probability parameters
    graph.optimization_step(gradients);
}
```

## Running Examples

```bash
# Run complete example
cargo run --example differentiable_graph --features "tensor,tensor-sparse,rand"

# Run tests
cargo test --features "tensor,tensor-sparse,rand" --lib tensor::differentiable
```

## References

- **IDGL**: Iterative Deep Graph Learning (NeurIPS 2020)
- **GDC**: Graph Diffusion Convolution (NeurIPS 2019)
- **ProGNN**: Graph Structure Learning for Robust GNNs (KDD 2020)
- **Gumbel-Softmax**: Categorical Reparameterization with Gumbel-Softmax (ICLR 2017)

## Notes

1. **Temperature Selection**: Higher temperature (τ>1) makes distribution smoother, lower temperature (τ<0.5) makes distribution more discrete
2. **Learning Rate Tuning**: Edge learning rate is typically one order of magnitude larger than node learning rate
3. **Regularization Balance**: Sparse and smooth regularization weights need adjustment based on specific tasks
4. **Gradient Clipping**: For large-scale graphs, consider gradient clipping to prevent numerical instability
