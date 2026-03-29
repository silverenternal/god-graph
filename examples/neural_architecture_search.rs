//! Neural Architecture Search (NAS) Example
//!
//! This example demonstrates how to use DifferentiableGraph for gradient-based
//! neural architecture search. The goal is to automatically discover optimal
//! network topologies for specific tasks.
//!
//! ## Key Concepts
//!
//! 1. **Architecture as Graph**: Network topology is represented as a graph
//!    where nodes are operations and edges are data flow connections.
//!
//! 2. **Continuous Relaxation**: Edge probabilities represent the importance
//!    of each connection, enabling gradient-based optimization.
//!
//! 3. **Bi-Level Optimization**:
//!    - Inner loop: Optimize network weights (W)
//!    - Outer loop: Optimize architecture (A) based on validation loss
//!
//! 4. **DARTS-style Search**: Differentiable ARchiTecture Search approach
//!    using continuous relaxation and gradient descent.
//!
//! ## Search Space
//!
//! This example implements a cell-based search space similar to DARTS:
//! - **Normal Cell**: Preserves feature map size
//! - **Reduction Cell**: Reduces spatial dimensions
//! - **Operations**: Conv3x3, Conv5x5, MaxPool, AvgPool, Skip, etc.

#[cfg(feature = "tensor")]
mod neural_architecture_search {
    use god_gragh::tensor::differentiable::{
        DifferentiableGraph, GradientConfig, GraphTransformer, ThresholdEditPolicy,
    };
    use std::collections::HashMap;

    /// Operation types in the search space
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum OperationType {
        Conv3x3,
        Conv5x5,
        DilatedConv3x3,
        MaxPool3x3,
        AvgPool3x3,
        SkipConnect,
        Zero, // Represents pruned connection
    }

    impl OperationType {
        pub fn name(&self) -> &'static str {
            match self {
                OperationType::Conv3x3 => "Conv3x3",
                OperationType::Conv5x5 => "Conv5x5",
                OperationType::DilatedConv3x3 => "DilatedConv3x3",
                OperationType::MaxPool3x3 => "MaxPool3x3",
                OperationType::AvgPool3x3 => "AvgPool3x3",
                OperationType::SkipConnect => "SkipConnect",
                OperationType::Zero => "Zero",
            }
        }

        pub fn latency_cost(&self) -> f64 {
            // Relative latency cost (normalized)
            match self {
                OperationType::Conv3x3 => 1.0,
                OperationType::Conv5x5 => 2.5,
                OperationType::DilatedConv3x3 => 1.5,
                OperationType::MaxPool3x3 => 0.3,
                OperationType::AvgPool3x3 => 0.3,
                OperationType::SkipConnect => 0.0,
                OperationType::Zero => 0.0,
            }
        }

        pub fn param_count(&self, channels: usize) -> usize {
            match self {
                OperationType::Conv3x3 => channels * channels * 3 * 3,
                OperationType::Conv5x5 => channels * channels * 5 * 5,
                OperationType::DilatedConv3x3 => channels * channels * 3 * 3,
                OperationType::MaxPool3x3 | OperationType::AvgPool3x3 => 0,
                OperationType::SkipConnect => 0,
                OperationType::Zero => 0,
            }
        }
    }

    /// Example 1: Basic NAS setup with DARTS-style search space
    pub fn basic_nas_setup() {
        println!("=== Example 1: Basic NAS Setup (DARTS-style) ===\n");

        // Create a cell-based search space
        // Each cell has 4 nodes (input, intermediate x2, output)
        let num_nodes = 4;
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(num_nodes);

        // Initialize all possible connections with equal probability
        // This represents a "super-net" containing all candidate architectures
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                // Each edge represents a mixture of operations
                // Initial probability = uniform distribution
                graph.add_learnable_edge(i, j, 0.5);
            }
        }

        println!("DARTS-style cell created:");
        println!("  Nodes: {} (input, intermediate x2, output)", num_nodes);
        println!("  Possible edges: {}", graph.num_edges());
        println!("  Initial edge probability: 0.5 (uniform)");
        println!();

        // Display the search space
        println!("Search space (candidate architectures):");
        let prob_matrix = graph.get_probability_matrix();
        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.01 {
                    println!("  Edge {}→{}: P = {:.4}", i, j, prob);
                }
            }
        }

        println!();
        println!("Each edge contains a mixture of operations:");
        let operations = [
            OperationType::Conv3x3,
            OperationType::Conv5x5,
            OperationType::MaxPool3x3,
            OperationType::SkipConnect,
        ];
        for op in &operations {
            println!(
                "  - {}: cost = {:.1}M params (for 64 channels)",
                op.name(),
                op.param_count(64) as f64 / 1e6
            );
        }
    }

    /// Example 2: Gradient-based architecture optimization
    pub fn gradient_based_architecture_search() {
        println!("\n=== Example 2: Gradient-Based Architecture Search ===\n");

        let num_nodes = 5;
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(num_nodes);

        // Initialize super-net with all possible connections
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                graph.add_learnable_edge(i, j, 0.5);
            }
        }

        println!("Initial super-net architecture:");
        print_architecture(&graph);

        // Simulate architecture gradient computation
        // In real NAS, this comes from validation loss gradient
        let mut architecture_gradients = HashMap::new();

        // Simulate gradients based on a target architecture:
        // Optimal: 0→1→2→3→4 (chain) with skip 0→2
        let target_edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)];

        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                let is_target = target_edges.contains(&(i, j));
                // Negative gradient for important edges (keep them)
                // Positive gradient for redundant edges (prune them)
                let grad = if is_target { -0.5 } else { 0.3 };
                architecture_gradients.insert((i, j), grad);
            }
        }

        println!("\nArchitecture gradients (from validation loss):");
        for ((src, dst), &grad) in &architecture_gradients {
            let recommendation = if grad < -0.3 {
                "KEEP"
            } else if grad > 0.2 {
                "PRUNE"
            } else {
                "UNCERTAIN"
            };
            println!(
                "  {}→{}: ∂L_val/∂α = {:.3} [{}]",
                src, dst, grad, recommendation
            );
        }

        // Compute structure gradients and update
        let structure_gradients = graph.compute_structure_gradients(&architecture_gradients);
        graph.update_structure(&structure_gradients);

        println!("\nAfter one optimization step:");
        print_architecture(&graph);

        // Multiple optimization steps
        println!("\nRunning 10 optimization steps...");
        for _ in 0..10 {
            let mut grads = HashMap::new();
            for i in 0..num_nodes {
                for j in (i + 1)..num_nodes {
                    let is_target = target_edges.contains(&(i, j));
                    let grad = if is_target { -0.3 } else { 0.2 };
                    grads.insert((i, j), grad);
                }
            }
            let grads = graph.compute_structure_gradients(&grads);
            graph.update_structure(&grads);
        }

        println!("\nAfter 10 optimization steps:");
        print_architecture(&graph);

        // Extract final architecture
        println!("\nExtracted architecture (threshold P > 0.5):");
        let prob_matrix = graph.get_probability_matrix();
        let mut kept_edges = Vec::new();
        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.5 {
                    println!("  {}→{}: P = {:.4} ✓", i, j, prob);
                    kept_edges.push((i, j));
                } else if prob > 0.1 {
                    println!("  {}→{}: P = {:.4} (pruned)", i, j, prob);
                }
            }
        }

        println!(
            "\nFinal architecture has {} edges (from {} possible)",
            kept_edges.len(),
            num_nodes * (num_nodes - 1) / 2
        );
    }

    /// Example 3: Multi-objective NAS (accuracy + latency)
    pub fn multi_objective_nas() {
        println!("\n=== Example 3: Multi-Objective NAS (Accuracy + Latency) ===\n");

        let num_nodes = 5;

        // Configure with sparsity to encourage efficient architectures
        let config = GradientConfig::new(1.0, true, 0.1, 0.01).with_sparsity(0.05); // Encourage fewer connections

        let mut graph = DifferentiableGraph::<Vec<f64>>::with_config(num_nodes, config);

        // Initialize with operation-specific probabilities
        // Higher probability for efficient operations initially
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                // Bias towards efficient operations
                let distance = (j - i) as f64;
                let initial_prob = 0.5 / distance; // Prefer shorter connections
                graph.add_learnable_edge(i, j, initial_prob);
            }
        }

        println!("Initial architecture (biased towards efficiency):");
        print_architecture_with_cost(&graph);

        // Multi-objective optimization
        // Loss = L_accuracy + λ * L_latency
        let latency_weight = 0.1;

        println!(
            "\nRunning multi-objective optimization (λ = {:.2}):",
            latency_weight
        );

        for step in 0..15 {
            let mut accuracy_gradients = HashMap::new();

            // Simulate accuracy gradients (same as before)
            for i in 0..num_nodes {
                for j in (i + 1)..num_nodes {
                    // Simulate: longer connections are less important for accuracy
                    let distance = (j - i) as f64;
                    let grad = if distance <= 2.0 { -0.3 } else { 0.2 };
                    accuracy_gradients.insert((i, j), grad);
                }
            }

            // Add latency gradients
            // Latency gradient is always positive (want to reduce connections)
            // Stronger for expensive operations
            for ((i, j), grad) in accuracy_gradients.iter_mut() {
                let distance = (j - i) as f64;
                let latency_grad = latency_weight * distance * 0.1;
                *grad += latency_grad;
            }

            let structure_gradients = graph.compute_structure_gradients(&accuracy_gradients);
            graph.update_structure(&structure_gradients);

            if step % 5 == 0 {
                println!("\nStep {}:", step);
                print_architecture_with_cost(&graph);
            }
        }

        println!("\nFinal efficient architecture:");
        print_architecture_with_cost(&graph);

        // Compute total cost
        let total_cost = compute_architecture_cost(&graph);
        println!("\nTotal latency cost: {:.2}", total_cost);
    }

    /// Example 4: Architecture transfer across tasks
    pub fn architecture_transfer() {
        println!("\n=== Example 4: Architecture Transfer Across Tasks ===\n");

        // Search architecture on Task A (e.g., ImageNet classification)
        println!("Phase 1: Search on Task A (source task)...");
        let mut graph_a = DifferentiableGraph::<Vec<f64>>::new(5);

        for i in 0..5 {
            for j in (i + 1)..5 {
                graph_a.add_learnable_edge(i, j, 0.5);
            }
        }

        // Optimize for Task A
        for _ in 0..10 {
            let mut grads = HashMap::new();
            for i in 0..5 {
                for j in (i + 1)..5 {
                    // Task A prefers dense connections
                    let grad = -0.2;
                    grads.insert((i, j), grad);
                }
            }
            let structure_gradients = graph_a.compute_structure_gradients(&grads);
            graph_a.update_structure(&structure_gradients);
        }

        println!("Task A architecture:");
        print_architecture(&graph_a);

        // Transfer to Task B (e.g., COCO detection)
        println!("\nPhase 2: Transfer to Task B (target task)...");
        println!("  Strategy: Initialize with Task A architecture, fine-tune");

        let mut graph_b = DifferentiableGraph::<Vec<f64>>::new(5);

        // Copy architecture from Task A
        let prob_matrix_a = graph_a.get_probability_matrix();
        for i in 0..5 {
            for j in (i + 1)..5 {
                let transferred_prob = prob_matrix_a[i][j];
                graph_b.add_learnable_edge(i, j, transferred_prob);
            }
        }

        println!("Transferred architecture (before fine-tuning):");
        print_architecture(&graph_b);

        // Fine-tune for Task B
        for _ in 0..5 {
            let mut grads = HashMap::new();
            for i in 0..5 {
                for j in (i + 1)..5 {
                    // Task B has different preferences
                    let is_important = (j - i) <= 2 || (i == 0 && j == 4); // Skip connection
                    let grad = if is_important { -0.3 } else { 0.1 };
                    grads.insert((i, j), grad);
                }
            }
            let structure_gradients = graph_b.compute_structure_gradients(&grads);
            graph_b.update_structure(&structure_gradients);
        }

        println!("\nTask B architecture (after fine-tuning):");
        print_architecture(&graph_b);

        println!("\nKey insight: Architecture transfer provides better initialization");
        println!("than random, reducing search time on target task.");
    }

    /// Example 5: Progressive search space expansion
    pub fn progressive_search_space() {
        println!("\n=== Example 5: Progressive Search Space Expansion ===\n");

        // Start with small search space
        let initial_nodes = 4;
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(initial_nodes);

        for i in 0..initial_nodes {
            for j in (i + 1)..initial_nodes {
                graph.add_learnable_edge(i, j, 0.5);
            }
        }

        println!("Stage 1: Small search space ({} nodes)", initial_nodes);
        print_architecture(&graph);

        // Optimize small space
        for _ in 0..5 {
            let mut grads = HashMap::new();
            for i in 0..initial_nodes {
                for j in (i + 1)..initial_nodes {
                    grads.insert((i, j), -0.2);
                }
            }
            let structure_gradients = graph.compute_structure_gradients(&grads);
            graph.update_structure(&structure_gradients);
        }

        println!("\nAfter optimization:");
        print_architecture(&graph);

        // Expand search space
        println!("\nStage 2: Expanding search space to 6 nodes...");

        // Create new larger graph
        let new_nodes = 6;
        let mut new_graph = DifferentiableGraph::<Vec<f64>>::new(new_nodes);

        // Transfer learned architecture
        let prob_matrix = graph.get_probability_matrix();
        for i in 0..initial_nodes {
            for j in (i + 1)..initial_nodes {
                new_graph.add_learnable_edge(i, j, prob_matrix[i][j]);
            }
        }

        // Initialize new edges with moderate probability
        for i in 0..new_nodes {
            for j in (i + 1)..new_nodes {
                if i >= initial_nodes || j >= initial_nodes {
                    new_graph.add_learnable_edge(i, j, 0.3);
                }
            }
        }

        println!("Expanded architecture:");
        print_architecture(&new_graph);

        // Optimize expanded space
        for _ in 0..5 {
            let mut grads = HashMap::new();
            for i in 0..new_nodes {
                for j in (i + 1)..new_nodes {
                    let grad = if i < initial_nodes && j < initial_nodes {
                        -0.2 // Keep optimized structure
                    } else {
                        -0.1 // Learn new connections
                    };
                    grads.insert((i, j), grad);
                }
            }
            let structure_gradients = new_graph.compute_structure_gradients(&grads);
            new_graph.update_structure(&structure_gradients);
        }

        println!("\nAfter optimization:");
        print_architecture(&new_graph);

        println!("\nBenefit: Progressive expansion finds good architectures faster");
        println!("than searching the full space from scratch.");
    }

    /// Example 6: Architecture analysis and visualization
    pub fn architecture_analysis() {
        println!("\n=== Example 6: Architecture Analysis ===\n");

        let num_nodes = 6;
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(num_nodes);

        // Create a realistic searched architecture
        let edges = vec![
            (0, 1, 0.9),
            (0, 2, 0.7),
            (1, 2, 0.3),
            (1, 3, 0.8),
            (2, 3, 0.2),
            (2, 4, 0.85),
            (3, 4, 0.4),
            (3, 5, 0.75),
            (4, 5, 0.9),
            (0, 5, 0.1), // Long skip
        ];

        for (i, j, p) in edges {
            graph.add_learnable_edge(i, j, p);
        }

        println!("Searched architecture:");
        print_architecture(&graph);

        // Analyze architecture properties
        println!("\nArchitecture Analysis:");

        // 1. Depth (longest path)
        let depth = compute_architecture_depth(&graph);
        println!("  Effective depth: {} layers", depth);

        // 2. Skip connections
        let skip_connections = count_skip_connections(&graph);
        println!("  Skip connections: {}", skip_connections);

        // 3. Sparsity
        let sparsity = compute_sparsity(&graph);
        println!("  Sparsity: {:.1}%", sparsity * 100.0);

        // 4. FLOPs estimate
        let flops_estimate = estimate_flops(&graph, 64);
        println!("  Estimated FLOPs: {:.2}M", flops_estimate as f64 / 1e6);

        // 5. Parameter count
        let param_count = estimate_parameters(&graph, 64);
        println!("  Estimated parameters: {:.2}M", param_count as f64 / 1e6);

        // Architecture diagram
        println!("\nArchitecture Diagram:");
        print_architecture_diagram(&graph);
    }

    // Helper functions

    fn print_architecture<T: Clone + Default>(graph: &DifferentiableGraph<T>) {
        let prob_matrix = graph.get_probability_matrix();
        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.01 {
                    let status = if prob > 0.7 {
                        "STRONG"
                    } else if prob > 0.4 {
                        "MODERATE"
                    } else {
                        "WEAK"
                    };
                    println!("  {}→{}: P = {:.4} [{}]", i, j, prob, status);
                }
            }
        }
    }

    fn print_architecture_with_cost<T: Clone + Default>(graph: &DifferentiableGraph<T>) {
        let prob_matrix = graph.get_probability_matrix();
        let mut total_cost = 0.0;

        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.01 {
                    let distance = (j - i) as f64;
                    let cost = distance * prob;
                    total_cost += cost;
                    println!("  {}→{}: P = {:.4}, cost = {:.3}", i, j, prob, cost);
                }
            }
        }
        println!("  Total cost: {:.3}", total_cost);
    }

    fn compute_architecture_cost<T: Clone + Default>(graph: &DifferentiableGraph<T>) -> f64 {
        let prob_matrix = graph.get_probability_matrix();
        let mut total_cost = 0.0;

        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.01 {
                    let distance = (j - i) as f64;
                    total_cost += distance * prob;
                }
            }
        }

        total_cost
    }

    fn compute_architecture_depth<T: Clone + Default>(graph: &DifferentiableGraph<T>) -> usize {
        // Simple BFS to find longest path
        let num_nodes = graph.num_nodes();
        let prob_matrix = graph.get_probability_matrix();

        let mut max_depth = 0;
        for start in 0..num_nodes {
            let mut visited = vec![false; num_nodes];
            let mut queue = vec![(start, 1)];
            visited[start] = true;

            while let Some((node, depth)) = queue.pop() {
                max_depth = max_depth.max(depth);

                for next in 0..num_nodes {
                    if !visited[next] && prob_matrix[node].get(next).map_or(false, |&p| p > 0.5) {
                        visited[next] = true;
                        queue.push((next, depth + 1));
                    }
                }
            }
        }

        max_depth
    }

    fn count_skip_connections<T: Clone + Default>(graph: &DifferentiableGraph<T>) -> usize {
        let prob_matrix = graph.get_probability_matrix();
        let mut count = 0;

        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.5 && j > i + 1 {
                    count += 1;
                }
            }
        }

        count
    }

    fn compute_sparsity<T: Clone + Default>(graph: &DifferentiableGraph<T>) -> f64 {
        let prob_matrix = graph.get_probability_matrix();
        let total_possible = graph.num_nodes() * (graph.num_nodes() - 1) / 2;
        let active_edges = prob_matrix
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&p| p > 0.5)
            .count();

        1.0 - (active_edges as f64 / total_possible as f64)
    }

    fn estimate_flops<T: Clone + Default>(
        graph: &DifferentiableGraph<T>,
        channels: usize,
    ) -> usize {
        let prob_matrix = graph.get_probability_matrix();
        let mut flops = 0;

        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.5 {
                    // Estimate Conv3x3 FLOPs
                    let distance = j - i;
                    let op_flops = channels * channels * 3 * 3 * distance;
                    flops += op_flops;
                }
            }
        }

        flops
    }

    fn estimate_parameters<T: Clone + Default>(
        graph: &DifferentiableGraph<T>,
        channels: usize,
    ) -> usize {
        let prob_matrix = graph.get_probability_matrix();
        let mut params = 0;

        for (_i, row) in prob_matrix.iter().enumerate() {
            for (_j, &prob) in row.iter().enumerate() {
                if prob > 0.5 {
                    // Estimate Conv3x3 parameters
                    let op_params = channels * channels * 3 * 3;
                    params += op_params;
                }
            }
        }

        params
    }

    fn print_architecture_diagram<T: Clone + Default>(graph: &DifferentiableGraph<T>) {
        let num_nodes = graph.num_nodes();
        let prob_matrix = graph.get_probability_matrix();

        // Print nodes
        print!("  ");
        for i in 0..num_nodes {
            print!("[{}]", i);
            if i < num_nodes - 1 {
                print!("───");
            }
        }
        println!();

        // Print connections
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                let prob = prob_matrix[i][j];
                if prob > 0.5 {
                    let style = if prob > 0.8 { "═══" } else { "---" };
                    println!("  {}{}{}", style, style, style);
                    println!("  ↑   {}→{}", i, j);
                }
            }
        }
    }

    pub fn run_all_examples() {
        basic_nas_setup();
        gradient_based_architecture_search();
        multi_objective_nas();
        architecture_transfer();
        progressive_search_space();
        architecture_analysis();
    }
}

fn main() {
    #[cfg(feature = "tensor")]
    neural_architecture_search::run_all_examples();

    #[cfg(not(feature = "tensor"))]
    println!(
        "Please enable tensor feature: cargo run --example neural_architecture_search --features tensor"
    );
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "tensor")]
    #[test]
    fn test_nas_compiles() {
        use god_gragh::tensor::differentiable::DifferentiableGraph;

        let _graph = DifferentiableGraph::<Vec<f64>>::new(4);
    }
}
