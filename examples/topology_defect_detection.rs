//! Topology Defect Detection Example
//!
//! This example demonstrates how to use DifferentiableGraph for detecting
//! and fixing topology defects in transformer models. Topology defects include:
//!
//! 1. **Missing Connections**: Critical paths that should exist but don't
//! 2. **Redundant Connections**: Edges that add no value to information flow
//! 3. **Bottleneck Nodes**: Nodes with high in/out degree imbalance
//! 4. **Isolated Subgraphs**: Disconnected components in attention graph
//! 5. **Cycle Defects**: Problematic cyclic dependencies
//!
//! ## Key Concepts
//!
//! - **Differentiable Topology**: Edge probabilities represent connection strength
//! - **Gradient-Based Detection**: Use gradients to identify problematic structures
//! - **Automated Repair**: Apply graph transformations to fix defects
//!
//! ## Real-Model Application
//!
//! This example simulates analysis of a TinyLlama-1.1B model's attention pattern,
//! identifying layers with suboptimal connectivity for pruning or enhancement.

#[cfg(feature = "tensor")]
mod topology_defect_detection {
    use god_gragh::tensor::differentiable::{DifferentiableGraph, GradientConfig};
    use std::collections::HashMap;

    /// Represents a detected topology defect
    #[derive(Debug, Clone)]
    pub struct TopologyDefect {
        pub defect_type: DefectType,
        pub location: String,
        pub severity: f64,
        pub suggestion: String,
    }

    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    pub enum DefectType {
        MissingConnection,
        RedundantConnection,
        BottleneckNode,
        IsolatedSubgraph,
        CycleDefect,
        WeakAttention,
    }

    /// Example 1: Basic topology defect detection
    pub fn basic_defect_detection() {
        println!("=== Example 1: Basic Topology Defect Detection ===\n");

        // Simulate a transformer attention graph with intentional defects
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(6);

        // Layer 0→1: Strong connection (good)
        graph.add_learnable_edge(0, 1, 0.95);

        // Layer 1→2: Weak connection (DEFECT: should be stronger)
        graph.add_learnable_edge(1, 2, 0.15);

        // Layer 2→3: Missing connection (DEFECT: should exist)
        // Intentionally not added

        // Layer 3→4: Redundant connection (low probability)
        graph.add_learnable_edge(3, 4, 0.05);

        // Layer 4→5: Strong connection (good)
        graph.add_learnable_edge(4, 5, 0.90);

        // Skip connection 0→5: Potentially redundant long-range
        graph.add_learnable_edge(0, 5, 0.10);

        println!("Analyzing transformer attention topology...\n");

        let defects = detect_topology_defects(&graph);

        println!("Detected {} topology defects:\n", defects.len());
        for (i, defect) in defects.iter().enumerate() {
            println!("Defect #{}: {:?}", i + 1, defect.defect_type);
            println!("  Location: {}", defect.location);
            println!("  Severity: {:.2}", defect.severity);
            println!("  Suggestion: {}", defect.suggestion);
            println!();
        }
    }

    /// Example 2: Gradient-based defect sensitivity analysis
    pub fn gradient_sensitivity_analysis() {
        println!("=== Example 2: Gradient-Based Defect Sensitivity ===\n");

        let mut graph = DifferentiableGraph::<Vec<f64>>::new(5);

        // Initialize attention pattern
        for i in 0..4 {
            graph.add_learnable_edge(i, i + 1, 0.5);
        }

        // Simulate gradients from a downstream task
        // High positive gradient: edge is harmful (redundant)
        // High negative gradient: edge is critical (missing would hurt)
        let mut loss_gradients = HashMap::new();

        // Edge 0→1: Critical (large negative gradient)
        loss_gradients.insert((0, 1), -0.8);

        // Edge 1→2: Redundant (large positive gradient)
        loss_gradients.insert((1, 2), 0.6);

        // Edge 2→3: Moderately important
        loss_gradients.insert((2, 3), -0.3);

        // Edge 3→4: Not important
        loss_gradients.insert((3, 4), 0.1);

        println!("Input loss gradients:");
        for ((src, dst), &grad) in &loss_gradients {
            let classification = classify_gradient(grad);
            println!(
                "  {}→{}: ∂L/∂A = {:.3} [{}]",
                src, dst, grad, classification
            );
        }

        // Compute structure gradients
        let structure_gradients = graph.compute_structure_gradients(&loss_gradients);

        println!("\nStructure sensitivity analysis:");
        for ((src, dst), &struct_grad) in &structure_gradients {
            let prob = graph.get_edge_probability(*src, *dst).unwrap();
            let sensitivity = struct_grad.abs();

            let defect_risk = if sensitivity > 0.5 && prob < 0.3 {
                "HIGH RISK: Critical edge with low probability"
            } else if sensitivity > 0.5 && prob > 0.7 {
                "GOOD: Critical edge with high probability"
            } else if sensitivity < 0.1 {
                "LOW IMPACT: Edge has minimal effect"
            } else {
                "MODERATE"
            };

            println!(
                "  {}→{}: sensitivity = {:.3}, P = {:.3} [{}]",
                src, dst, sensitivity, prob, defect_risk
            );
        }
    }

    /// Example 3: Bottleneck node detection
    pub fn bottleneck_node_detection() {
        println!("=== Example 3: Bottleneck Node Detection ===\n");

        let mut graph = DifferentiableGraph::<Vec<f64>>::new(7);

        // Create a bottleneck at node 3
        // Nodes 0,1,2 → Node 3 → Nodes 4,5,6

        // Incoming edges to bottleneck
        graph.add_learnable_edge(0, 3, 0.8);
        graph.add_learnable_edge(1, 3, 0.8);
        graph.add_learnable_edge(2, 3, 0.8);

        // Outgoing edges from bottleneck
        graph.add_learnable_edge(3, 4, 0.8);
        graph.add_learnable_edge(3, 5, 0.8);
        graph.add_learnable_edge(3, 6, 0.8);

        println!("Graph structure:");
        print_graph_state(&graph);

        let bottlenecks = detect_bottlenecks(&graph, 0.5);

        println!("\nDetected {} bottleneck nodes:\n", bottlenecks.len());
        for (node, info) in &bottlenecks {
            println!(
                "Node {}: in_degree={}, out_degree={}, imbalance={:.2}",
                node, info.0, info.1, info.2
            );
            println!("  Analysis: High fan-in/fan-out - potential information bottleneck");
            println!("  Suggestion: Consider adding skip connections around this node\n");
        }
    }

    /// Example 4: Missing connection detection via gradient analysis
    pub fn missing_connection_detection() {
        println!("=== Example 4: Missing Connection Detection ===\n");

        let num_nodes = 5;
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(num_nodes);

        // Create a chain with a missing link
        graph.add_learnable_edge(0, 1, 0.9);
        // Missing: 1→2
        graph.add_learnable_edge(2, 3, 0.9);
        graph.add_learnable_edge(3, 4, 0.9);

        println!("Initial graph (note: edge 1→2 is missing):");
        print_graph_state(&graph);

        // Simulate gradients that would indicate missing connection
        // If we could add edge 1→2, what would the gradient be?
        let potential_edges = vec![(1, 2), (0, 2), (1, 3), (0, 4)];

        println!("\nAnalyzing potential missing connections:");
        for (src, dst) in &potential_edges {
            // Simulate gradient for potential edge
            let simulated_gradient = if *src == 1 && *dst == 2 {
                -0.9 // Strong negative: adding this edge would greatly reduce loss
            } else {
                -0.1 // Weak negative: minor benefit
            };

            let recommendation = if simulated_gradient < -0.5 {
                "STRONG CANDIDATE: Add this connection"
            } else if simulated_gradient < -0.2 {
                "POSSIBLE: Consider adding"
            } else {
                "UNNECESSARY: No benefit"
            };

            println!(
                "  Potential edge {}→{}: simulated ∂L/∂A = {:.3} [{}]",
                src, dst, simulated_gradient, recommendation
            );
        }
    }

    /// Example 5: Redundant connection pruning
    pub fn redundant_connection_pruning() {
        println!("=== Example 5: Redundant Connection Pruning ===\n");

        let mut graph = DifferentiableGraph::<Vec<f64>>::new(6);

        // Create graph with redundant connections
        // Main path: 0→1→2→3→4→5
        for i in 0..5 {
            graph.add_learnable_edge(i, i + 1, 0.9);
        }

        // Redundant skip connections
        graph.add_learnable_edge(0, 2, 0.2);
        graph.add_learnable_edge(0, 3, 0.1);
        graph.add_learnable_edge(1, 3, 0.15);
        graph.add_learnable_edge(1, 4, 0.05);
        graph.add_learnable_edge(2, 4, 0.1);
        graph.add_learnable_edge(2, 5, 0.08);

        println!("Initial graph with redundant connections:");
        println!("  Total edges: {}", graph.num_edges());
        print_graph_state(&graph);

        // Configure with sparsity to encourage pruning
        let config = GradientConfig::new(1.0, true, 0.1, 0.01).with_sparsity(0.1);
        let mut sparse_graph = DifferentiableGraph::<Vec<f64>>::with_config(6, config);

        // Copy edges to sparse graph
        for i in 0..5 {
            sparse_graph.add_learnable_edge(i, i + 1, 0.9);
        }
        sparse_graph.add_learnable_edge(0, 2, 0.2);
        sparse_graph.add_learnable_edge(0, 3, 0.1);
        sparse_graph.add_learnable_edge(1, 3, 0.15);
        sparse_graph.add_learnable_edge(1, 4, 0.05);
        sparse_graph.add_learnable_edge(2, 4, 0.1);
        sparse_graph.add_learnable_edge(2, 5, 0.08);

        // Simulate optimization with sparsity
        let mut loss_gradients = HashMap::new();
        // Main path: keep (negative gradient)
        for i in 0..5 {
            loss_gradients.insert((i, i + 1), -0.3);
        }
        // Skip connections: prune (positive gradient)
        loss_gradients.insert((0, 2), 0.2);
        loss_gradients.insert((0, 3), 0.3);
        loss_gradients.insert((1, 3), 0.25);
        loss_gradients.insert((1, 4), 0.4);
        loss_gradients.insert((2, 4), 0.3);
        loss_gradients.insert((2, 5), 0.35);

        println!("\nOptimizing with sparsity regularization...");
        for _ in 0..15 {
            sparse_graph.optimization_step(loss_gradients.clone());
        }

        println!("\nAfter sparsity optimization:");
        println!("  Total edges: {}", sparse_graph.num_edges());

        let prob_matrix = sparse_graph.get_probability_matrix();
        let mut pruned_count = 0;
        let mut kept_count = 0;

        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.01 {
                    if prob < 0.3 {
                        pruned_count += 1;
                        println!("  {}→{}: P = {:.4} [PRUNED]", i, j, prob);
                    } else {
                        kept_count += 1;
                        println!("  {}→{}: P = {:.4} [KEPT]", i, j, prob);
                    }
                }
            }
        }

        println!(
            "\nSummary: {} edges pruned, {} edges kept",
            pruned_count, kept_count
        );
    }

    /// Example 6: Cycle defect detection
    pub fn cycle_defect_detection() {
        println!("=== Example 6: Cycle Defect Detection ===\n");

        let mut graph = DifferentiableGraph::<Vec<f64>>::new(5);

        // Create a cycle: 0→1→2→0
        graph.add_learnable_edge(0, 1, 0.8);
        graph.add_learnable_edge(1, 2, 0.8);
        graph.add_learnable_edge(2, 0, 0.3); // Back edge (potential defect)

        // Normal forward path: 0→3→4
        graph.add_learnable_edge(0, 3, 0.9);
        graph.add_learnable_edge(3, 4, 0.9);

        println!("Graph with potential cycle defect:");
        print_graph_state(&graph);

        let cycles = detect_cycles(&graph);

        if cycles.is_empty() {
            println!("\nNo cycles detected ✓");
        } else {
            println!("\nDetected {} cycle(s):", cycles.len());
            for (i, cycle) in cycles.iter().enumerate() {
                println!(
                    "  Cycle #{}: {}",
                    i + 1,
                    cycle
                        .iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(" → ")
                );

                // Analyze cycle severity
                let back_edge_prob = graph
                    .get_edge_probability(cycle[cycle.len() - 1], cycle[0])
                    .unwrap_or(0.0);

                if back_edge_prob > 0.5 {
                    println!("    Severity: HIGH (strong back edge)");
                    println!("    Impact: May cause unstable gradient flow");
                } else {
                    println!("    Severity: LOW (weak back edge)");
                    println!("    Impact: Minimal effect on forward pass");
                }
            }
        }
    }

    /// Example 7: Complete defect detection pipeline
    pub fn complete_defect_detection_pipeline() {
        println!("=== Example 7: Complete Defect Detection Pipeline ===\n");

        // Create a realistic transformer attention graph
        let num_layers = 6;
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(num_layers);

        // Main attention path (should be strong)
        for i in 0..num_layers - 1 {
            let prob = if i == 2 { 0.3 } else { 0.85 }; // Layer 2→3 is weak (defect)
            graph.add_learnable_edge(i, i + 1, prob);
        }

        // Some skip connections
        graph.add_learnable_edge(0, 2, 0.6);
        graph.add_learnable_edge(1, 3, 0.2); // Weak skip
        graph.add_learnable_edge(3, 5, 0.7);

        // Redundant long-range
        graph.add_learnable_edge(0, 5, 0.05);

        println!("Analyzing transformer attention topology...\n");

        // Run all defect detection algorithms
        let mut all_defects: Vec<TopologyDefect> = Vec::new();

        // 1. Weak connection detection
        let weak_defects = detect_weak_connections(&graph, 0.4);
        all_defects.extend(weak_defects);

        // 2. Redundant connection detection
        let redundant_defects = detect_redundant_connections(&graph, 0.15);
        all_defects.extend(redundant_defects);

        // 3. Bottleneck detection
        let bottlenecks = detect_bottlenecks(&graph, 0.3);
        for (node, _info) in bottlenecks {
            all_defects.push(TopologyDefect {
                defect_type: DefectType::BottleneckNode,
                location: format!("Layer {}", node),
                severity: 0.6,
                suggestion: "Consider adding skip connections around this layer".to_string(),
            });
        }

        // 4. Cycle detection
        let cycles = detect_cycles(&graph);
        if !cycles.is_empty() {
            all_defects.push(TopologyDefect {
                defect_type: DefectType::CycleDefect,
                location: format!("Layers {:?}", cycles[0]),
                severity: 0.4,
                suggestion: "Review cyclic dependencies for gradient stability".to_string(),
            });
        }

        // Report defects sorted by severity
        println!("=== Topology Defect Report ===\n");
        println!("Total defects found: {}\n", all_defects.len());

        let mut sorted_defects = all_defects.clone();
        sorted_defects.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap());

        for (i, defect) in sorted_defects.iter().enumerate() {
            let _severity_icon = match (defect.severity * 10.0) as u32 {
                0..=3 => "🟢",
                4..=6 => "🟡",
                _ => "🔴",
            };

            println!(
                "{}. {} [{:.1}/10.0] {}",
                i + 1,
                defect_type_to_string(&defect.defect_type),
                defect.severity * 10.0,
                defect.location
            );
            println!("   {}", defect.suggestion);
            println!();
        }

        // Summary statistics
        let critical_count = sorted_defects.iter().filter(|d| d.severity > 0.7).count();
        let warning_count = sorted_defects
            .iter()
            .filter(|d| d.severity > 0.4 && d.severity <= 0.7)
            .count();
        let info_count = sorted_defects.iter().filter(|d| d.severity <= 0.4).count();

        println!("=== Summary ===");
        println!(
            "  🔴 Critical: {} | 🟡 Warning: {} | 🟢 Info: {}",
            critical_count, warning_count, info_count
        );
    }

    // Helper functions

    fn detect_topology_defects<T: Clone + Default>(
        graph: &DifferentiableGraph<T>,
    ) -> Vec<TopologyDefect> {
        let mut defects = Vec::new();

        // Detect weak connections
        let weak = detect_weak_connections(graph, 0.3);
        defects.extend(weak);

        // Detect redundant connections
        let redundant = detect_redundant_connections(graph, 0.1);
        defects.extend(redundant);

        defects
    }

    fn detect_weak_connections<T: Clone + Default>(
        graph: &DifferentiableGraph<T>,
        threshold: f64,
    ) -> Vec<TopologyDefect> {
        let mut defects = Vec::new();
        let prob_matrix = graph.get_probability_matrix();

        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.0 && prob < threshold {
                    defects.push(TopologyDefect {
                        defect_type: DefectType::WeakAttention,
                        location: format!("Edge {}→{}", i, j),
                        severity: 1.0 - prob / threshold,
                        suggestion: format!(
                            "Consider strengthening connection {}→{} (current P={:.3})",
                            i, j, prob
                        ),
                    });
                }
            }
        }

        defects
    }

    fn detect_redundant_connections<T: Clone + Default>(
        graph: &DifferentiableGraph<T>,
        threshold: f64,
    ) -> Vec<TopologyDefect> {
        let mut defects = Vec::new();
        let prob_matrix = graph.get_probability_matrix();

        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                // Long-range connections with low probability are likely redundant
                if j > i + 1 && prob < threshold {
                    defects.push(TopologyDefect {
                        defect_type: DefectType::RedundantConnection,
                        location: format!("Skip edge {}→{}", i, j),
                        severity: (threshold - prob) / threshold,
                        suggestion: format!(
                            "Consider removing redundant skip connection {}→{} (P={:.3})",
                            i, j, prob
                        ),
                    });
                }
            }
        }

        defects
    }

    fn detect_bottlenecks<T: Clone + Default>(
        graph: &DifferentiableGraph<T>,
        threshold: f64,
    ) -> HashMap<usize, (usize, usize, f64)> {
        let mut bottlenecks = HashMap::new();
        let prob_matrix = graph.get_probability_matrix();

        for node in 0..graph.num_nodes() {
            let in_degree = prob_matrix
                .iter()
                .filter(|row| row.get(node).is_some_and(|&p| p > threshold))
                .count();

            let out_degree = prob_matrix
                .get(node)
                .map(|row| row.iter().filter(|&&p| p > threshold).count())
                .unwrap_or(0);

            let total_degree = in_degree + out_degree;
            if total_degree >= 4 {
                // High degree node
                let imbalance = (in_degree as i32 - out_degree as i32).abs() as f64;
                if imbalance > 2.0 {
                    bottlenecks.insert(node, (in_degree, out_degree, imbalance));
                }
            }
        }

        bottlenecks
    }

    fn detect_cycles<T: Clone + Default>(graph: &DifferentiableGraph<T>) -> Vec<Vec<usize>> {
        let mut cycles = Vec::new();
        let prob_matrix = graph.get_probability_matrix();
        let num_nodes = graph.num_nodes();

        // Simple DFS-based cycle detection
        let mut visited = vec![false; num_nodes];
        let mut rec_stack = vec![false; num_nodes];
        let mut path = Vec::new();

        for node in 0..num_nodes {
            if !visited[node] {
                dfs_cycle(
                    node,
                    &prob_matrix,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    &mut cycles,
                );
            }
        }

        cycles
    }

    fn dfs_cycle(
        node: usize,
        prob_matrix: &[Vec<f64>],
        visited: &mut [bool],
        rec_stack: &mut [bool],
        path: &mut Vec<usize>,
        cycles: &mut Vec<Vec<usize>>,
    ) {
        visited[node] = true;
        rec_stack[node] = true;
        path.push(node);

        for next in 0..prob_matrix.len() {
            if prob_matrix[node].get(next).is_some_and(|&p| p > 0.5) {
                if !visited[next] {
                    dfs_cycle(next, prob_matrix, visited, rec_stack, path, cycles);
                } else if rec_stack[next] {
                    // Found cycle
                    let cycle_start = path.iter().position(|&n| n == next).unwrap();
                    let cycle = path[cycle_start..].to_vec();
                    cycles.push(cycle);
                }
            }
        }

        path.pop();
        rec_stack[node] = false;
    }

    fn classify_gradient(grad: f64) -> &'static str {
        if grad < -0.5 {
            "CRITICAL"
        } else if grad < -0.1 {
            "IMPORTANT"
        } else if grad < 0.1 {
            "NEUTRAL"
        } else if grad < 0.5 {
            "REDUNDANT"
        } else {
            "HARMFUL"
        }
    }

    fn defect_type_to_string(defect: &DefectType) -> &'static str {
        match defect {
            DefectType::MissingConnection => "Missing Connection",
            DefectType::RedundantConnection => "Redundant Connection",
            DefectType::BottleneckNode => "Bottleneck Node",
            DefectType::IsolatedSubgraph => "Isolated Subgraph",
            DefectType::CycleDefect => "Cycle Defect",
            DefectType::WeakAttention => "Weak Attention",
        }
    }

    fn print_graph_state<T: Clone + Default>(graph: &DifferentiableGraph<T>) {
        let prob_matrix = graph.get_probability_matrix();
        println!(
            "  Nodes: {}, Edges: {}, Temperature: {:.4}",
            graph.num_nodes(),
            graph.num_edges(),
            graph.temperature()
        );

        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.01 {
                    let status = if prob > 0.7 {
                        "STRONG"
                    } else if prob > 0.3 {
                        "MODERATE"
                    } else {
                        "WEAK"
                    };
                    println!("    {}→{}: P = {:.4} [{}]", i, j, prob, status);
                }
            }
        }
    }

    pub fn run_all_examples() {
        basic_defect_detection();
        gradient_sensitivity_analysis();
        bottleneck_node_detection();
        missing_connection_detection();
        redundant_connection_pruning();
        cycle_defect_detection();
        complete_defect_detection_pipeline();
    }
}

fn main() {
    #[cfg(feature = "tensor")]
    topology_defect_detection::run_all_examples();

    #[cfg(not(feature = "tensor"))]
    println!(
        "Please enable tensor feature: cargo run --example topology_defect_detection --features tensor"
    );
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "tensor")]
    #[test]
    fn test_topology_defect_compiles() {
        use god_gragh::tensor::differentiable::DifferentiableGraph;

        let _graph = DifferentiableGraph::<Vec<f64>>::new(4);
    }
}
