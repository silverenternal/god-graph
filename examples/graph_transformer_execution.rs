//! GraphTransformer Execution Example
//!
//! This example demonstrates how to use the GraphTransformer for forward execution
//! with explicit tensor passing on edges.
//!
//! ## What this example shows:
//!
//! 1. **Building a Transformer computation graph** from input token IDs
//! 2. **Executing forward pass** with topological sorting
//! 3. **Tensor message passing** on edges (Q/K/V projections)
//! 4. **Visualizing the computation graph** in DOT format
//! 5. **Pruning weak attention edges** dynamically
//!
//! ## GraphTransformer Positioning
//!
//! **GraphTransformer is designed for**:
//! - ✅ Visualizing attention topology
//! - ✅ Dynamic edge pruning (weak attention removal)
//! - ✅ Custom connection experiments (long-range, sparse attention)
//! - ✅ Topology defect detection
//! - ✅ Educational/research purposes
//!
//! **GraphTransformer is NOT for**:
//! - ❌ High-performance inference (use llama.cpp or vllm)
//! - ❌ Large-scale training (use PyTorch or JAX)
//!
//! ## Running this example
//!
//! ```bash
//! cargo run --example graph_transformer_execution --features tensor
//! ```

use god_gragh::tensor::traits::TensorBase;
use god_gragh::tensor::DenseTensor;
use god_gragh::transformer::graph_transformer::GraphTransformer;

fn main() {
    println!("=== GraphTransformer Execution Example ===\n");

    // Configuration
    let num_layers = 2;
    let num_heads = 4;
    let hidden_dim = 256;
    let input_ids = vec![1, 2, 3, 4, 5]; // Sequence of 5 tokens

    println!("Configuration:");
    println!("  Layers: {}", num_layers);
    println!("  Heads: {}", num_heads);
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Sequence length: {}", input_ids.len());
    println!();

    // 1. Create GraphTransformer
    println!("1. Creating GraphTransformer...");
    let mut transformer = GraphTransformer::new(num_layers, num_heads, hidden_dim);

    // 2. Build computation graph
    println!("2. Building computation graph from input IDs...");
    transformer.build_graph(&input_ids);

    println!("  Graph built:");
    println!("    Nodes: {}", transformer.num_nodes());
    println!("    Edges: {}", transformer.num_edges());
    println!();

    // 3. Export to DOT format for visualization
    println!("3. Exporting computation graph to DOT format...");
    let dot = transformer.to_dot();
    std::fs::write("graph_transformer.dot", &dot).expect("Failed to write DOT file");
    println!("  Saved to: graph_transformer.dot");
    println!("  To visualize: dot -Tpng graph_transformer.dot -o graph_transformer.png");
    println!();

    // 4. Execute forward pass
    println!("4. Executing forward pass...");
    let output = transformer.forward(&input_ids);
    println!("  Forward pass completed!");
    println!("  Output shape: {:?}", output.shape());
    println!("  Output dtype: {:?}", output.dtype());

    // Print first few values of output
    let output_data = output.data();
    println!("  Output (first 10 values):");
    for (i, &val) in output_data
        .iter()
        .take(10.min(output_data.len()))
        .enumerate()
    {
        println!("    [{}]: {:.6}", i, val);
    }
    println!();

    // 5. Analyze attention patterns
    println!("5. Analyzing attention patterns...");
    analyze_attention_patterns(&transformer);
    println!();

    // 6. Prune weak attention edges
    println!("6. Pruning weak attention edges (threshold=0.05)...");
    let threshold = 0.05;
    let pruned_count = transformer.prune_weak_edges(threshold);
    println!(
        "  Pruned {} edges with weight < {}",
        pruned_count, threshold
    );
    println!("  Remaining edges: {}", transformer.num_edges());
    println!();

    // 7. Re-execute after pruning
    println!("7. Re-executing forward pass after pruning...");
    let output_pruned = transformer.forward(&input_ids);
    println!("  Output shape: {:?}", output_pruned.shape());

    // Compare outputs
    let output_pruned_data = output_pruned.data();
    println!("  Output (first 10 values):");
    for (i, &val) in output_pruned_data
        .iter()
        .take(10.min(output_pruned_data.len()))
        .enumerate()
    {
        println!("    [{}]: {:.6}", i, val);
    }
    println!();

    // 8. Export pruned graph
    println!("8. Exporting pruned graph to DOT format...");
    let dot_pruned = transformer.to_dot();
    std::fs::write("graph_transformer_pruned.dot", &dot_pruned)
        .expect("Failed to write pruned DOT file");
    println!("  Saved to: graph_transformer_pruned.dot");
    println!();

    // 9. Demonstrate edge tensor passing
    println!("9. Demonstrating edge tensor passing semantics...");
    demonstrate_edge_tensor_passing();
    println!();

    println!("=== Example completed successfully! ===");
}

/// Analyze attention patterns in the graph
fn analyze_attention_patterns(transformer: &GraphTransformer) {
    // Note: In a real implementation, we would inspect edge weights
    // This is a placeholder for demonstration
    println!("  Attention analysis:");
    println!("    - Total nodes: {}", transformer.num_nodes());
    println!("    - Total edges: {}", transformer.num_edges());
    println!(
        "    - Average edges per node: {:.2}",
        transformer.num_edges() as f64 / transformer.num_nodes() as f64
    );
}

/// Demonstrate edge tensor passing with Q/K/V projections
fn demonstrate_edge_tensor_passing() {
    use god_gragh::transformer::graph_transformer::edges::{DataFlowOp, GraphEdge, SkipType};

    println!("  Edge tensor passing semantics:");

    // 1. Self-attention edge with Q projection
    let q_proj = DenseTensor::zeros(vec![1, 64]);
    let sa_edge = GraphEdge::self_attention_with_message(0, 1, 0.8, 2, 0, q_proj);
    println!("    - SelfAttention edge:");
    println!("        Has message: {}", sa_edge.message().is_some());
    println!(
        "        Attention weight: {:?}",
        sa_edge.get_self_attention().map(|s| s.weight)
    );

    // 2. DataFlow edge with activation tensor
    let activation = DenseTensor::zeros(vec![1, 256]);
    let df_edge =
        GraphEdge::data_flow_with_message(1, 2, DataFlowOp::AttentionToOutput, 0, activation);
    println!("    - DataFlow edge:");
    println!("        Has message: {}", df_edge.message().is_some());
    println!(
        "        Operation: {:?}",
        df_edge.get_data_flow().map(|d| d.operation)
    );

    // 3. Residual edge with passthrough tensor
    let residual = DenseTensor::zeros(vec![1, 256]);
    let res_edge = GraphEdge::residual_with_tensor(2, 3, 0, SkipType::PreNorm, residual);
    println!("    - Residual edge:");
    println!("        Has message: {}", res_edge.message().is_some());
    println!(
        "        Skip type: {:?}",
        res_edge.get_residual().map(|r| r.skip_type)
    );

    println!();
    println!("  Tensor passing workflow:");
    println!("    1. Source node computes output tensor");
    println!("    2. Edge carries tensor as message (Q/K/V projection)");
    println!("    3. Target node receives and aggregates messages");
    println!("    4. Attention weights determine contribution of each message");
}

#[cfg(test)]
mod tests {
    use super::*;
    use god_gragh::transformer::graph_transformer::edges::{DataFlowOp, GraphEdge};

    #[test]
    fn test_graph_transformer_basic_execution() {
        let mut transformer = GraphTransformer::new(2, 4, 256);
        let input_ids = vec![1, 2, 3, 4];

        transformer.build_graph(&input_ids);
        let output = transformer.forward(&input_ids);

        assert_eq!(output.ndim(), 2);
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 256);
    }

    #[test]
    fn test_edge_tensor_passing() {
        // Test SelfAttention edge with message
        let q_proj = DenseTensor::zeros(vec![1, 64]);
        let mut sa_edge = GraphEdge::self_attention_with_message(0, 1, 0.8, 2, 0, q_proj);

        assert!(sa_edge.message().is_some());
        assert_eq!(sa_edge.message().unwrap().shape(), &[1, 64]);

        // Test DataFlow edge with message
        let activation = DenseTensor::zeros(vec![1, 256]);
        let df_edge =
            GraphEdge::data_flow_with_message(1, 2, DataFlowOp::AttentionToOutput, 0, activation);

        assert!(df_edge.message().is_some());
        assert_eq!(df_edge.message().unwrap().shape(), &[1, 256]);
    }

    #[test]
    fn test_graph_export() {
        let mut transformer = GraphTransformer::new(1, 2, 128);
        transformer.build_graph(&[1, 2, 3]);

        let dot = transformer.to_dot();

        assert!(dot.contains("digraph Transformer"));
        assert!(dot.contains("rankdir=TB"));
    }
}
