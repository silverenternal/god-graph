//! Export Transformer model to DOT format for visualization
//!
//! This example demonstrates how to:
//! 1. Load a model from Safetensors format
//! 2. Export the architecture to DOT (Graphviz) format
//! 3. Visualize the model structure
//!
//! ## Usage
//!
//! ```bash
//! # Export to DOT format
//! cargo run --example export_model_dot --features transformer -- model.safetensors model.dot
//!
//! # Render to PNG using Graphviz
//! dot -Tpng model.dot -o model.png
//!
//! # Render to SVG
//! dot -Tsvg model.dot -o model.svg
//! ```

use god_gragh::export::dot::{write_dot_to_file, DotOptions};
use god_gragh::graph::traits::{GraphBase, GraphQuery};
use god_gragh::graph::Graph;
use god_gragh::transformer::optimization::switch::{ModelSwitch, OperatorType, WeightTensor};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: cargo run --example export_model_dot --features transformer -- [model.safetensors] [output.dot]");
        println!();
        println!("Arguments:");
        println!("  model.safetensors  Input model file (default: test_model.safetensors)");
        println!("  output.dot         Output DOT file (default: model.dot)");
        println!();
        println!("After exporting, visualize with Graphviz:");
        println!("  dot -Tpng model.dot -o model.png");
        println!("  dot -Tsvg model.dot -o model.svg");
        println!();
        println!("Running with default test model...");

        // Create a demo transformer graph for demonstration
        let graph = create_demo_transformer();
        export_transformer_to_dot(&graph, "model.dot")?;
        println!("✓ Exported demo transformer to model.dot");
    } else {
        let input_path = &args[1];
        let output_path = if args.len() > 2 {
            &args[2]
        } else {
            "model.dot"
        };

        println!("Loading model from: {}", input_path);
        let graph = ModelSwitch::load_from_safetensors(input_path)?;
        println!(
            "✓ Loaded model with {} nodes, {} edges",
            graph.node_count(),
            graph.edge_count()
        );

        export_transformer_to_dot(&graph, output_path)?;
        println!("✓ Exported to {}", output_path);
    }

    Ok(())
}

/// Create a demo transformer graph for visualization
fn create_demo_transformer() -> Graph<OperatorType, WeightTensor> {
    use god_gragh::graph::builders::GraphBuilder;
    use rand::prelude::*;

    let mut rng = rand::thread_rng();

    let mut builder = GraphBuilder::directed();

    // Embedding layer (node 0)
    builder = builder.with_node(OperatorType::Embedding {
        vocab_size: 32000,
        embed_dim: 512,
    });

    // Transformer layers: each layer has 4 nodes (Attn, Norm1, MLP, Norm2)
    // Layer 0: nodes 1,2,3,4
    // Layer 1: nodes 5,6,7,8
    // Layer 2: nodes 9,10,11,12
    let mut prev_layer_idx = 0;
    for layer_idx in 0..3 {
        let base_idx = 1 + layer_idx * 4;

        // Attention (nodes 1, 5, 9)
        let attn_data: Vec<f64> = (0..512 * 512).map(|_| rng.gen()).collect();
        builder = builder.with_node(OperatorType::Attention {
            num_heads: 8,
            hidden_dim: 512,
        });
        builder = builder.with_edge(
            prev_layer_idx,
            base_idx,
            WeightTensor::new("attn".to_string(), attn_data, vec![512, 512]),
        );

        // Norm after attention (nodes 2, 6, 10)
        let norm1_data: Vec<f64> = (0..512).map(|_| 1.0).collect();
        builder = builder.with_node(OperatorType::Norm {
            norm_type: "RMSNorm".to_string(),
            eps: 1e-6,
        });
        builder = builder.with_edge(
            base_idx,
            base_idx + 1,
            WeightTensor::new("norm1".to_string(), norm1_data, vec![512]),
        );

        // Residual connection (skip connection)
        let res1_data: Vec<f64> = (0..512).map(|_| 1.0).collect();
        builder = builder.with_edge(
            prev_layer_idx,
            base_idx + 1,
            WeightTensor::new("residual1".to_string(), res1_data, vec![512]),
        );

        // MLP (nodes 3, 7, 11)
        let mlp1_data: Vec<f64> = (0..512 * 2048).map(|_| rng.gen()).collect();
        builder = builder.with_node(OperatorType::MLP {
            hidden_dim: 512,
            activation: "SiLU".to_string(),
        });
        builder = builder.with_edge(
            base_idx + 1,
            base_idx + 2,
            WeightTensor::new("mlp1".to_string(), mlp1_data, vec![512, 2048]),
        );

        // Norm after MLP (nodes 4, 8, 12)
        let norm2_data: Vec<f64> = (0..512).map(|_| 1.0).collect();
        builder = builder.with_node(OperatorType::Norm {
            norm_type: "RMSNorm".to_string(),
            eps: 1e-6,
        });
        builder = builder.with_edge(
            base_idx + 2,
            base_idx + 3,
            WeightTensor::new("norm2".to_string(), norm2_data, vec![512]),
        );

        // Residual connection
        let res2_data: Vec<f64> = (0..512).map(|_| 1.0).collect();
        builder = builder.with_edge(
            base_idx + 1,
            base_idx + 3,
            WeightTensor::new("residual2".to_string(), res2_data, vec![512]),
        );

        prev_layer_idx = base_idx + 3;
    }

    // Final output projection: node 13
    // prev_layer_idx = 12 after 3 layers
    let output_data: Vec<f64> = (0..512 * 32000).map(|_| rng.gen()).collect();
    builder = builder.with_node(OperatorType::Linear {
        in_features: 512,
        out_features: 32000,
    });
    builder = builder.with_edge(
        prev_layer_idx,
        prev_layer_idx + 1,
        WeightTensor::new("output".to_string(), output_data, vec![512, 32000]),
    );

    builder.build().unwrap()
}

/// Export transformer to DOT format with optimized layout
fn export_transformer_to_dot(
    graph: &god_gragh::graph::Graph<OperatorType, WeightTensor>,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create custom options for transformer visualization
    let options = DotOptions::new()
        .with_name("Transformer")
        .hide_edge_labels() // Hide edge labels for cleaner look
        .undirected(); // Remove arrowheads for cleaner look

    // Generate DOT content
    let mut dot = String::from("digraph Transformer {\n");

    // Graph attributes
    dot.push_str("    rankdir=TB;\n"); // Top to bottom
    dot.push_str("    splines=ortho;\n"); // Orthogonal edges
    dot.push_str("    node [shape=box, style=\"rounded,filled\"];\n");
    dot.push_str("    edge [color=gray, arrowsize=0.5];\n");
    dot.push_str("\n");

    // Collect nodes by type for subgraph clustering
    let mut embedding_nodes = Vec::new();
    let mut attention_nodes = Vec::new();
    let mut mlp_nodes = Vec::new();
    let mut norm_nodes = Vec::new();
    let mut linear_nodes = Vec::new();

    // Also store node data for later use
    let mut embedding_data = Vec::new();
    let mut attention_data = Vec::new();
    let mut mlp_data = Vec::new();
    let mut norm_data = Vec::new();
    let mut linear_data = Vec::new();

    for node in graph.nodes() {
        let idx: usize = node.index.index();
        match node.data {
            OperatorType::Embedding {
                vocab_size,
                embed_dim,
            } => {
                embedding_nodes.push(idx);
                embedding_data.push((vocab_size, embed_dim));
            }
            OperatorType::Attention {
                num_heads,
                hidden_dim,
            } => {
                attention_nodes.push(idx);
                attention_data.push((num_heads, hidden_dim));
            }
            OperatorType::MLP {
                hidden_dim,
                activation,
            } => {
                mlp_nodes.push(idx);
                mlp_data.push((hidden_dim, activation.clone()));
            }
            OperatorType::Norm { norm_type, eps } => {
                norm_nodes.push(idx);
                norm_data.push((norm_type.clone(), eps));
            }
            OperatorType::Linear {
                in_features,
                out_features,
            } => {
                linear_nodes.push(idx);
                linear_data.push((in_features, out_features));
            }
            _ => {}
        }
    }

    // Embedding subgraph
    if !embedding_nodes.is_empty() {
        dot.push_str("    subgraph cluster_embedding {\n");
        dot.push_str("        label=\"Embedding\";\n");
        dot.push_str("        style=filled;\n");
        dot.push_str("        fillcolor=lightyellow;\n");
        for (i, &idx) in embedding_nodes.iter().enumerate() {
            let (vocab_size, embed_dim) = embedding_data[i];
            dot.push_str(&format!(
                "        n{} [label=\"Embedding\\n{}x{}\" fillcolor=\"#FFC107\"];\n",
                idx, vocab_size, embed_dim
            ));
        }
        dot.push_str("    }\n\n");
    }

    // Attention layers subgraph
    if !attention_nodes.is_empty() {
        dot.push_str("    subgraph cluster_attention {\n");
        dot.push_str("        label=\"Attention Layers\";\n");
        dot.push_str("        style=filled;\n");
        dot.push_str("        fillcolor=lightgreen;\n");
        for (i, &idx) in attention_nodes.iter().enumerate() {
            let (num_heads, hidden_dim) = attention_data[i];
            dot.push_str(&format!(
                "        n{} [label=\"Attention\\n{} heads\\n{} dim\" fillcolor=\"#4CAF50\"];\n",
                idx, num_heads, hidden_dim
            ));
        }
        dot.push_str("    }\n\n");
    }

    // MLP layers subgraph
    if !mlp_nodes.is_empty() {
        dot.push_str("    subgraph cluster_mlp {\n");
        dot.push_str("        label=\"MLP Layers\";\n");
        dot.push_str("        style=filled;\n");
        dot.push_str("        fillcolor=lightcoral;\n");
        for (i, &idx) in mlp_nodes.iter().enumerate() {
            let (hidden_dim, activation) = &mlp_data[i];
            dot.push_str(&format!(
                "        n{} [label=\"MLP\\n{}\\n{}\" fillcolor=\"#f44336\"];\n",
                idx, hidden_dim, activation
            ));
        }
        dot.push_str("    }\n\n");
    }

    // Norm layers subgraph
    if !norm_nodes.is_empty() {
        dot.push_str("    subgraph cluster_norm {\n");
        dot.push_str("        label=\"Normalization Layers\";\n");
        dot.push_str("        style=filled;\n");
        dot.push_str("        fillcolor=lightblue;\n");
        for (i, &idx) in norm_nodes.iter().enumerate() {
            let (norm_type, eps) = &norm_data[i];
            dot.push_str(&format!(
                "        n{} [label=\"{}\\neps={:.0e}\" fillcolor=\"#2196F3\"];\n",
                idx, norm_type, eps
            ));
        }
        dot.push_str("    }\n\n");
    }

    // Linear layers subgraph
    if !linear_nodes.is_empty() {
        dot.push_str("    subgraph cluster_linear {\n");
        dot.push_str("        label=\"Linear Layers\";\n");
        dot.push_str("        style=filled;\n");
        dot.push_str("        fillcolor=lightgray;\n");
        for (i, &idx) in linear_nodes.iter().enumerate() {
            let (in_features, out_features) = linear_data[i];
            dot.push_str(&format!(
                "        n{} [label=\"Linear\\n{}→{}\" fillcolor=\"#9E9E9E\"];\n",
                idx, in_features, out_features
            ));
        }
        dot.push_str("    }\n\n");
    }

    // Export edges
    for edge in graph.edges() {
        let source: usize = edge.source.index();
        let target: usize = edge.target.index();
        dot.push_str(&format!("    n{} -> n{};\n", source, target));
    }

    dot.push_str("}");

    // Write to file
    write_dot_to_file(&dot, output_path)?;

    Ok(())
}
