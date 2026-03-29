//! CAD-LLM Model Switch Example
//!
//! This example demonstrates how to:
//! 1. Load a model from Safetensors format
//! 2. Validate the topology
//! 3. Verify weights
//! 4. Export back to Safetensors

#[cfg(feature = "safetensors")]
use god_gragh::transformer::optimization::ModelSwitch;

#[cfg(feature = "safetensors")]
use god_gragh::graph::traits::{GraphBase, GraphOps};
#[cfg(feature = "safetensors")]
use god_gragh::graph::Graph;
#[cfg(feature = "safetensors")]
use god_gragh::transformer::optimization::switch::{OperatorType, WeightTensor};

#[cfg(feature = "safetensors")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CAD-LLM Model Switch Example ===\n");

    // Create a demo graph for demonstration
    println!("Creating demo GodGraph...");
    let mut graph = Graph::<OperatorType, WeightTensor>::directed();

    let embed_node = graph
        .add_node(OperatorType::Embedding {
            vocab_size: 1000,
            embed_dim: 128,
        })
        .unwrap();

    let attn_node = graph
        .add_node(OperatorType::Attention {
            num_heads: 8,
            hidden_dim: 256,
        })
        .unwrap();

    let mlp_node = graph
        .add_node(OperatorType::MLP {
            hidden_dim: 512,
            activation: "silu".to_string(),
        })
        .unwrap();

    let norm_node = graph
        .add_node(OperatorType::Norm {
            norm_type: "rmsnorm".to_string(),
            eps: 1e-6,
        })
        .unwrap();

    // Add weight tensors
    graph
        .add_edge(
            embed_node,
            embed_node,
            WeightTensor::new(
                "model.embeddings.weight".to_string(),
                vec![1.0; 1000 * 128],
                vec![1000, 128],
            ),
        )
        .unwrap();

    graph
        .add_edge(
            attn_node,
            attn_node,
            WeightTensor::new(
                "model.layers.0.attention.qkv.weight".to_string(),
                vec![0.5; 256 * 3 * 256],
                vec![256, 3, 256],
            ),
        )
        .unwrap();

    graph
        .add_edge(
            mlp_node,
            mlp_node,
            WeightTensor::new(
                "model.layers.0.mlp.fc1.weight".to_string(),
                vec![0.25; 256 * 512],
                vec![256, 512],
            ),
        )
        .unwrap();

    graph
        .add_edge(
            norm_node,
            norm_node,
            WeightTensor::new("model.norm.weight".to_string(), vec![1.0; 256], vec![256]),
        )
        .unwrap();

    println!(
        "Created graph with {} nodes and {} edges\n",
        graph.node_count(),
        graph.edge_count()
    );

    // 1. Validate topology
    println!("Step 1: Validating topology...");
    let report = ModelSwitch::validate_topology(&graph)?;
    println!("  Topology valid: {}", report.is_valid);
    println!("  Connected components: {}", report.connected_components);
    println!("  Is DAG: {}", report.is_dag);
    println!("  Issues: {:?}", report.issues);
    println!();

    // 2. Export to Safetensors
    println!("Step 2: Exporting to Safetensors...");
    let export_path = "demo_export.safetensors";
    ModelSwitch::save_to_safetensors(&graph, export_path)?;
    println!("  Exported to: {}", export_path);

    // Check file size
    let metadata = std::fs::metadata(export_path)?;
    println!("  File size: {:.2} KB\n", metadata.len() as f64 / 1024.0);

    // 3. Load back from Safetensors
    println!("Step 3: Loading back from Safetensors...");
    let loaded_graph = ModelSwitch::load_from_safetensors(export_path)?;
    println!(
        "  Loaded graph with {} nodes and {} edges\n",
        loaded_graph.node_count(),
        loaded_graph.edge_count()
    );

    // 4. Verify weights
    println!("Step 4: Verifying weights...");
    let diff = ModelSwitch::verify_weights(&graph, &loaded_graph)?;
    println!("  Max L2 difference: {:.6e}", diff.max_l2_diff);
    println!("  Avg L2 difference: {:.6e}", diff.avg_l2_diff);
    println!("  Tensor count: {}\n", diff.tensor_count);

    // Cleanup
    std::fs::remove_file(export_path)?;
    println!("=== Example Complete ===");
    println!("\nNote: The current implementation exports weight data only.");
    println!("Graph topology is inferred from weight names during loading.");

    Ok(())
}

#[cfg(not(feature = "safetensors"))]
fn main() {
    println!("This example requires the 'safetensors' feature.");
    println!("Run with: cargo run --example cad_llm_switch --features safetensors");
}

#[cfg(test)]
mod tests {
    use super::*;
    use god_gragh::graph::Graph;
    use god_gragh::transformer::optimization::switch::{OperatorType, WeightTensor};

    #[test]
    #[cfg(feature = "safetensors")]
    fn test_model_switch_api() {
        // Create a simple test graph
        let mut graph = Graph::<OperatorType, WeightTensor>::directed();

        let node = graph
            .add_node(OperatorType::Linear {
                in_features: 512,
                out_features: 512,
            })
            .unwrap();

        // Add a weight
        graph
            .add_edge(
                node,
                node,
                WeightTensor::new(
                    "test.weight".to_string(),
                    vec![1.0; 512 * 512],
                    vec![512, 512],
                ),
            )
            .unwrap();

        // Validate topology
        let report = ModelSwitch::validate_topology(&graph).unwrap();
        println!("Topology report: {:?}", report);
    }
}
