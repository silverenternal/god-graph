//! CAD-LLM Topology Editor Example
//!
//! This example demonstrates how to:
//! 1. Create a CAD-style topology editor
//! 2. Detect topology defects
//! 3. Add and solve constraints
//! 4. Extract and replace modules
//! 5. Use edit history for rollback

use god_gragh::graph::traits::{GraphBase, GraphOps};
use god_gragh::graph::Graph;
use god_gragh::transformer::optimization::switch::{OperatorType, WeightTensor};
use god_gragh::transformer::optimization::{CadStyleEditor, TopologyConstraint};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CAD-LLM Topology Editor Example ===\n");

    // Create a sample computation graph
    let mut graph = create_sample_transformer_graph();
    println!("Created sample transformer graph:");
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}\n", graph.edge_count());

    // 1. Create editor
    println!("Step 1: Creating CAD-style editor...");
    let mut editor = CadStyleEditor::with_defaults(&mut graph);
    println!("  Editor created with default transformer constraints\n");

    // 2. Detect topology defects
    println!("Step 2: Detecting topology defects...");
    let defects = editor.detect_defects()?;
    println!("  Found {} defects:", defects.len());
    print_defects(&defects);
    println!();

    // 3. Add custom constraints
    println!("Step 3: Adding custom constraints...");
    editor.add_constraint(TopologyConstraint::ResidualConnection {
        from_layer: "attention".to_string(),
        to_layer: "output".to_string(),
    })?;
    editor.add_constraint(TopologyConstraint::GradientFlow {
        from: "embedding".to_string(),
        to: "lm_head".to_string(),
    })?;
    println!("  Added ResidualConnection constraint");
    println!("  Added GradientFlow constraint\n");

    // 4. Solve constraints
    println!("Step 4: Solving constraints...");
    let report = editor.solve_constraints()?;
    println!("  All constraints satisfied: {}", report.all_satisfied);
    println!(
        "  Satisfied: {}/{}",
        report.satisfied_count,
        report.satisfied_count + report.violated_count
    );

    for detail in &report.constraint_details {
        println!(
            "    - {}: {}",
            detail.description,
            if detail.satisfied { "✓" } else { "✗" }
        );
    }
    println!();

    // 5. Extract module
    println!("Step 5: Extracting module...");
    let module = editor.extract_module("attention")?;
    println!("  Extracted module:");
    println!("    Nodes: {}", module.node_count());
    println!("    Edges: {}", module.edge_count());
    println!(
        "    Cached: {}\n",
        editor.module_cache().contains_key("attention")
    );

    // 6. Validate assembly
    println!("Step 6: Validating assembly...");
    let assembly_report = editor.validate_assembly()?;
    println!("  Assembly valid: {}", assembly_report.is_valid);
    println!("  Modules checked: {}", assembly_report.module_count);
    println!(
        "  Interface mismatches: {}\n",
        assembly_report.interface_mismatches
    );

    // 7. Edit history
    println!("Step 7: Checking edit history...");
    println!("  History entries: {}", editor.history_len());
    for (i, entry) in editor.history().iter().enumerate() {
        println!(
            "    [{}] {} ({} operations)",
            i,
            entry.description,
            entry.operations.len()
        );
    }
    println!();

    // 8. Rollback demonstration
    println!("Step 8: Rollback demonstration...");
    if editor.history_len() > 0 {
        let success = editor.undo()?;
        println!("  Undo successful: {}", success);
    } else {
        println!("  No operations to undo");
    }
    println!();

    println!("=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("  - CAD-style editing brings engineering rigor to LLM optimization");
    println!("  - Topology defects are automatically detected and fixed");
    println!("  - Constraints ensure architectural integrity");
    println!("  - Edit history enables safe experimentation");

    Ok(())
}

/// Create a sample transformer computation graph
fn create_sample_transformer_graph() -> Graph<OperatorType, WeightTensor> {
    let mut graph = Graph::<OperatorType, WeightTensor>::directed();

    // Add embedding layer
    let _embedding = graph
        .add_node(OperatorType::Embedding {
            vocab_size: 32000,
            embed_dim: 512,
        })
        .unwrap();

    // Add attention layer
    let _attention = graph
        .add_node(OperatorType::Attention {
            num_heads: 8,
            hidden_dim: 512,
        })
        .unwrap();

    // Add MLP layer
    let _mlp = graph
        .add_node(OperatorType::MLP {
            hidden_dim: 2048,
            activation: "gelu".to_string(),
        })
        .unwrap();

    // Add normalization
    let _norm = graph
        .add_node(OperatorType::Norm {
            norm_type: "layernorm".to_string(),
            eps: 1e-6,
        })
        .unwrap();

    // Add residual connections
    let _residual1 = graph.add_node(OperatorType::Residual).unwrap();
    let _residual2 = graph.add_node(OperatorType::Residual).unwrap();

    // Add LM head
    let _lm_head = graph
        .add_node(OperatorType::Linear {
            in_features: 512,
            out_features: 32000,
        })
        .unwrap();

    graph
}

/// Print topology defects in a readable format
fn print_defects(defects: &[god_gragh::transformer::optimization::constraints::TopologyDefect]) {
    if defects.is_empty() {
        println!("    No defects found!");
        return;
    }

    for (i, defect) in defects.iter().enumerate() {
        let severity = match defect.severity {
            god_gragh::transformer::optimization::constraints::Severity::Info => "ℹ",
            god_gragh::transformer::optimization::constraints::Severity::Warning => "⚠",
            god_gragh::transformer::optimization::constraints::Severity::Error => "✗",
            god_gragh::transformer::optimization::constraints::Severity::Critical => "‼",
        };

        println!(
            "    [{}] {}: {:?} - {}",
            i + 1,
            severity,
            defect.defect_type,
            defect.description
        );
        if let Some(fix) = &defect.suggested_fix {
            println!("        → {}", fix);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use god_gragh::transformer::optimization::constraints::Severity;

    #[test]
    fn test_editor_creation() {
        let mut graph = create_sample_transformer_graph();
        let editor = CadStyleEditor::new(&mut graph);

        assert_eq!(editor.history_len(), 0);
    }

    #[test]
    fn test_defect_detection() {
        let mut graph = Graph::<OperatorType, WeightTensor>::directed();

        // Add isolated node
        graph
            .add_node(OperatorType::Linear {
                in_features: 512,
                out_features: 512,
            })
            .unwrap();

        let editor = CadStyleEditor::new(&mut graph);
        let defects = editor.detect_defects().unwrap();

        assert!(!defects.is_empty());
    }

    #[test]
    fn test_constraint_solving() {
        let mut graph = create_sample_transformer_graph();
        let mut editor = CadStyleEditor::with_defaults(&mut graph);

        editor
            .add_constraint(TopologyConstraint::ResidualConnection {
                from_layer: "attn".to_string(),
                to_layer: "output".to_string(),
            })
            .unwrap();

        let report = editor.solve_constraints().unwrap();
        assert!(report.satisfied_count + report.violated_count > 0);
    }

    #[test]
    fn test_module_extraction() {
        let mut graph = create_sample_transformer_graph();
        let mut editor = CadStyleEditor::new(&mut graph);

        let module = editor.extract_module("attention").unwrap();
        assert!(editor.module_cache().contains_key("attention"));
    }
}
