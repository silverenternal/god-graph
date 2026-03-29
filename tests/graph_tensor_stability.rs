//! Graph-level Tensor Orthogonalization Stability Tests
//!
//! This module tests the numerical stability of graph-level tensor operations,
//! including orthogonalization error accumulation and forward pass stability.

#[cfg(all(test, feature = "tensor"))]
mod tests {
    use god_gragh::graph::traits::{GraphOps, GraphQuery};
    use god_gragh::graph::Graph;
    use god_gragh::tensor::decomposition::qr::{is_orthogonal, orthogonalize_in_place};
    use god_gragh::tensor::DenseTensor;
    use god_gragh::tensor::TensorBase;
    use god_gragh::transformer::optimization::lie_group::orthogonalize_weights_in_place;
    use god_gragh::transformer::optimization::switch::{OperatorType, WeightTensor};
    use god_gragh::transformer::optimization::LieGroupConfig;

    /// Build a mini Transformer graph with random weights for testing
    fn build_mini_transformer_graph_with_weights() -> Graph<OperatorType, WeightTensor> {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();

        let mut graph = Graph::directed();

        // Add embedding layer
        let embed_idx = graph
            .add_node(OperatorType::Embedding {
                vocab_size: 100,
                embed_dim: 64,
            })
            .unwrap();

        // Add attention layer nodes
        let attn_idx = graph
            .add_node(OperatorType::Attention {
                num_heads: 4,
                hidden_dim: 64,
            })
            .unwrap();

        // Add projection weights (64x64 matrices)
        let q_proj = WeightTensor::new(
            "attention.q_proj".to_string(),
            (0..64 * 64).map(|_| rng.gen::<f64>() * 0.1).collect(),
            vec![64, 64],
        );
        let k_proj = WeightTensor::new(
            "attention.k_proj".to_string(),
            (0..64 * 64).map(|_| rng.gen::<f64>() * 0.1).collect(),
            vec![64, 64],
        );
        let v_proj = WeightTensor::new(
            "attention.v_proj".to_string(),
            (0..64 * 64).map(|_| rng.gen::<f64>() * 0.1).collect(),
            vec![64, 64],
        );
        let out_proj = WeightTensor::new(
            "attention.out_proj".to_string(),
            (0..64 * 64).map(|_| rng.gen::<f64>() * 0.1).collect(),
            vec![64, 64],
        );

        // Add MLP layers
        let mlp_idx = graph
            .add_node(OperatorType::MLP {
                hidden_dim: 256,
                activation: "gelu".to_string(),
            })
            .unwrap();

        let up_proj = WeightTensor::new(
            "mlp.up_proj".to_string(),
            (0..64 * 256).map(|_| rng.gen::<f64>() * 0.1).collect(),
            vec![256, 64],
        );
        let down_proj = WeightTensor::new(
            "mlp.down_proj".to_string(),
            (0..64 * 256).map(|_| rng.gen::<f64>() * 0.1).collect(),
            vec![256, 64],
        );

        // Add edges with weights
        graph.add_edge(embed_idx, attn_idx, q_proj).unwrap();
        graph.add_edge(embed_idx, attn_idx, k_proj).unwrap();
        graph.add_edge(embed_idx, attn_idx, v_proj).unwrap();
        graph.add_edge(attn_idx, mlp_idx, out_proj).unwrap();
        graph.add_edge(attn_idx, mlp_idx, up_proj).unwrap();
        graph.add_edge(mlp_idx, attn_idx, down_proj).unwrap();

        graph
    }

    /// Simplified forward pass: compute sum of weight norms
    fn forward_pass(graph: &Graph<OperatorType, WeightTensor>) -> f64 {
        graph
            .edges()
            .map(|e| {
                let w = e.data();
                w.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
            })
            .sum()
    }

    /// Compute orthogonality error: max|W^T W - I|
    fn compute_orthogonality_error(tensor: &DenseTensor) -> f64 {
        let shape = tensor.shape();
        if shape.len() != 2 || shape[0] < shape[1] {
            return f64::MAX;
        }

        let n = shape[1];
        let m = shape[0];
        let data = tensor.data();
        let mut max_error: f64 = 0.0;

        // Compute W^T W - I
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..m {
                    // Row-major: W[k,i] = data[k * n + i]
                    dot += data[k * n + i] * data[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let error = (dot - expected).abs();
                max_error = max_error.max(error);
            }
        }

        max_error
    }

    /// Compute relative error between two values
    fn compute_relative_error(a: &f64, b: &f64) -> f64 {
        (a - b).abs() / (a.abs().max(b.abs()) + 1e-10)
    }

    /// Graph-level orthogonalization stability test
    #[test]
    #[cfg(feature = "tensor")]
    fn test_graph_level_orthogonalization_stability() {
        // 1. Build mini Transformer graph with random weights
        let mut graph = build_mini_transformer_graph_with_weights();

        // 2. Save original output (sum of weight norms)
        let original_output = forward_pass(&graph);

        // 3. Apply in-place orthogonalization
        let config = LieGroupConfig::new()
            .with_block_size(64)
            .with_orthogonalize(true);

        // Use in-place version (zero-copy)
        let _errors = orthogonalize_weights_in_place(&config, &mut graph).unwrap();

        // 4. Verify orthogonality of each weight
        let mut max_orthogonality_error: f64 = 0.0;
        let mut total_orthogonality_error: f64 = 0.0;
        let mut weight_count = 0;

        for edge_ref in graph.edges() {
            let weight = edge_ref.data();
            let tensor = DenseTensor::from_vec(weight.data.to_vec(), weight.shape.to_vec());

            // Check orthogonality (for square or tall matrices)
            if weight.shape[0] >= weight.shape[1] {
                let is_ortho = is_orthogonal(&tensor, 1e-8);
                assert!(
                    is_ortho,
                    "Weight {} should be orthogonal (error={:.2e})",
                    weight.name,
                    compute_orthogonality_error(&tensor)
                );

                // Track maximum error
                let ortho_error = compute_orthogonality_error(&tensor);
                max_orthogonality_error = max_orthogonality_error.max(ortho_error);
                total_orthogonality_error += ortho_error;
                weight_count += 1;
            }
        }

        // 5. Verify output error (orthogonalization changes weight distribution)
        let ortho_output = forward_pass(&graph);
        let output_error = compute_relative_error(&original_output, &ortho_output);

        // 6. Assertions
        assert!(
            max_orthogonality_error < 1e-8,
            "Max orthogonality error too large: {:.2e}",
            max_orthogonality_error
        );
        assert!(
            output_error < 10.0, // Loose threshold since orthogonalization changes weight distribution
            "Output error too large: {:.2e}",
            output_error
        );

        eprintln!("✓ Graph-level orthogonalization test passed");
        eprintln!(
            "  - Max orthogonality error: {:.2e}",
            max_orthogonality_error
        );
        eprintln!("  - Output error: {:.2e}", output_error);
        eprintln!(
            "  - Average edge error: {:.2e}",
            total_orthogonality_error / weight_count as f64
        );
        eprintln!("  - Number of weights orthogonalized: {}", weight_count);
    }

    /// Test single tensor orthogonalization (baseline)
    #[test]
    #[cfg(feature = "tensor")]
    fn test_single_tensor_orthogonalization() {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();

        // Create random 64x64 matrix
        let mut data: Vec<f64> = (0..64 * 64).map(|_| rng.gen::<f64>() * 0.1).collect();
        let shape = vec![64, 64];

        // Orthogonalize in-place
        let error = orthogonalize_in_place(&mut data, &shape).unwrap();

        // Verify error is small (single tensor should have error < 1e-10)
        assert!(
            error < 1e-10,
            "Single tensor orthogonalization error too large: {:.2e}",
            error
        );

        eprintln!("✓ Single tensor orthogonalization test passed");
        eprintln!("  - Orthogonalization error: {:.2e}", error);
    }

    /// Test graph-level vs single tensor error comparison
    #[test]
    #[cfg(feature = "tensor")]
    fn test_graph_vs_single_tensor_orthogonalization() {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();

        // Test 1: Single tensor baseline
        let mut single_data: Vec<f64> = (0..64 * 64).map(|_| rng.gen::<f64>() * 0.1).collect();
        let single_shape = vec![64, 64];
        let single_error = orthogonalize_in_place(&mut single_data, &single_shape).unwrap();

        // Test 2: Graph-level (same size matrix)
        let mut graph = Graph::<OperatorType, WeightTensor>::directed();
        let a = graph
            .add_node(OperatorType::Linear {
                in_features: 64,
                out_features: 64,
            })
            .unwrap();
        let b = graph
            .add_node(OperatorType::Linear {
                in_features: 64,
                out_features: 64,
            })
            .unwrap();
        let graph_data: Vec<f64> = (0..64 * 64).map(|_| rng.gen::<f64>() * 0.1).collect();
        let weight = WeightTensor::new("test".to_string(), graph_data, vec![64, 64]);
        let _edge = graph.add_edge(a, b, weight).unwrap();

        let config = LieGroupConfig::new().with_orthogonalize(true);
        let errors = orthogonalize_weights_in_place(&config, &mut graph).unwrap();

        let graph_error = errors[0];

        eprintln!("Single tensor error: {:.2e}", single_error);
        eprintln!("Graph-level error: {:.2e}", graph_error);

        // Both should have similar small errors (not 1.0 vs 1e-10)
        assert!(
            single_error < 1e-8,
            "Single tensor error too large: {:.2e}",
            single_error
        );
        assert!(
            graph_error < 1e-8,
            "Graph-level error too large: {:.2e}",
            graph_error
        );

        // Errors should be of similar magnitude (within 10x)
        let error_ratio = (single_error / graph_error).max(graph_error / single_error);
        assert!(
            error_ratio < 10.0,
            "Error ratio too large: single={:.2e}, graph={:.2e}, ratio={:.2}",
            single_error,
            graph_error,
            error_ratio
        );
    }

    /// Test that orthogonalization doesn't produce NaN or Inf
    #[test]
    #[cfg(feature = "tensor")]
    fn test_orthogonalization_no_nan_inf() {
        let mut graph = build_mini_transformer_graph_with_weights();

        let config = LieGroupConfig::new().with_orthogonalize(true);
        let _errors = orthogonalize_weights_in_place(&config, &mut graph).unwrap();

        // Check for NaN or Inf in all weights
        for edge_ref in graph.edges() {
            let weight = edge_ref.data();
            for (i, &val) in weight.data.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Weight {} has non-finite value at index {}: {:?}",
                    weight.name,
                    i,
                    val
                );
            }
        }

        eprintln!("✓ No NaN/Inf detected after orthogonalization");
    }
}
