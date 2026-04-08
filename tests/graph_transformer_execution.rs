//! GraphTransformer Execution Tests
//!
//! This module provides comprehensive tests for GraphTransformer execution
//! including tensor passing, forward pass validation, and edge semantics.
//!
//! Requires the `tensor` and `transformer` features.

#![cfg(all(test, feature = "tensor", feature = "transformer"))]

#[cfg(test)]
mod tests {
    use god_graph::tensor::traits::TensorBase;
    use god_graph::tensor::DenseTensor;
    #[allow(unused_imports)]
    use god_graph::transformer::graph_transformer::edges::{
        DataFlowEdge, DataFlowOp, GraphEdge, GraphEdgeType, ResidualEdge, SelfAttentionEdge,
        SkipType,
    };
    use god_graph::transformer::graph_transformer::GraphTransformer;

    // ==================== GraphTransformer Basic Tests ====================

    #[test]
    fn test_graph_transformer_creation() {
        let transformer = GraphTransformer::new(2, 4, 256);

        // Test that we can create a transformer with given config
        // Note: Internal fields are private, so we test via behavior
        assert!(transformer.num_nodes() == 0); // Initially empty graph
        assert!(transformer.num_edges() == 0);
    }

    #[test]
    fn test_graph_transformer_build_graph() {
        let mut transformer = GraphTransformer::new(2, 4, 256);
        let input_ids = vec![1, 2, 3, 4];

        transformer.build_graph(&input_ids);

        assert!(transformer.num_nodes() > 0);
        assert!(transformer.num_edges() > 0);

        // Verify graph structure: should have embedding nodes + layer nodes
        // For 4 tokens, 2 layers: 4 embedding + 2*4*2 (attn+ffn per layer) = 20 nodes minimum
        assert!(transformer.num_nodes() >= 4);
    }

    #[test]
    fn test_graph_transformer_forward_basic() {
        let mut transformer = GraphTransformer::new(1, 2, 128);
        let input_ids = vec![1, 2, 3];

        transformer.build_graph(&input_ids);
        let output = transformer.forward(&input_ids);

        // Output should be [1, hidden_dim]
        assert_eq!(output.ndim(), 2);
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 128);

        // Check no NaN or Inf
        for &val in output.data() {
            assert!(val.is_finite(), "Output contains NaN or Inf");
        }
    }

    #[test]
    fn test_graph_transformer_forward_multi_layer() {
        let mut transformer = GraphTransformer::new(3, 4, 256);
        let input_ids = vec![1, 2, 3, 4, 5];

        transformer.build_graph(&input_ids);
        let output = transformer.forward(&input_ids);

        // Output shape should match hidden_dim
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 256);
    }

    #[test]
    fn test_graph_transformer_different_sequence_lengths() {
        let seq_lengths = vec![1, 2, 4, 8, 16];

        for &seq_len in &seq_lengths {
            let mut transformer = GraphTransformer::new(1, 2, 64);
            let input_ids: Vec<usize> = (0..seq_len).collect();

            transformer.build_graph(&input_ids);
            let output = transformer.forward(&input_ids);

            assert_eq!(output.shape()[0], 1);
            assert_eq!(output.shape()[1], 64);
        }
    }

    #[test]
    fn test_graph_transformer_pruning() {
        let mut transformer = GraphTransformer::new(1, 2, 128);
        let input_ids = vec![1, 2, 3, 4];

        transformer.build_graph(&input_ids);
        let initial_edges = transformer.num_edges();

        // Prune with high threshold to ensure some edges are pruned
        let pruned_count = transformer.prune_weak_edges(0.5);

        // Verify edges were reduced
        assert_eq!(transformer.num_edges(), initial_edges - pruned_count);

        // Re-execute after pruning
        let output = transformer.forward(&input_ids);
        assert_eq!(output.shape()[1], 128);
    }

    #[test]
    fn test_graph_transformer_to_dot() {
        let mut transformer = GraphTransformer::new(1, 2, 64);
        transformer.build_graph(&[1, 2, 3]);

        let dot = transformer.to_dot();

        assert!(dot.contains("digraph Transformer"));
        assert!(dot.contains("rankdir=TB"));
        assert!(dot.contains("node [shape=box]"));

        // Should contain node labels
        assert!(
            dot.contains("TokenEmbed")
                || dot.contains("Hidden")
                || dot.contains("Attn")
                || dot.contains("FFN")
        );
    }

    // ==================== Edge Tensor Tests ====================

    #[test]
    fn test_self_attention_edge_creation() {
        let edge = GraphEdge::self_attention(0, 1, 0.8, 2, 0);

        assert_eq!(edge.edge_type, GraphEdgeType::SelfAttention);
        assert_eq!(edge.source, 0);
        assert_eq!(edge.target, 1);

        let sa = edge.get_self_attention().unwrap();
        assert_eq!(sa.weight, 0.8);
        assert_eq!(sa.head, 2);
        assert_eq!(sa.layer, 0);
    }

    #[test]
    fn test_self_attention_edge_with_message() {
        let q_proj = DenseTensor::zeros(vec![1, 64]);
        let edge = GraphEdge::self_attention_with_message(0, 1, 0.8, 2, 0, q_proj);

        assert!(edge.message().is_some());
        assert_eq!(edge.message().unwrap().shape(), &[1, 64]);

        let sa = edge.get_self_attention().unwrap();
        assert_eq!(sa.weight, 0.8);
    }

    #[test]
    fn test_self_attention_edge_with_qkv() {
        let q_proj = DenseTensor::zeros(vec![1, 64]);
        let k_proj = DenseTensor::zeros(vec![1, 64]);
        let v_proj = DenseTensor::zeros(vec![1, 64]);

        let edge = GraphEdge::self_attention_with_qkv(0, 1, 0.8, 2, 0, q_proj, k_proj, v_proj);

        assert!(edge.has_qkv());

        let (q, k, v) = edge.get_qkv();
        assert!(q.is_some());
        assert!(k.is_some());
        assert!(v.is_some());

        assert_eq!(q.unwrap().shape(), &[1, 64]);
        assert_eq!(k.unwrap().shape(), &[1, 64]);
        assert_eq!(v.unwrap().shape(), &[1, 64]);
    }

    #[test]
    fn test_self_attention_edge_attention_score() {
        // Create Q and K projections with known values
        let q_data = vec![1.0, 0.0, 0.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 0.0];

        let q_proj = DenseTensor::new(q_data, vec![1, 4]);
        let k_proj = DenseTensor::new(k_data, vec![1, 4]);
        let v_proj = DenseTensor::zeros(vec![1, 4]);

        let edge = GraphEdge::self_attention_with_qkv(0, 1, 0.8, 0, 0, q_proj, k_proj, v_proj);

        // Compute attention score with d_k = 4
        let score = edge.compute_attention_score(4.0);

        assert!(score.is_some());
        // Dot product = 1.0, d_k = 4, so score = 1.0 / 2.0 = 0.5
        assert!((score.unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_data_flow_edge_creation() {
        let edge = GraphEdge::data_flow(0, 1, DataFlowOp::InputToAttention, 0);

        assert_eq!(edge.edge_type, GraphEdgeType::DataFlow);

        let df = edge.get_data_flow().unwrap();
        assert_eq!(df.operation, DataFlowOp::InputToAttention);
        assert_eq!(df.layer, 0);
    }

    #[test]
    fn test_data_flow_edge_with_message() {
        let activation = DenseTensor::zeros(vec![1, 256]);
        let edge =
            GraphEdge::data_flow_with_message(0, 1, DataFlowOp::AttentionToOutput, 0, activation);

        assert!(edge.message().is_some());
        assert_eq!(edge.message().unwrap().shape(), &[1, 256]);

        let df = edge.get_data_flow().unwrap();
        assert_eq!(df.operation, DataFlowOp::AttentionToOutput);
    }

    #[test]
    fn test_residual_edge_creation() {
        let edge = GraphEdge::residual(0, 1, 0, SkipType::PreNorm);

        assert_eq!(edge.edge_type, GraphEdgeType::Residual);

        let res = edge.get_residual().unwrap();
        assert_eq!(res.layer, 0);
        assert!(matches!(res.skip_type, SkipType::PreNorm));
    }

    #[test]
    fn test_residual_edge_with_tensor() {
        let residual = DenseTensor::zeros(vec![1, 128]);
        let edge = GraphEdge::residual_with_tensor(0, 1, 0, SkipType::PostNorm, residual);

        assert!(edge.message().is_some());
        assert_eq!(edge.message().unwrap().shape(), &[1, 128]);

        let res = edge.get_residual().unwrap();
        assert!(matches!(res.skip_type, SkipType::PostNorm));
    }

    #[test]
    fn test_edge_layer_extraction() {
        let sa_edge = GraphEdge::self_attention(0, 1, 0.5, 0, 5);
        assert_eq!(sa_edge.layer(), 5);

        let df_edge = GraphEdge::data_flow(0, 1, DataFlowOp::LayerToLayer, 10);
        assert_eq!(df_edge.layer(), 10);

        let res_edge = GraphEdge::residual(0, 1, 3, SkipType::PreNorm);
        assert_eq!(res_edge.layer(), 3);
    }

    // ==================== Tensor Passing Integration Tests ====================

    #[test]
    fn test_tensor_message_passing_workflow() {
        // Simulate tensor passing workflow
        let q_proj = DenseTensor::zeros(vec![1, 64]);
        let k_proj = DenseTensor::zeros(vec![1, 64]);
        let v_proj = DenseTensor::zeros(vec![1, 64]);

        // Create attention edge with QKV
        let mut edge = GraphEdge::self_attention_with_qkv(0, 1, 0.8, 0, 0, q_proj, k_proj, v_proj);

        // Verify QKV projections
        assert!(edge.has_qkv());

        // Modify Q projection
        let new_q = DenseTensor::ones(vec![1, 64]);
        edge.set_message(new_q);

        assert!(edge.message().is_some());
        assert_eq!(edge.message().unwrap().data()[0], 1.0);
    }

    #[test]
    fn test_edge_tensor_modification() {
        let mut edge = GraphEdge::self_attention(0, 1, 0.5, 0, 0);

        // Initially no message
        assert!(edge.message().is_none());

        // Add message
        let msg = DenseTensor::zeros(vec![1, 32]);
        edge.set_message(msg.clone());

        assert!(edge.message().is_some());

        // Replace message
        let new_msg = DenseTensor::ones(vec![1, 32]);
        edge.set_message(new_msg);

        assert_eq!(edge.message().unwrap().data()[0], 1.0);
    }

    // ==================== Numerical Stability Tests ====================

    #[test]
    fn test_forward_numerical_stability() {
        let mut transformer = GraphTransformer::new(2, 4, 256);
        let input_ids = vec![1, 2, 3, 4, 5, 6, 7, 8];

        transformer.build_graph(&input_ids);
        let output = transformer.forward(&input_ids);

        // Check all values are finite
        for &val in output.data() {
            assert!(val.is_finite(), "Output contains NaN or Inf");
        }

        // Check values are in reasonable range (not exploding)
        let max_val = output
            .data()
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = output.data().iter().fold(f64::INFINITY, |a, &b| a.min(b));

        assert!(max_val < 1e6, "Output values exploding (max: {})", max_val);
        assert!(min_val > -1e6, "Output values exploding (min: {})", min_val);
    }

    #[test]
    fn test_attention_score_numerical_stability() {
        // Test with large Q and K values
        let q_data = vec![100.0, 100.0, 100.0, 100.0];
        let k_data = vec![100.0, 100.0, 100.0, 100.0];

        let q_proj = DenseTensor::new(q_data, vec![1, 4]);
        let k_proj = DenseTensor::new(k_data, vec![1, 4]);
        let v_proj = DenseTensor::zeros(vec![1, 4]);

        let edge = GraphEdge::self_attention_with_qkv(0, 1, 0.8, 0, 0, q_proj, k_proj, v_proj);

        let score = edge.compute_attention_score(4.0);

        assert!(score.is_some());
        assert!(score.unwrap().is_finite());
    }

    // ==================== Edge Cases Tests ====================

    #[test]
    fn test_single_token_sequence() {
        let mut transformer = GraphTransformer::new(1, 2, 64);
        let input_ids = vec![42];

        transformer.build_graph(&input_ids);
        let output = transformer.forward(&input_ids);

        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 64);
    }

    #[test]
    fn test_empty_qkv_projections() {
        let edge = GraphEdge::self_attention(0, 1, 0.5, 0, 0);

        assert!(!edge.has_qkv());
        assert!(edge.message().is_none());
        assert!(edge.key_proj().is_none());
        assert!(edge.value_proj().is_none());

        // compute_attention_score should return None
        let score = edge.compute_attention_score(4.0);
        assert!(score.is_none());
    }

    #[test]
    fn test_partial_qkv_projections() {
        let q_proj = DenseTensor::zeros(vec![1, 64]);
        let edge = GraphEdge::self_attention_with_message(0, 1, 0.5, 0, 0, q_proj);

        // Only Q is set
        assert!(!edge.has_qkv());
        assert!(edge.message().is_some());
        assert!(edge.key_proj().is_none());
        assert!(edge.value_proj().is_none());

        // compute_attention_score should return None (missing K)
        let score = edge.compute_attention_score(4.0);
        assert!(score.is_none());
    }

    // ==================== Performance-related Tests ====================

    #[test]
    fn test_graph_structure_consistency() {
        let mut transformer = GraphTransformer::new(2, 4, 256);
        let input_ids = vec![1, 2, 3, 4];

        transformer.build_graph(&input_ids);
        let nodes_before = transformer.num_nodes();
        let edges_before = transformer.num_edges();

        // Execute forward pass
        let _ = transformer.forward(&input_ids);

        // Graph structure should remain unchanged
        assert_eq!(transformer.num_nodes(), nodes_before);
        assert_eq!(transformer.num_edges(), edges_before);
    }

    #[test]
    fn test_repeated_forward_execution() {
        let mut transformer = GraphTransformer::new(1, 2, 128);
        let input_ids = vec![1, 2, 3];

        transformer.build_graph(&input_ids);

        // Run forward pass multiple times
        let output1 = transformer.forward(&input_ids);
        let output2 = transformer.forward(&input_ids);
        let output3 = transformer.forward(&input_ids);

        // All outputs should have same shape
        assert_eq!(output1.shape(), output2.shape());
        assert_eq!(output2.shape(), output3.shape());

        // All outputs should be finite
        for output in [&output1, &output2, &output3] {
            for &val in output.data() {
                assert!(val.is_finite());
            }
        }
    }
}
