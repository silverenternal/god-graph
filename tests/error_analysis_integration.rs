//! Error Accumulation Analysis Integration Tests
//!
//! This test suite validates the error accumulation analysis tool
//! by testing it with real graph-level tensor operations.

#[cfg(all(test, feature = "tensor"))]
mod tests {
    use god_gragh::graph::traits::{GraphOps, GraphQuery};
    use god_gragh::graph::Graph;
    use god_gragh::transformer::optimization::error_analysis::ErrorAccumulator;
    use god_gragh::transformer::optimization::switch::{OperatorType, WeightTensor};
    use god_gragh::transformer::optimization::{LieGroupConfig, LieGroupOptimizer};

    /// Test error accumulation during orthogonalization
    #[test]
    fn test_error_accumulation_during_orthogonalization() {
        // Create a simple graph with multiple weight tensors
        let mut graph = build_test_graph();

        // Create optimizer with orthogonalization enabled
        let config = LieGroupConfig::new()
            .with_orthogonalize(true)
            .with_iterations(1);
        let optimizer = LieGroupOptimizer::new(config);

        // Orthogonalize all weights (this records errors in the accumulator)
        optimizer.orthogonalize_weights(&mut graph).unwrap();

        // Check that errors were recorded
        let error_accumulator = optimizer.error_accumulator();
        assert!(
            error_accumulator.num_layers() > 0,
            "No errors were recorded"
        );
        assert!(
            error_accumulator.total_recordings() > 0,
            "No error recordings"
        );

        // Verify error statistics are reasonable (should be very small for orthogonalization)
        let stats = error_accumulator.compute_statistics();
        assert!(stats.mean < 1e-10, "Mean error too large: {}", stats.mean);
        assert!(stats.max < 1e-9, "Max error too large: {}", stats.max);

        println!("✓ Error accumulation test passed");
        println!("  - Layers tracked: {}", error_accumulator.num_layers());
        println!(
            "  - Total recordings: {}",
            error_accumulator.total_recordings()
        );
        println!("  - Mean error: {:.2e}", stats.mean);
        println!("  - Max error: {:.2e}", stats.max);
    }

    /// Test error report generation
    #[test]
    fn test_error_report_generation() {
        let mut graph = build_test_graph();

        let config = LieGroupConfig::new().with_orthogonalize(true);
        let optimizer = LieGroupOptimizer::new(config);

        optimizer.orthogonalize_weights(&mut graph).unwrap();

        // Generate error report
        let error_accumulator = optimizer.error_accumulator();
        let report = error_accumulator.generate_report();

        // Verify report structure
        assert_eq!(report.layer_stats.len(), error_accumulator.num_layers());
        assert!(report.global_stats.count > 0);

        // Print report for manual inspection
        println!("{}", report);
    }

    /// Test error tracking with manual recording
    #[test]
    fn test_manual_error_recording() {
        let mut accumulator = ErrorAccumulator::new();

        // Simulate errors from different layers
        accumulator.record_error("embeddings/token", 1.2e-14);
        accumulator.record_error("embeddings/position", 1.5e-14);
        accumulator.record_error("layer_0/q_proj", 2.1e-14);
        accumulator.record_error("layer_0/k_proj", 1.8e-14);
        accumulator.record_error("layer_0/v_proj", 2.3e-14);
        accumulator.record_error("layer_0/out_proj", 1.9e-14);
        accumulator.record_error("layer_1/q_proj", 3.1e-14);
        accumulator.record_error("layer_1/k_proj", 2.8e-14);
        accumulator.record_error("layer_1/v_proj", 3.5e-14);
        accumulator.record_error("layer_1/out_proj", 2.9e-14);
        accumulator.record_error("lm_head", 5.2e-14);

        // Verify statistics
        let stats = accumulator.compute_statistics();
        assert_eq!(stats.count, 11);
        assert_eq!(accumulator.num_layers(), 11);

        // Verify error report
        let report = accumulator.generate_report();
        assert_eq!(report.layer_stats.len(), 11);

        // lm_head should have the highest error
        assert_eq!(report.layer_stats[0].layer_name, "lm_head");
        assert!((report.layer_stats[0].max - 5.2e-14).abs() < 1e-20);

        println!("✓ Manual error recording test passed");
        println!("{}", report);
    }

    /// Test error accumulation with multiple errors per layer
    #[test]
    fn test_multiple_errors_per_layer() {
        let mut accumulator = ErrorAccumulator::new();

        // Record multiple errors for the same layer (simulating multiple weights)
        let q_proj_errors = vec![1.0e-14, 1.2e-14, 1.1e-14];
        accumulator.record_errors("layer_0/q_proj", q_proj_errors);

        let k_proj_errors = vec![1.5e-14, 1.3e-14, 1.4e-14];
        accumulator.record_errors("layer_0/k_proj", k_proj_errors);

        // Verify per-layer statistics
        let q_errors = accumulator.get_layer_errors("layer_0/q_proj").unwrap();
        assert_eq!(q_errors.len(), 3);

        let k_errors = accumulator.get_layer_errors("layer_0/k_proj").unwrap();
        assert_eq!(k_errors.len(), 3);

        // Verify global statistics
        let stats = accumulator.compute_statistics();
        assert_eq!(stats.count, 6);
        assert_eq!(accumulator.num_layers(), 2);

        println!("✓ Multiple errors per layer test passed");
        println!("  - Total recordings: {}", stats.count);
        println!("  - Mean error: {:.2e}", stats.mean);
    }

    /// Test error accumulator reset
    #[test]
    fn test_error_accumulator_reset() {
        let mut accumulator = ErrorAccumulator::new();

        // Record some errors
        accumulator.record_error("layer_0", 1.0e-14);
        accumulator.record_error("layer_1", 2.0e-14);

        assert_eq!(accumulator.num_layers(), 2);
        assert_eq!(accumulator.total_recordings(), 2);

        // Reset
        accumulator.reset();

        assert_eq!(accumulator.num_layers(), 0);
        assert_eq!(accumulator.total_recordings(), 0);
        assert_eq!(accumulator.total_error(), 0.0);

        println!("✓ Error accumulator reset test passed");
    }

    /// Test error bounds checking
    #[test]
    fn test_error_bounds() {
        let mut accumulator = ErrorAccumulator::new();

        // Record errors with known bounds
        accumulator.record_error("layer_0", 1.0e-14);
        accumulator.record_error("layer_1", 5.0e-14);
        accumulator.record_error("layer_2", 2.0e-14);

        let stats = accumulator.compute_statistics();

        // Verify min/max
        assert!((stats.min - 1.0e-14).abs() < 1e-20);
        assert!((stats.max - 5.0e-14).abs() < 1e-20);

        // Verify total
        let expected_total = 8.0e-14;
        assert!((stats.total - expected_total).abs() < 1e-20);

        println!("✓ Error bounds test passed");
        println!("  - Min error: {:.2e}", stats.min);
        println!("  - Max error: {:.2e}", stats.max);
        println!("  - Total error: {:.2e}", stats.total);
    }

    /// Test error accumulation with in-place orthogonalization
    #[test]
    fn test_in_place_orthogonalization_error_tracking() {
        let mut graph = build_test_graph();

        let config = LieGroupConfig::new()
            .with_orthogonalize(true)
            .with_iterations(1);
        let optimizer = LieGroupOptimizer::new(config);

        // Get edge indices
        let edge_indices: Vec<_> = graph.edges().map(|e| e.index()).collect();

        // Orthogonalize each weight individually and track errors
        let mut manual_errors = Vec::new();
        for edge_idx in edge_indices {
            let error = optimizer
                .orthogonalize_single_weight(&mut graph, edge_idx)
                .unwrap();
            manual_errors.push(error);
        }

        // Verify errors are small (orthogonalization should be numerically stable)
        for (i, &error) in manual_errors.iter().enumerate() {
            assert!(error < 1e-10, "Error {} too large: {:.2e}", i, error);
        }

        println!("✓ In-place orthogonalization test passed");
        println!("  - Weights orthogonalized: {}", manual_errors.len());
        println!(
            "  - Max error: {:.2e}",
            manual_errors.iter().cloned().fold(0.0_f64, f64::max)
        );
    }

    /// Build a test graph with multiple weight tensors
    fn build_test_graph() -> Graph<OperatorType, WeightTensor> {
        let mut graph = Graph::directed();

        // Add embedding layer
        let embed_node = graph
            .add_node(OperatorType::Embedding {
                vocab_size: 100,
                embed_dim: 64,
            })
            .unwrap();

        // Add attention layers
        let mut prev_node = embed_node;
        for layer in 0..2 {
            // Q projection
            let q_proj = graph
                .add_node(OperatorType::Linear {
                    in_features: 64,
                    out_features: 64,
                })
                .unwrap();
            graph
                .add_edge(
                    prev_node,
                    q_proj,
                    create_weight_tensor(64, 64, &format!("layer_{}/q_proj", layer)),
                )
                .unwrap();

            // K projection
            let k_proj = graph
                .add_node(OperatorType::Linear {
                    in_features: 64,
                    out_features: 64,
                })
                .unwrap();
            graph
                .add_edge(
                    q_proj,
                    k_proj,
                    create_weight_tensor(64, 64, &format!("layer_{}/k_proj", layer)),
                )
                .unwrap();

            // V projection
            let v_proj = graph
                .add_node(OperatorType::Linear {
                    in_features: 64,
                    out_features: 64,
                })
                .unwrap();
            graph
                .add_edge(
                    k_proj,
                    v_proj,
                    create_weight_tensor(64, 64, &format!("layer_{}/v_proj", layer)),
                )
                .unwrap();

            // Output projection
            let out_proj = graph
                .add_node(OperatorType::Linear {
                    in_features: 64,
                    out_features: 64,
                })
                .unwrap();
            graph
                .add_edge(
                    v_proj,
                    out_proj,
                    create_weight_tensor(64, 64, &format!("layer_{}/out_proj", layer)),
                )
                .unwrap();

            prev_node = out_proj;
        }

        // Add LM head
        let lm_head = graph
            .add_node(OperatorType::Linear {
                in_features: 64,
                out_features: 100,
            })
            .unwrap();
        graph
            .add_edge(prev_node, lm_head, create_weight_tensor(100, 64, "lm_head"))
            .unwrap();

        graph
    }

    /// Create a weight tensor with random data
    fn create_weight_tensor(rows: usize, cols: usize, name: &str) -> WeightTensor {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();

        WeightTensor::new(name.to_string(), data, vec![rows, cols])
    }
}
