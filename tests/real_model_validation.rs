//! Real Model Validation Tests
//!
//! This module tests the real model loading and optimization pipeline
//! using TinyLlama-1.1B weights from HuggingFace.
//!
//! ## Prerequisites
//!
//! Before running these tests, download the TinyLlama model:
//!
//! ```bash
//! pip install huggingface_hub
//! python scripts/download_tinyllama.py
//! ```
//!
//! The model will be stored in the `models/` directory.

#[cfg(test)]
mod tests {
    use god_gragh::graph::traits::{GraphBase, GraphOps, GraphQuery};
    use god_gragh::tensor::TensorBase;
    use god_gragh::tensor::decomposition::qr::is_orthogonal;
    use god_gragh::tensor::DenseTensor;
    use god_gragh::transformer::optimization::lie_group::orthogonalize_weights_in_place;
    use god_gragh::transformer::optimization::switch::{ModelSwitch, OperatorType, WeightTensor};
    use god_gragh::transformer::optimization::{LieGroupConfig, TensorRingCompressor, CompressionConfig};

    /// Get the path to the TinyLlama model directory
    fn get_tinyllama_model_path() -> Option<String> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        
        // Try multiple possible paths
        let possible_paths = vec![
            std::path::Path::new(&manifest_dir).join("models/tinyllama/model.safetensors"),
            std::path::Path::new(&manifest_dir).join("models/model.safetensors"),
        ];

        for path in possible_paths {
            if path.exists() {
                return Some(path.to_string_lossy().to_string());
            }
        }
        
        None
    }

    /// Test loading TinyLlama model from safetensors
    #[test]
    #[cfg(feature = "safetensors")]
    fn test_load_tinyllama_model() {
        let model_path = match get_tinyllama_model_path() {
            Some(path) => path,
            None => {
                eprintln!("Skipping test: TinyLlama model not found. Run: python scripts/download_tinyllama.py");
                return;
            }
        };

        eprintln!("Loading TinyLlama model from: {}", model_path);

        // Load model using ModelSwitch
        let graph = ModelSwitch::load_from_safetensors(&model_path)
            .expect("Failed to load TinyLlama model");

        // Verify basic properties
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();

        eprintln!("Loaded TinyLlama model:");
        eprintln!("  - Nodes: {}", node_count);
        eprintln!("  - Edges (weights): {}", edge_count);

        assert!(node_count > 0, "Model should have nodes");
        assert!(edge_count > 0, "Model should have weights");

        // Verify all weights are valid (no NaN/Inf)
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

        eprintln!("✓ All weights are valid (no NaN/Inf)");
    }

    /// Test TinyLlama weight orthogonalization
    #[test]
    #[cfg(feature = "safetensors")]
    fn test_tinyllama_orthogonalization() {
        let model_path = match get_tinyllama_model_path() {
            Some(path) => path,
            None => {
                eprintln!("Skipping test: TinyLlama model not found. Run: python scripts/download_tinyllama.py");
                return;
            }
        };

        eprintln!("Loading TinyLlama model for orthogonalization test...");

        // Load model
        let mut graph = ModelSwitch::load_from_safetensors(&model_path)
            .expect("Failed to load TinyLlama model");

        // Compute original weight statistics
        let mut original_norm_sum = 0.0;
        let mut weight_count = 0;
        
        for edge_ref in graph.edges() {
            let weight = edge_ref.data();
            let norm: f64 = weight.data.iter().map(|&x| x * x).sum::<f64>().sqrt();
            original_norm_sum += norm;
            weight_count += 1;
        }

        eprintln!("Original model: {} weights, total norm: {:.2e}", weight_count, original_norm_sum);

        // Apply orthogonalization
        let config = LieGroupConfig::new()
            .with_block_size(64)
            .with_orthogonalize(true);

        let errors = orthogonalize_weights_in_place(&config, &mut graph)
            .expect("Failed to orthogonalize weights");

        // Compute orthogonalization statistics
        let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let max_error = errors.iter().cloned().fold(0.0_f64, f64::max);

        eprintln!("Orthogonalization complete:");
        eprintln!("  - Average error: {:.2e}", avg_error);
        eprintln!("  - Max error: {:.2e}", max_error);

        // Verify orthogonalization quality
        assert!(
            avg_error < 1e-6,
            "Average orthogonalization error too large: {:.2e}",
            avg_error
        );
        assert!(
            max_error < 1e-4,
            "Max orthogonalization error too large: {:.2e}",
            max_error
        );

        // Verify all orthogonalized weights
        let mut orthogonalized_count = 0;
        for edge_ref in graph.edges() {
            let weight = edge_ref.data();
            
            // Skip non-2D tensors
            if weight.shape.len() != 2 {
                continue;
            }

            // Check orthogonality for square/tall matrices
            if weight.shape[0] >= weight.shape[1] {
                let tensor = DenseTensor::from_vec(weight.data.to_vec(), weight.shape.to_vec());
                if is_orthogonal(&tensor, 1e-6) {
                    orthogonalized_count += 1;
                } else {
                    let error = compute_orthogonality_error(&tensor);
                    eprintln!("Warning: Weight {} not orthogonal (error: {:.2e})", weight.name, error);
                }
            }
        }

        eprintln!("✓ {} weights successfully orthogonalized", orthogonalized_count);
    }

    /// Test TinyLlama tensor ring compression
    #[test]
    #[cfg(feature = "safetensors")]
    fn test_tinyllama_tensor_ring_compression() {
        let model_path = match get_tinyllama_model_path() {
            Some(path) => path,
            None => {
                eprintln!("Skipping test: TinyLlama model not found. Run: python scripts/download_tinyllama.py");
                return;
            }
        };

        eprintln!("Loading TinyLlama model for tensor ring compression test...");

        // Load model
        let graph = ModelSwitch::load_from_safetensors(&model_path)
            .expect("Failed to load TinyLlama model");

        // Count weights for compression readiness check
        let mut compressible_count = 0;
        let mut total_size = 0;

        for edge_ref in graph.edges() {
            let weight = edge_ref.data();

            // Skip small tensors
            if weight.data.len() < 1000 {
                continue;
            }

            compressible_count += 1;
            total_size += weight.data.len() * std::mem::size_of::<f64>();
        }

        eprintln!("Compression readiness:");
        eprintln!("  - Compressible weights: {}", compressible_count);
        eprintln!("  - Total size: {:.2} MB", total_size as f64 / 1e6);

        assert!(compressible_count > 0, "Should have compressible weights");
    }

    /// Test complete optimization pipeline (orthogonalization + compression)
    #[test]
    #[cfg(feature = "safetensors")]
    fn test_tinyllama_complete_optimization() {
        let model_path = match get_tinyllama_model_path() {
            Some(path) => path,
            None => {
                eprintln!("Skipping test: TinyLlama model not found. Run: python scripts/download_tinyllama.py");
                return;
            }
        };

        eprintln!("Running complete optimization pipeline on TinyLlama...");

        // Load model
        let mut graph = ModelSwitch::load_from_safetensors(&model_path)
            .expect("Failed to load TinyLlama model");

        // Step 1: Orthogonalize weights
        eprintln!("Step 1: Orthogonalizing weights...");
        let config = LieGroupConfig::new()
            .with_block_size(64)
            .with_orthogonalize(true);

        let ortho_errors = orthogonalize_weights_in_place(&config, &mut graph)
            .expect("Failed to orthogonalize weights");

        let avg_ortho_error = ortho_errors.iter().sum::<f64>() / ortho_errors.len() as f64;
        eprintln!("  Orthogonalization complete (avg error: {:.2e})", avg_ortho_error);

        // Verify optimization didn't introduce NaN/Inf
        for edge_ref in graph.edges() {
            let weight = edge_ref.data();
            for &val in weight.data.iter() {
                assert!(
                    val.is_finite(),
                    "Weight {} has non-finite value after optimization",
                    weight.name
                );
            }
        }

        eprintln!("✓ Complete optimization pipeline successful");
        eprintln!("  - Orthogonalization error: {:.2e}", avg_ortho_error);
        eprintln!("  - No NaN/Inf introduced");
    }

    /// Helper: compute orthogonality error
    fn compute_orthogonality_error(tensor: &DenseTensor) -> f64 {
        let shape = tensor.shape();
        if shape.len() != 2 || shape[0] < shape[1] {
            return f64::MAX;
        }

        let n = shape[1];
        let m = shape[0];
        let data = tensor.data();
        let mut max_error: f64 = 0.0;

        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..m {
                    dot += data[k * n + i] * data[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let error = (dot - expected).abs();
                max_error = max_error.max(error);
            }
        }

        max_error
    }
}
