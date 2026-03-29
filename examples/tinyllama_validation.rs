//! TinyLlama Real Model Validation Example
//!
//! This example demonstrates loading real TinyLlama-1.1B weights from HuggingFace,
//! applying orthogonalization, and validating output quality.
//!
//! ## Prerequisites
//!
//! Before running this example, download the TinyLlama model:
//!
//! ```bash
//! pip install huggingface_hub
//! huggingface-cli download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-5T \
//!     --include 'model.safetensors' --include 'config.json' \
//!     --local-dir models/tinyllama
//! ```
//!
//! ## Workflow
//!
//! 1. Load TinyLlama weights from Safetensors
//! 2. Validate weight integrity (no NaN/Inf)
//! 3. Apply Lie group orthogonalization
//! 4. Compute orthogonalization error metrics
//! 5. Generate optimization report
//!
//! ## Run
//!
//! ```bash
//! cargo run --example tinyllama_validation --features tensor,safetensors
//! ```

#[cfg(all(feature = "tensor", feature = "safetensors"))]
mod validation {
    use god_gragh::graph::traits::GraphQuery;
    use god_gragh::tensor::decomposition::qr::is_orthogonal;
    use god_gragh::tensor::{DenseTensor, TensorBase};
    use god_gragh::transformer::optimization::lie_group::orthogonalize_weights_in_place;
    use god_gragh::transformer::optimization::switch::{ModelSwitch, WeightTensor};
    use god_gragh::transformer::optimization::LieGroupConfig;
    use std::collections::HashMap;
    use std::path::Path;

    /// Format a number with thousand separators
    fn format_number(n: usize) -> String {
        let s = n.to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result.chars().rev().collect()
    }

    /// Get the path to the TinyLlama model directory
    fn get_tinyllama_model_path() -> Option<String> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

        // Try multiple possible paths
        let possible_paths = vec![
            Path::new(&manifest_dir).join("models/tinyllama/model.safetensors"),
            Path::new(&manifest_dir).join("models/model.safetensors"),
        ];

        for path in possible_paths {
            if path.exists() {
                return Some(path.to_string_lossy().to_string());
            }
        }

        None
    }

    /// Statistics for a single weight tensor
    #[derive(Debug, Clone)]
    struct WeightStats {
        name: String,
        shape: Vec<usize>,
        numel: usize,
        l2_norm: f64,
        min_val: f64,
        max_val: f64,
        mean: f64,
        std: f64,
    }

    /// Statistics for the entire model
    #[derive(Debug, Clone)]
    struct ModelStats {
        total_weights: usize,
        total_params: usize,
        weight_stats: Vec<WeightStats>,
        total_size_mb: f64,
    }

    /// Orthogonalization results
    #[derive(Debug, Clone)]
    struct OrthogonalizationReport {
        total_weights: usize,
        orthogonalized_weights: usize,
        skipped_weights: usize,
        avg_error: f64,
        max_error: f64,
        min_error: f64,
        errors: Vec<f64>,
    }

    /// Compute statistics for a weight tensor
    fn compute_weight_stats(weight: &WeightTensor) -> WeightStats {
        let data = &weight.data;
        let numel = data.len();

        let l2_norm: f64 = data.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = data.iter().sum::<f64>() / numel as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / numel as f64;
        let std = variance.sqrt();

        WeightStats {
            name: weight.name.clone(),
            shape: weight.shape.to_vec(),
            numel,
            l2_norm,
            min_val,
            max_val,
            mean,
            std,
        }
    }

    /// Compute model statistics
    fn compute_model_stats(
        graph: &god_gragh::graph::Graph<
            god_gragh::transformer::optimization::switch::OperatorType,
            WeightTensor,
        >,
    ) -> ModelStats {
        let mut weight_stats = Vec::new();
        let mut total_params = 0;

        for edge_ref in graph.edges() {
            let weight = edge_ref.data();
            let stats = compute_weight_stats(weight);
            total_params += stats.numel;
            weight_stats.push(stats);
        }

        let total_size_mb = (total_params * std::mem::size_of::<f64>()) as f64 / 1e6;

        ModelStats {
            total_weights: weight_stats.len(),
            total_params,
            total_size_mb,
            weight_stats,
        }
    }

    /// Validate all weights are finite (no NaN/Inf)
    fn validate_weights_finite(
        graph: &god_gragh::graph::Graph<
            god_gragh::transformer::optimization::switch::OperatorType,
            WeightTensor,
        >,
    ) -> Result<(), String> {
        for edge_ref in graph.edges() {
            let weight = edge_ref.data();
            for (i, &val) in weight.data.iter().enumerate() {
                if !val.is_finite() {
                    return Err(format!(
                        "Weight '{}' has non-finite value at index {}: {:?}",
                        weight.name, i, val
                    ));
                }
            }
        }
        Ok(())
    }

    /// Check orthogonality of a 2D weight tensor
    fn check_weight_orthogonality(weight: &WeightTensor, tolerance: f64) -> bool {
        // Only check 2D tensors
        if weight.shape.len() != 2 {
            return false;
        }

        let rows = weight.shape[0];
        let cols = weight.shape[1];

        // Only check square or tall matrices
        if rows < cols {
            return false;
        }

        let tensor = DenseTensor::from_vec(weight.data.to_vec(), weight.shape.to_vec());
        is_orthogonal(&tensor, tolerance)
    }

    /// Compute orthogonality error for a weight matrix
    fn compute_orthogonality_error(tensor: &DenseTensor) -> f64 {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return f64::MAX;
        }

        let n = shape[1]; // columns
        let m = shape[0]; // rows

        if m < n {
            return f64::MAX;
        }

        let data = tensor.data();
        let mut max_error: f64 = 0.0;

        // Compute W^T W and check if it equals I
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

    /// Print model statistics
    fn print_model_stats(stats: &ModelStats) {
        println!("\n{}", "=".repeat(70));
        println!("MODEL STATISTICS");
        println!("{}", "=".repeat(70));
        println!("Total weights:     {}", stats.total_weights);
        println!("Total parameters:  {}", format_number(stats.total_params));
        println!("Total size:        {:.2} MB", stats.total_size_mb);
        println!();

        // Group weights by layer type
        let mut by_type: HashMap<String, Vec<&WeightStats>> = HashMap::new();
        for ws in &stats.weight_stats {
            let layer_type = if ws.name.contains("embed") {
                "embedding"
            } else if ws.name.contains("attention") || ws.name.contains("attn") {
                "attention"
            } else if ws.name.contains("mlp") || ws.name.contains("ffn") {
                "mlp"
            } else if ws.name.contains("norm") || ws.name.contains("ln") {
                "normalization"
            } else {
                "other"
            };
            by_type.entry(layer_type.to_string()).or_default().push(ws);
        }

        println!("Weights by type:");
        for (layer_type, weights) in &by_type {
            let count = weights.len();
            let params: usize = weights.iter().map(|w| w.numel).sum();
            println!(
                "  {:20}: {:4} weights, {:>10} params",
                layer_type,
                count,
                format_number(params)
            );
        }
        println!();

        // Show top 10 largest weights
        println!("Top 10 largest weights:");
        let mut sorted_stats = stats.weight_stats.clone();
        sorted_stats.sort_by(|a, b| b.numel.cmp(&a.numel));
        for (i, ws) in sorted_stats.iter().take(10).enumerate() {
            println!(
                "  {:2}. {:50} {:>12} params, shape: {:?}",
                i + 1,
                ws.name,
                format_number(ws.numel),
                ws.shape
            );
        }
    }

    /// Print orthogonalization report
    fn print_orthogonalization_report(report: &OrthogonalizationReport) {
        println!("\n{}", "=".repeat(70));
        println!("ORTHOGONALIZATION REPORT");
        println!("{}", "=".repeat(70));
        println!("Total weights:        {}", report.total_weights);
        println!(
            "Orthogonalized:       {} ({:.1}%)",
            report.orthogonalized_weights,
            report.orthogonalized_weights as f64 / report.total_weights as f64 * 100.0
        );
        println!(
            "Skipped (non-2D):     {} ({:.1}%)",
            report.skipped_weights,
            report.skipped_weights as f64 / report.total_weights as f64 * 100.0
        );
        println!();
        println!("Error statistics:");
        println!("  Average error:      {:.2e}", report.avg_error);
        println!("  Max error:          {:.2e}", report.max_error);
        println!("  Min error:          {:.2e}", report.min_error);
        println!();

        // Quality assessment
        let quality = if report.avg_error < 1e-10 {
            "EXCELLENT (near machine precision)"
        } else if report.avg_error < 1e-8 {
            "VERY GOOD"
        } else if report.avg_error < 1e-6 {
            "GOOD"
        } else if report.avg_error < 1e-4 {
            "ACCEPTABLE"
        } else {
            "POOR (error too large)"
        };
        println!("Quality assessment:   {}", quality);
    }

    /// Print validation summary
    fn print_validation_summary(finite_check: bool, ortho_report: &OrthogonalizationReport) {
        println!("\n{}", "=".repeat(70));
        println!("VALIDATION SUMMARY");
        println!("{}", "=".repeat(70));

        let mut all_passed = true;

        // Finite check
        if finite_check {
            println!("✓ Weight finite check:    PASSED (no NaN/Inf)");
        } else {
            println!("✗ Weight finite check:    FAILED (NaN/Inf detected)");
            all_passed = false;
        }

        // Orthogonalization quality
        if ortho_report.avg_error < 1e-6 {
            println!(
                "✓ Orthogonalization:      PASSED (avg error: {:.2e})",
                ortho_report.avg_error
            );
        } else {
            println!(
                "✗ Orthogonalization:      FAILED (avg error: {:.2e})",
                ortho_report.avg_error
            );
            all_passed = false;
        }

        println!();
        if all_passed {
            println!("🎉 ALL VALIDATION CHECKS PASSED");
        } else {
            println!("❌ SOME VALIDATION CHECKS FAILED");
        }
        println!("{}", "=".repeat(70));
    }

    pub fn run() {
        println!("TinyLlama-1.1B Real Model Validation");
        println!("{}", "=".repeat(70));

        // Step 1: Load model
        println!("\nStep 1: Loading TinyLlama model from Safetensors...");
        let model_path = match get_tinyllama_model_path() {
            Some(path) => path,
            None => {
                eprintln!("ERROR: TinyLlama model not found.");
                eprintln!("Please download the model using:");
                eprintln!("  huggingface-cli download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-5T \\");
                eprintln!("      --include 'model.safetensors' --include 'config.json' \\");
                eprintln!("      --local-dir models/tinyllama");
                return;
            }
        };
        println!("  Model path: {}", model_path);

        let graph = match ModelSwitch::load_from_safetensors(&model_path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("ERROR: Failed to load model: {:?}", e);
                return;
            }
        };
        println!("  ✓ Model loaded successfully");

        // Step 2: Compute model statistics
        println!("\nStep 2: Computing model statistics...");
        let stats = compute_model_stats(&graph);
        print_model_stats(&stats);

        // Step 3: Validate weights (no NaN/Inf)
        println!("\nStep 3: Validating weight integrity...");
        let finite_check = validate_weights_finite(&graph).is_ok();
        if finite_check {
            println!("  ✓ All weights are finite (no NaN/Inf)");
        } else {
            println!("  ✗ Invalid weights detected");
        }

        // Step 4: Apply orthogonalization
        println!("\nStep 4: Applying Lie group orthogonalization...");
        let mut graph = graph; // Make mutable
        let config = LieGroupConfig::new()
            .with_block_size(64)
            .with_orthogonalize(true);

        let errors = match orthogonalize_weights_in_place(&config, &mut graph) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("ERROR: Orthogonalization failed: {:?}", e);
                return;
            }
        };
        println!("  ✓ Orthogonalization complete");

        // Step 5: Compute orthogonalization report
        println!("\nStep 5: Computing orthogonalization metrics...");
        let orthogonalized_count = errors.len();
        let skipped_count = stats.total_weights - orthogonalized_count;

        let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let max_error = errors.iter().cloned().fold(0.0_f64, f64::max);
        let min_error = errors.iter().cloned().fold(f64::MAX, f64::min);

        let ortho_report = OrthogonalizationReport {
            total_weights: stats.total_weights,
            orthogonalized_weights: orthogonalized_count,
            skipped_weights: skipped_count,
            avg_error,
            max_error,
            min_error,
            errors,
        };

        print_orthogonalization_report(&ortho_report);

        // Step 6: Verify orthogonalized weights
        println!("\nStep 6: Verifying orthogonalized weights...");
        let mut verified_count = 0;
        let tolerance = 1e-6;

        for edge_ref in graph.edges() {
            let weight = edge_ref.data();
            if check_weight_orthogonality(weight, tolerance) {
                verified_count += 1;
            }
        }
        println!(
            "  ✓ {} weights verified as orthogonal (tolerance: {:.0e})",
            verified_count, tolerance
        );

        // Step 7: Final validation summary
        print_validation_summary(finite_check, &ortho_report);

        println!("\nOptimization pipeline complete!");
    }
}

#[cfg(not(all(feature = "tensor", feature = "safetensors")))]
fn main() {
    println!("This example requires the 'tensor' and 'safetensors' features.");
    println!("Run with: cargo run --example tinyllama_validation --features tensor,safetensors");
}

#[cfg(all(feature = "tensor", feature = "safetensors"))]
fn main() {
    validation::run();
}
