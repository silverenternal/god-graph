//! Error Accumulation Analysis for Graph-Level Tensor Operations
//!
//! This module provides tools for tracking and analyzing numerical errors
//! that accumulate during graph-level tensor operations like orthogonalization
//! and decomposition.
//!
//! ## Purpose
//!
//! When performing operations like QR decomposition on weights across a graph,
//! numerical errors can accumulate. This module helps:
//!
//! - Track per-layer error contributions
//! - Generate statistical reports on error distributions
//! - Identify problematic layers with high error
//!
//! ## Example
//!
//! ```no_run
//! use god_gragh::transformer::optimization::error_analysis::ErrorAccumulator;
//!
//! let mut accumulator = ErrorAccumulator::new();
//!
//! // Record errors from each layer
//! accumulator.record_error("layer_0/q_proj", 1.2e-14);
//! accumulator.record_error("layer_0/k_proj", 1.5e-14);
//! accumulator.record_error("layer_1/q_proj", 2.1e-14);
//!
//! // Generate and print report
//! let report = accumulator.generate_report();
//! println!("{}", report);
//! ```

use std::collections::HashMap;
use std::fmt;

/// Error accumulator for tracking numerical errors across graph operations
#[derive(Debug, Clone, Default)]
pub struct ErrorAccumulator {
    /// Errors grouped by layer name
    layer_errors: HashMap<String, Vec<f64>>,
    /// Total accumulated error
    total_error: f64,
    /// Minimum error observed
    min_error: f64,
    /// Maximum error observed
    max_error: f64,
}

impl ErrorAccumulator {
    /// Create a new error accumulator
    pub fn new() -> Self {
        Self {
            layer_errors: HashMap::new(),
            total_error: 0.0,
            min_error: f64::INFINITY,
            max_error: f64::NEG_INFINITY,
        }
    }

    /// Record an error for a specific layer
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name/identifier of the layer
    /// * `error` - Numerical error value (should be positive)
    pub fn record_error(&mut self, layer_name: &str, error: f64) {
        self.layer_errors
            .entry(layer_name.to_string())
            .or_insert_with(Vec::new)
            .push(error);

        self.total_error += error;
        if error < self.min_error {
            self.min_error = error;
        }
        if error > self.max_error {
            self.max_error = error;
        }
    }

    /// Record multiple errors for a layer at once
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name/identifier of the layer
    /// * `errors` - Iterator of error values
    pub fn record_errors<I>(&mut self, layer_name: &str, errors: I)
    where
        I: IntoIterator<Item = f64>,
    {
        let layer_vec = self
            .layer_errors
            .entry(layer_name.to_string())
            .or_insert_with(Vec::new);

        for error in errors {
            layer_vec.push(error);
            self.total_error += error;
            if error < self.min_error {
                self.min_error = error;
            }
            if error > self.max_error {
                self.max_error = error;
            }
        }
    }

    /// Get the total accumulated error
    pub fn total_error(&self) -> f64 {
        self.total_error
    }

    /// Get the minimum error observed
    pub fn min_error(&self) -> f64 {
        self.min_error
    }

    /// Get the maximum error observed
    pub fn max_error(&self) -> f64 {
        self.max_error
    }

    /// Get the number of layers tracked
    pub fn num_layers(&self) -> usize {
        self.layer_errors.len()
    }

    /// Get the total number of error recordings
    pub fn total_recordings(&self) -> usize {
        self.layer_errors.values().map(|v| v.len()).sum()
    }

    /// Get errors for a specific layer
    pub fn get_layer_errors(&self, layer_name: &str) -> Option<&[f64]> {
        self.layer_errors.get(layer_name).map(|v| v.as_slice())
    }

    /// Compute global error statistics
    pub fn compute_statistics(&self) -> ErrorStatistics {
        let all_errors: Vec<f64> = self.layer_errors.values().flatten().copied().collect();

        if all_errors.is_empty() {
            return ErrorStatistics {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                total: 0.0,
                count: 0,
            };
        }

        let count = all_errors.len();
        let total = all_errors.iter().sum::<f64>();
        let mean = total / count as f64;

        // Compute standard deviation
        let variance = all_errors
            .iter()
            .map(|&e| (e - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();

        let min = all_errors
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = all_errors
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        ErrorStatistics {
            mean,
            std_dev,
            min,
            max,
            total,
            count,
        }
    }

    /// Generate a detailed error report
    pub fn generate_report(&self) -> ErrorReport {
        let global_stats = self.compute_statistics();

        // Compute per-layer statistics
        let mut layer_stats: Vec<LayerErrorStats> = self
            .layer_errors
            .iter()
            .map(|(layer_name, errors)| {
                let count = errors.len();
                let total = errors.iter().sum::<f64>();
                let mean = total / count as f64;
                let max = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min = errors.iter().cloned().fold(f64::INFINITY, f64::min);

                // Compute layer standard deviation
                let variance = errors
                    .iter()
                    .map(|&e| (e - mean).powi(2))
                    .sum::<f64>()
                    / count as f64;
                let std_dev = variance.sqrt();

                LayerErrorStats {
                    layer_name: layer_name.clone(),
                    mean,
                    std_dev,
                    min,
                    max,
                    count,
                }
            })
            .collect();

        // Sort by max error (descending) to highlight problematic layers
        layer_stats.sort_by(|a, b| b.max.partial_cmp(&a.max).unwrap_or(std::cmp::Ordering::Equal));

        ErrorReport {
            global_stats,
            layer_stats,
        }
    }

    /// Reset the accumulator (clear all recorded errors)
    pub fn reset(&mut self) {
        self.layer_errors.clear();
        self.total_error = 0.0;
        self.min_error = f64::INFINITY;
        self.max_error = f64::NEG_INFINITY;
    }
}

/// Global error statistics
#[derive(Debug, Clone)]
pub struct ErrorStatistics {
    /// Mean error across all recordings
    pub mean: f64,
    /// Standard deviation of errors
    pub std_dev: f64,
    /// Minimum error observed
    pub min: f64,
    /// Maximum error observed
    pub max: f64,
    /// Total accumulated error
    pub total: f64,
    /// Number of error recordings
    pub count: usize,
}

impl fmt::Display for ErrorStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Error Statistics:")?;
        writeln!(f, "  Count:   {}", self.count)?;
        writeln!(f, "  Mean:    {:.2e}", self.mean)?;
        writeln!(f, "  Std Dev: {:.2e}", self.std_dev)?;
        writeln!(f, "  Min:     {:.2e}", self.min)?;
        writeln!(f, "  Max:     {:.2e}", self.max)?;
        writeln!(f, "  Total:   {:.2e}", self.total)
    }
}

/// Per-layer error statistics
#[derive(Debug, Clone)]
pub struct LayerErrorStats {
    /// Layer name
    pub layer_name: String,
    /// Mean error for this layer
    pub mean: f64,
    /// Standard deviation for this layer
    pub std_dev: f64,
    /// Minimum error for this layer
    pub min: f64,
    /// Maximum error for this layer
    pub max: f64,
    /// Number of recordings for this layer
    pub count: usize,
}

impl fmt::Display for LayerErrorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  {}:", self.layer_name)?;
        writeln!(f, "    Count:   {}", self.count)?;
        writeln!(f, "    Mean:    {:.2e}", self.mean)?;
        writeln!(f, "    Std Dev: {:.2e}", self.std_dev)?;
        writeln!(f, "    Min:     {:.2e}", self.min)?;
        writeln!(f, "    Max:     {:.2e}", self.max)
    }
}

/// Comprehensive error report
#[derive(Debug, Clone)]
pub struct ErrorReport {
    /// Global statistics across all layers
    pub global_stats: ErrorStatistics,
    /// Per-layer statistics (sorted by max error, descending)
    pub layer_stats: Vec<LayerErrorStats>,
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        writeln!(f, "╔══════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           ERROR ACCUMULATION REPORT                      ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ GLOBAL STATISTICS                                        ║")?;
        writeln!(f, "╟──────────────────────────────────────────────────────────╢")?;
        writeln!(f, "║   Total Recordings: {:>20} ║", self.global_stats.count)?;
        writeln!(f, "║   Mean Error:       {:>20.2e} ║", self.global_stats.mean)?;
        writeln!(f, "║   Std Dev:          {:>20.2e} ║", self.global_stats.std_dev)?;
        writeln!(f, "║   Min Error:        {:>20.2e} ║", self.global_stats.min)?;
        writeln!(f, "║   Max Error:        {:>20.2e} ║", self.global_stats.max)?;
        writeln!(f, "║   Total Error:      {:>20.2e} ║", self.global_stats.total)?;
        writeln!(f, "╟──────────────────────────────────────────────────────────╢")?;
        writeln!(f, "║ PER-LAYER STATISTICS (sorted by max error)             ║")?;
        writeln!(f, "╟──────────────────────────────────────────────────────────╢")?;

        // Show top 10 layers by max error
        let display_count = self.layer_stats.len().min(10);
        for (i, layer) in self.layer_stats.iter().take(display_count).enumerate() {
            writeln!(
                f,
                "║   {:2}. {:<28} {:.2e} ║",
                i + 1,
                truncate_name(&layer.layer_name, 28),
                layer.max
            )?;
        }

        if self.layer_stats.len() > display_count {
            writeln!(
                f,
                "║   ... and {} more layers                              ║",
                self.layer_stats.len() - display_count
            )?;
        }

        writeln!(f, "╚══════════════════════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Truncate a string to a maximum length, adding ellipsis if needed
fn truncate_name(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max_len + 3..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_accumulator_basic() {
        let mut accumulator = ErrorAccumulator::new();

        accumulator.record_error("layer_0", 1.0e-14);
        accumulator.record_error("layer_0", 1.5e-14);
        accumulator.record_error("layer_1", 2.0e-14);

        assert_eq!(accumulator.num_layers(), 2);
        assert_eq!(accumulator.total_recordings(), 3);
        assert!((accumulator.total_error() - 4.5e-14).abs() < 1e-20);
    }

    #[test]
    fn test_error_accumulator_statistics() {
        let mut accumulator = ErrorAccumulator::new();

        // Record known errors
        let errors = vec![1.0e-14, 2.0e-14, 3.0e-14, 4.0e-14, 5.0e-14];
        for (i, &error) in errors.iter().enumerate() {
            accumulator.record_error(&format!("layer_{}", i), error);
        }

        let stats = accumulator.compute_statistics();

        // Mean should be (1+2+3+4+5)/5 = 3.0e-14
        assert!((stats.mean - 3.0e-14).abs() < 1e-20);
        // Min should be 1.0e-14
        assert!((stats.min - 1.0e-14).abs() < 1e-20);
        // Max should be 5.0e-14
        assert!((stats.max - 5.0e-14).abs() < 1e-20);
        // Total should be 15.0e-14
        assert!((stats.total - 15.0e-14).abs() < 1e-20);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_error_accumulator_multiple_errors_per_layer() {
        let mut accumulator = ErrorAccumulator::new();

        // Record multiple errors for the same layer
        accumulator.record_error("layer_0", 1.0e-14);
        accumulator.record_error("layer_0", 2.0e-14);
        accumulator.record_error("layer_0", 3.0e-14);

        let layer_errors = accumulator.get_layer_errors("layer_0").unwrap();
        assert_eq!(layer_errors.len(), 3);
        assert!((layer_errors[0] - 1.0e-14).abs() < 1e-20);
        assert!((layer_errors[1] - 2.0e-14).abs() < 1e-20);
        assert!((layer_errors[2] - 3.0e-14).abs() < 1e-20);
    }

    #[test]
    fn test_error_accumulator_record_multiple() {
        let mut accumulator = ErrorAccumulator::new();

        let errors = vec![1.0e-14, 2.0e-14, 3.0e-14];
        accumulator.record_errors("layer_0", errors);

        assert_eq!(accumulator.total_recordings(), 3);
        let layer_errors = accumulator.get_layer_errors("layer_0").unwrap();
        assert_eq!(layer_errors.len(), 3);
    }

    #[test]
    fn test_error_accumulator_empty() {
        let accumulator = ErrorAccumulator::new();

        let stats = accumulator.compute_statistics();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.total, 0.0);
    }

    #[test]
    fn test_error_accumulator_reset() {
        let mut accumulator = ErrorAccumulator::new();
        accumulator.record_error("layer_0", 1.0e-14);
        accumulator.record_error("layer_1", 2.0e-14);

        assert_eq!(accumulator.num_layers(), 2);
        assert_eq!(accumulator.total_recordings(), 2);

        accumulator.reset();

        assert_eq!(accumulator.num_layers(), 0);
        assert_eq!(accumulator.total_recordings(), 0);
        assert_eq!(accumulator.total_error(), 0.0);
    }

    #[test]
    fn test_error_report_generation() {
        let mut accumulator = ErrorAccumulator::new();

        // Simulate errors from a small transformer graph
        accumulator.record_error("embeddings", 5.0e-15);
        accumulator.record_error("layer_0/q_proj", 1.2e-14);
        accumulator.record_error("layer_0/k_proj", 1.5e-14);
        accumulator.record_error("layer_0/v_proj", 1.1e-14);
        accumulator.record_error("layer_0/out_proj", 1.3e-14);
        accumulator.record_error("layer_1/q_proj", 2.1e-14);
        accumulator.record_error("layer_1/k_proj", 1.8e-14);
        accumulator.record_error("layer_1/v_proj", 2.3e-14);
        accumulator.record_error("layer_1/out_proj", 1.9e-14);
        accumulator.record_error("lm_head", 3.5e-14);

        let report = accumulator.generate_report();

        // Verify global stats
        assert_eq!(report.global_stats.count, 10);
        assert_eq!(report.layer_stats.len(), 10);

        // lm_head should have the highest max error
        assert_eq!(report.layer_stats[0].layer_name, "lm_head");
        assert!((report.layer_stats[0].max - 3.5e-14).abs() < 1e-20);
    }

    #[test]
    fn test_error_report_display() {
        let mut accumulator = ErrorAccumulator::new();
        accumulator.record_error("layer_0", 1.0e-14);
        accumulator.record_error("layer_1", 2.0e-14);

        let report = accumulator.generate_report();
        let display = format!("{}", report);

        assert!(display.contains("ERROR ACCUMULATION REPORT"));
        assert!(display.contains("GLOBAL STATISTICS"));
        assert!(display.contains("PER-LAYER STATISTICS"));
        assert!(display.contains("layer_0"));
        assert!(display.contains("layer_1"));
    }

    #[test]
    fn test_layer_error_stats_display() {
        let stats = LayerErrorStats {
            layer_name: "test_layer".to_string(),
            mean: 1.5e-14,
            std_dev: 0.5e-14,
            min: 1.0e-14,
            max: 2.0e-14,
            count: 3,
        };

        let display = format!("{}", stats);
        assert!(display.contains("test_layer"));
        assert!(display.contains("Count:"));
        assert!(display.contains("Mean:"));
    }

    #[test]
    fn test_error_statistics_display() {
        let stats = ErrorStatistics {
            mean: 1.5e-14,
            std_dev: 0.5e-14,
            min: 1.0e-14,
            max: 2.0e-14,
            total: 4.5e-14,
            count: 3,
        };

        let display = format!("{}", stats);
        assert!(display.contains("Error Statistics:"));
        assert!(display.contains("Count:"));
        assert!(display.contains("Mean:"));
    }
}
