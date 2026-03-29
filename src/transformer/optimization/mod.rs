//! CAD-LLM Topology Optimization Module
//!
//! This module implements the CAD-inspired topology optimization framework for LLMs,
//! combining Lie group decomposition, tensor ring compression, and topology-preserving
//! transformations to achieve lossless, controllable, and interpretable LLM optimization.
//!
//! ## Core Philosophy
//!
//! Transform LLM optimization from "alchemy trial-and-error" to "engineering design"
//! by establishing topology isomorphism between CAD feature trees and LLM computation graphs.
//!
//! ## Design Principles
//!
//! - **P1 No Blackbox**: Avoid traditional pruning/quantization/distillation blackbox optimization
//! - **P2 Topology Preserving**: Maintain computation graph topology, only optimize node weights
//! - **P3 No Parameter Bloat**: No new learnable parameters, prevent parameter inflation
//! - **P4 Engineering Paradigm**: Transform LLM optimization to engineering design modifications
//! - **P5 Mathematically Rigorous**: Based on Lie group decomposition, tensor ring compression
//! - **P6 CAD Inspired**: Reuse CAD feature tree editing, constraint solving, assembly validation
//!
//! ## Modules
//!
//! - [`switch`]: Bidirectional lossless conversion between Safetensors and GodGraph
//! - [`lie_group`]: Lie group decomposition and orthogonalization
//! - [`tensor_ring`]: Tensor ring compression for parameter reduction
//! - [`cad_editor`]: CAD-style topology editor for defect detection and constraint solving
//! - [`constraints`]: Topology constraints and validation
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "safetensors")]
//! use god_gragh::transformer::optimization::{ModelSwitch, LieGroupOptimizer, LieGroupConfig};
//! use god_gragh::transformer::optimization::{TensorRingCompressor, CompressionConfig};
//! use god_gragh::transformer::optimization::{CadStyleEditor, TopologyConstraint};
//!
//! # #[cfg(feature = "safetensors")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // 1. Load model from Safetensors
//! let mut graph = ModelSwitch::load_from_safetensors("model.safetensors")?;
//!
//! // 2. Validate topology
//! let report = ModelSwitch::validate_topology(&graph)?;
//! assert!(report.is_valid);
//!
//! // 3. Apply Lie group orthogonalization
//! let config = LieGroupConfig::new()
//!     .with_block_size(64)
//!     .with_orthogonalize(true);
//! let optimizer = LieGroupOptimizer::new(config);
//! optimizer.orthogonalize_weights(&mut graph)?;
//!
//! // 4. Apply tensor ring compression
//! let config = CompressionConfig::new()
//!     .with_target_ranks(vec![32, 64])
//!     .with_layers(vec!["qkv".to_string(), "mlp".to_string()]);
//! let compressor = TensorRingCompressor::new(config);
//! let compressed = compressor.compress_graph(&graph)?;
//!
//! // 5. CAD-style topology editing
//! let mut editor = CadStyleEditor::new(&mut graph);
//! let defects = editor.detect_defects()?;
//! editor.add_constraint(TopologyConstraint::ResidualConnection {
//!     from_layer: "layer.0".to_string(),
//!     to_layer: "layer.1".to_string(),
//! })?;
//! editor.solve_constraints()?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "safetensors"))]
//! # fn main() {}
//! ```

// switch module is always available when tensor feature is enabled
// safetensors feature is only needed for actual Safetensors file I/O
pub mod switch;

#[cfg(feature = "tensor")]
pub mod lie_group;

#[cfg(feature = "tensor")]
pub mod tensor_ring;

#[cfg(feature = "tensor")]
pub mod cad_editor;

#[cfg(feature = "tensor")]
pub mod constraints;

#[cfg(feature = "tensor")]
pub mod error_analysis;

// Re-export main types
#[cfg(feature = "safetensors")]
pub use switch::ModelSwitch;

#[cfg(feature = "tensor")]
pub use lie_group::{LieGroupConfig, LieGroupOptimizer, SOkBlock, decompose_into_so_blocks};

#[cfg(feature = "tensor")]
pub use tensor_ring::{CompressionConfig, TensorRingCompressor, mixed_precision_compress, adaptive_rank_selection};

#[cfg(feature = "tensor")]
pub use cad_editor::CadStyleEditor;

#[cfg(feature = "tensor")]
pub use constraints::{ConstraintReport, TopologyConstraint, TopologyDefect, TopologyValidator};

#[cfg(feature = "tensor")]
pub use error_analysis::{ErrorAccumulator, ErrorReport, ErrorStatistics, LayerErrorStats};

// Re-export TensorRing from decomposition module
#[cfg(feature = "tensor")]
pub use crate::tensor::decomposition::tensor_ring::TensorRing;
