//! Autograd engine for automatic differentiation
//!
//! This module provides a compute graph-based autograd engine similar to PyTorch's autograd.
//! It records operations during forward pass and computes gradients during backward pass.

pub mod compute_graph;
pub mod op;
pub mod tensor;
pub mod optimizer;

pub use compute_graph::{ComputeGraph, OpId, TensorId, OpType};
pub use tensor::DifferentiableTensor;
pub use optimizer::{Optimizer, Adam, Sgd};

/// Type alias for operations
pub type Op = OpType;
