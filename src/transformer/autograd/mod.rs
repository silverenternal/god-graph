//! Autograd engine for automatic differentiation
//!
//! This module provides a compute graph-based autograd engine similar to PyTorch's autograd.
//! It records operations during forward pass and computes gradients during backward pass.

pub mod compute_graph;
pub mod op;
pub mod optimizer;
pub mod tensor;

pub use compute_graph::{ComputeGraph, OpId, OpType, TensorId};
pub use optimizer::{Adam, Optimizer, Sgd};
pub use tensor::DifferentiableTensor;

/// Type alias for operations
pub type Op = OpType;
