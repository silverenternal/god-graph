//! Differentiable tensor wrapper with gradient support

use std::sync::Arc;
use crate::tensor::DenseTensor;
use crate::tensor::traits::{TensorOps, TensorBase};
use super::compute_graph::{ComputeGraph, OpId, TensorId, OpType};

/// A differentiable tensor that tracks gradients and compute graph information
#[derive(Debug, Clone)]
pub struct DifferentiableTensor {
    /// The underlying tensor data
    data: DenseTensor,
    /// Gradient of this tensor (computed during backward pass)
    grad: Option<DenseTensor>,
    /// ID of the operation that produced this tensor
    op_id: Option<OpId>,
    /// Tensor ID in the compute graph
    tensor_id: TensorId,
    /// Whether this tensor requires gradient
    requires_grad: bool,
    /// Reference to the compute graph (weak reference to avoid cycles)
    #[allow(dead_code)]
    graph: Option<Arc<ComputeGraph>>,
}

impl DifferentiableTensor {
    /// Create a new differentiable tensor
    pub fn new(data: DenseTensor, requires_grad: bool) -> Self {
        Self {
            data,
            grad: None,
            op_id: None,
            tensor_id: TensorId(0),
            requires_grad,
            graph: None,
        }
    }

    /// Create a differentiable tensor with compute graph tracking
    pub fn with_graph(data: DenseTensor, requires_grad: bool, graph: &mut ComputeGraph) -> Self {
        let tensor_id = graph.next_tensor_id();
        graph.store_value(tensor_id, data.clone());
        
        Self {
            data,
            grad: None,
            op_id: None,
            tensor_id,
            requires_grad,
            graph: Some(Arc::new(graph.clone())),
        }
    }

    /// Get the underlying data
    pub fn data(&self) -> &DenseTensor {
        &self.data
    }

    /// Get mutable reference to data
    pub fn data_mut(&mut self) -> &mut DenseTensor {
        &mut self.data
    }

    /// Get the gradient (if computed)
    pub fn grad(&self) -> Option<&DenseTensor> {
        self.grad.as_ref()
    }

    /// Get mutable reference to gradient
    pub fn grad_mut(&mut self) -> Option<&mut DenseTensor> {
        self.grad.as_mut()
    }

    /// Set the gradient
    pub fn set_grad(&mut self, grad: DenseTensor) {
        self.grad = Some(grad);
    }

    /// Clear the gradient
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Check if this tensor requires gradient
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the tensor ID
    pub fn tensor_id(&self) -> TensorId {
        self.tensor_id
    }

    /// Get the operation ID that produced this tensor
    pub fn op_id(&self) -> Option<OpId> {
        self.op_id
    }

    /// Set operation ID (called by compute graph)
    #[allow(dead_code)]
    pub(crate) fn set_op_id(&mut self, op_id: OpId) {
        self.op_id = Some(op_id);
    }

    /// Get shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &DifferentiableTensor, graph: &mut ComputeGraph) -> DifferentiableTensor {
        let output_id = graph.next_tensor_id();
        let result_data = self.data.matmul(&other.data);
        
        graph.record_op(OpType::MatMul, &[self.tensor_id, other.tensor_id], output_id);
        graph.store_value(output_id, result_data.clone());
        
        DifferentiableTensor {
            data: result_data,
            grad: None,
            op_id: None,
            tensor_id: output_id,
            requires_grad: self.requires_grad || other.requires_grad,
            graph: Some(Arc::new(graph.clone())),
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &DifferentiableTensor, graph: &mut ComputeGraph) -> DifferentiableTensor {
        let output_id = graph.next_tensor_id();
        let result_data = self.data.add(&other.data);
        
        graph.record_op(OpType::Add, &[self.tensor_id, other.tensor_id], output_id);
        graph.store_value(output_id, result_data.clone());
        
        DifferentiableTensor {
            data: result_data,
            grad: None,
            op_id: None,
            tensor_id: output_id,
            requires_grad: self.requires_grad || other.requires_grad,
            graph: Some(Arc::new(graph.clone())),
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &DifferentiableTensor, graph: &mut ComputeGraph) -> DifferentiableTensor {
        let output_id = graph.next_tensor_id();
        let result_data = self.data.sub(&other.data);
        
        graph.record_op(OpType::Sub, &[self.tensor_id, other.tensor_id], output_id);
        graph.store_value(output_id, result_data.clone());
        
        DifferentiableTensor {
            data: result_data,
            grad: None,
            op_id: None,
            tensor_id: output_id,
            requires_grad: self.requires_grad || other.requires_grad,
            graph: Some(Arc::new(graph.clone())),
        }
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &DifferentiableTensor, graph: &mut ComputeGraph) -> DifferentiableTensor {
        let output_id = graph.next_tensor_id();
        let result_data = self.data.mul(&other.data);
        
        graph.record_op(OpType::Mul, &[self.tensor_id, other.tensor_id], output_id);
        graph.store_value(output_id, result_data.clone());
        
        DifferentiableTensor {
            data: result_data,
            grad: None,
            op_id: None,
            tensor_id: output_id,
            requires_grad: self.requires_grad || other.requires_grad,
            graph: Some(Arc::new(graph.clone())),
        }
    }

    /// ReLU activation
    pub fn relu(&self, graph: &mut ComputeGraph) -> DifferentiableTensor {
        let output_id = graph.next_tensor_id();
        let result_data = self.data.relu();
        
        graph.record_op(OpType::ReLU, &[self.tensor_id], output_id);
        graph.store_value(output_id, result_data.clone());
        
        DifferentiableTensor {
            data: result_data,
            grad: None,
            op_id: None,
            tensor_id: output_id,
            requires_grad: self.requires_grad,
            graph: Some(Arc::new(graph.clone())),
        }
    }

    /// GELU activation
    pub fn gelu(&self, graph: &mut ComputeGraph) -> DifferentiableTensor {
        let output_id = graph.next_tensor_id();
        let result_data = self.data.gelu();
        
        graph.record_op(OpType::GELU, &[self.tensor_id], output_id);
        graph.store_value(output_id, result_data.clone());
        
        DifferentiableTensor {
            data: result_data,
            grad: None,
            op_id: None,
            tensor_id: output_id,
            requires_grad: self.requires_grad,
            graph: Some(Arc::new(graph.clone())),
        }
    }

    /// Softmax activation
    pub fn softmax(&self, dim: isize, graph: &mut ComputeGraph) -> DifferentiableTensor {
        let output_id = graph.next_tensor_id();
        let result_data = self.data.softmax(dim);
        
        graph.record_op(OpType::Softmax, &[self.tensor_id], output_id);
        graph.store_value(output_id, result_data.clone());
        
        DifferentiableTensor {
            data: result_data,
            grad: None,
            op_id: None,
            tensor_id: output_id,
            requires_grad: self.requires_grad,
            graph: Some(Arc::new(graph.clone())),
        }
    }

    /// Transpose
    pub fn transpose(&self, graph: &mut ComputeGraph) -> DifferentiableTensor {
        let output_id = graph.next_tensor_id();
        let result_data = self.data.transpose(None);

        graph.record_op(OpType::Transpose, &[self.tensor_id], output_id);
        graph.store_value(output_id, result_data.clone());
        
        DifferentiableTensor {
            data: result_data,
            grad: None,
            op_id: None,
            tensor_id: output_id,
            requires_grad: self.requires_grad,
            graph: Some(Arc::new(graph.clone())),
        }
    }

    /// Detach from compute graph (for inference)
    pub fn detach(&self) -> DifferentiableTensor {
        DifferentiableTensor {
            data: self.data.clone(),
            grad: None,
            op_id: None,
            tensor_id: TensorId(0),
            requires_grad: false,
            graph: None,
        }
    }

    /// Perform backward pass
    pub fn backward(&mut self, graph: &mut ComputeGraph) {
        if self.requires_grad {
            graph.backward(self.tensor_id);

            // Collect gradients
            if let Some(grad) = graph.get_gradient(self.tensor_id).cloned() {
                self.grad = Some(grad);
            }
        }
    }
}

/// Convert from DenseTensor
impl From<DenseTensor> for DifferentiableTensor {
    fn from(tensor: DenseTensor) -> Self {
        Self::new(tensor, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DenseTensor;

    #[test]
    fn test_differentiable_tensor_creation() {
        let data = DenseTensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let tensor = DifferentiableTensor::new(data.clone(), true);
        
        assert!(tensor.requires_grad());
        assert_eq!(tensor.data(), &data);
        assert!(tensor.grad().is_none());
    }

    #[test]
    fn test_differentiable_matmul() {
        let mut graph = ComputeGraph::new();
        
        let x = DenseTensor::new(vec![1.0, 2.0], vec![1, 2]);
        let w = DenseTensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        
        let x_diff = DifferentiableTensor::with_graph(x, true, &mut graph);
        let w_diff = DifferentiableTensor::with_graph(w, true, &mut graph);
        
        let out = x_diff.matmul(&w_diff, &mut graph);
        
        assert!(out.requires_grad());
        assert_eq!(out.shape(), &[1, 2]);
    }

    #[test]
    fn test_differentiable_relu() {
        let mut graph = ComputeGraph::new();
        
        let data = DenseTensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![1, 4]);
        let tensor = DifferentiableTensor::with_graph(data, true, &mut graph);
        
        let out = tensor.relu(&mut graph);
        
        // ReLU should zero out negative values
        assert_eq!(out.shape(), &[1, 4]);
    }
}
