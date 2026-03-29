//! Compute graph for tracking operations during forward pass

use crate::tensor::traits::{TensorBase, TensorOps};
use crate::tensor::DenseTensor;
use std::collections::HashMap;

/// Unique identifier for an operation node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub usize);

/// Unique identifier for a tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

/// Reference to an operation
#[derive(Debug, Clone)]
pub struct OpRef {
    /// Operation ID
    pub id: OpId,
    /// Operation type
    pub op_type: OpType,
    /// Input tensor IDs
    pub inputs: Vec<TensorId>,
    /// Output tensor ID
    pub output: TensorId,
}

/// Operation type
#[derive(Debug, Clone)]
pub enum OpType {
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// Matrix multiplication
    MatMul,
    /// Transpose operation
    Transpose,
    /// Sum reduction
    Sum,
    /// Mean reduction
    Mean,
    /// ReLU activation
    ReLU,
    /// GELU activation
    GELU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// SiLU/Swish activation
    SiLU,
    /// Softmax activation
    Softmax,
    /// Layer normalization
    LayerNorm,
    /// RMS normalization
    RMSNorm,
    /// Linear/fully connected layer
    Linear,
    /// Embedding lookup
    Embedding,
    /// Rotary position embedding
    RoPE,
    /// Scaled dot-product attention
    ScaledDotProduct,
}

/// Operation node in the compute graph
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Operation ID
    pub id: OpId,
    /// Operation type
    pub op_type: OpType,
    /// Input tensor IDs
    pub inputs: Vec<TensorId>,
    /// Output tensor ID
    pub output: TensorId,
}

/// Data edge representing tensor dependencies
#[derive(Debug, Clone)]
pub struct DataEdge {
    /// Source operation ID
    pub from: OpId,
    /// Destination operation ID
    pub to: OpId,
    /// Tensor ID
    pub tensor_id: TensorId,
}

/// Checkpoint for gradient checkpointing
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Tensor values
    pub tensors: HashMap<TensorId, DenseTensor>,
}

/// Compute graph for tracking operations
#[derive(Debug, Default, Clone)]
pub struct ComputeGraph {
    /// Operation nodes
    nodes: Vec<OpNode>,
    /// Data flow edges
    edges: Vec<DataEdge>,
    /// Gradient storage
    gradients: HashMap<TensorId, DenseTensor>,
    /// Tensor values (for forward pass)
    values: HashMap<TensorId, DenseTensor>,
    /// Checkpoint for memory optimization
    checkpoint: Option<Checkpoint>,
    /// Next operation ID
    next_op_id: usize,
    /// Next tensor ID
    next_tensor_id: usize,
    /// Whether to record operations (disable during eval mode)
    recording: bool,
}

impl ComputeGraph {
    /// Create a new compute graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            gradients: HashMap::new(),
            values: HashMap::new(),
            checkpoint: None,
            next_op_id: 0,
            next_tensor_id: 0,
            recording: true,
        }
    }

    /// Generate a new operation ID
    pub fn next_op_id(&mut self) -> OpId {
        let id = OpId(self.next_op_id);
        self.next_op_id += 1;
        id
    }

    /// Generate a new tensor ID
    pub fn next_tensor_id(&mut self) -> TensorId {
        let id = TensorId(self.next_tensor_id);
        self.next_tensor_id += 1;
        id
    }

    /// Record an operation in the compute graph
    pub fn record_op(&mut self, op_type: OpType, inputs: &[TensorId], output: TensorId) {
        if !self.recording {
            return;
        }

        let op_id = self.next_op_id();
        let node = OpNode {
            id: op_id,
            op_type: op_type.clone(),
            inputs: inputs.to_vec(),
            output,
        };
        self.nodes.push(node);

        // Create edges from input producers to this operation
        for &input_id in inputs {
            // Find the operation that produced this input
            if let Some(producer_op) = self.nodes.iter().rev().find(|n| n.output == input_id) {
                let edge = DataEdge {
                    from: producer_op.id,
                    to: op_id,
                    tensor_id: input_id,
                };
                self.edges.push(edge);
            }
        }
    }

    /// Store a tensor value
    pub fn store_value(&mut self, tensor_id: TensorId, value: DenseTensor) {
        self.values.insert(tensor_id, value);
    }

    /// Get a tensor value
    pub fn get_value(&self, tensor_id: TensorId) -> Option<&DenseTensor> {
        self.values.get(&tensor_id)
    }

    /// Get a mutable reference to a tensor value
    pub fn get_value_mut(&mut self, tensor_id: TensorId) -> Option<&mut DenseTensor> {
        self.values.get_mut(&tensor_id)
    }

    /// Store a gradient
    pub fn store_gradient(&mut self, tensor_id: TensorId, gradient: DenseTensor) {
        self.gradients.insert(tensor_id, gradient);
    }

    /// Get a gradient
    pub fn get_gradient(&self, tensor_id: TensorId) -> Option<&DenseTensor> {
        self.gradients.get(&tensor_id)
    }

    /// Perform backward pass to compute gradients
    ///
    /// # Arguments
    /// * `loss` - The tensor ID of the loss value
    ///
    /// # Returns
    /// A HashMap mapping tensor IDs to their computed gradients
    pub fn backward(&mut self, loss: TensorId) -> HashMap<TensorId, DenseTensor> {
        // Initialize gradient of loss as 1.0
        if let Some(loss_tensor) = self.values.get(&loss) {
            let shape = loss_tensor.shape().to_vec();
            let ones = DenseTensor::ones(shape);
            self.gradients.insert(loss, ones);
        }

        // Get topological order of operations (reverse)
        let topo_order = self.topological_sort();

        // Backpropagate in reverse topological order
        for op_id in topo_order.into_iter().rev() {
            // Clone node info to avoid borrow checker issues
            let (node_op_type, node_inputs, node_output) =
                if let Some(node) = self.nodes.iter().find(|n| n.id == op_id) {
                    (node.op_type.clone(), node.inputs.clone(), node.output)
                } else {
                    continue;
                };

            let grad_output = self.gradients.get(&node_output).cloned();

            if let Some(grad) = grad_output {
                // Compute gradients for inputs based on operation type
                let input_grads = self.compute_gradients(&node_op_type, &node_inputs, &grad);

                // Accumulate gradients for inputs
                for (i, &input_id) in node_inputs.iter().enumerate() {
                    if let Some(input_grad) = input_grads.get(&i) {
                        self.accumulate_gradient(input_id, input_grad.clone());
                    }
                }
            }
        }

        self.gradients.clone()
    }

    /// Compute gradients for a specific operation
    fn compute_gradients(
        &self,
        op_type: &OpType,
        inputs: &[TensorId],
        grad_output: &DenseTensor,
    ) -> HashMap<usize, DenseTensor> {
        let mut grads = HashMap::new();

        match op_type {
            OpType::Add => {
                // d(x+y)/dx = 1, d(x+y)/dy = 1
                for (i, _) in inputs.iter().enumerate() {
                    grads.insert(i, grad_output.clone());
                }
            }
            OpType::Sub => {
                // d(x-y)/dx = 1, d(x-y)/dy = -1
                for (i, _) in inputs.iter().enumerate() {
                    if i == 0 {
                        grads.insert(i, grad_output.clone());
                    } else {
                        grads.insert(i, grad_output.neg());
                    }
                }
            }
            OpType::Mul => {
                // Element-wise multiplication: d(x*y)/dx = y, d(x*y)/dy = x
                if inputs.len() >= 2 {
                    if let (Some(x), Some(y)) =
                        (self.values.get(&inputs[0]), self.values.get(&inputs[1]))
                    {
                        grads.insert(0, grad_output.mul(y));
                        grads.insert(1, grad_output.mul(x));
                    }
                }
            }
            OpType::MatMul => {
                // Matrix multiplication: d(X@W)/dX = d_out @ W.T, d(X@W)/dW = X.T @ d_out
                if inputs.len() >= 2 {
                    if let (Some(x), Some(w)) =
                        (self.values.get(&inputs[0]), self.values.get(&inputs[1]))
                    {
                        // Gradient w.r.t. input
                        let w_t = w.transpose(None);
                        let grad_x = grad_output.matmul(&w_t);
                        grads.insert(0, grad_x);

                        // Gradient w.r.t. weights
                        let x_t = x.transpose(None);
                        let grad_w = x_t.matmul(grad_output);
                        grads.insert(1, grad_w);
                    }
                }
            }
            OpType::ReLU => {
                // ReLU gradient: 1 if x > 0, else 0
                if let Some(x) = inputs.first().and_then(|id| self.values.get(id)) {
                    let mask = x.gt(0.0);
                    let grad = grad_output.mul(&mask);
                    grads.insert(0, grad);
                }
            }
            OpType::GELU => {
                // GELU gradient approximation
                if let Some(x) = inputs.first().and_then(|id| self.values.get(id)) {
                    let gelu_grad = x.gelu_derivative();
                    let grad = grad_output.mul(&gelu_grad);
                    grads.insert(0, grad);
                }
            }
            OpType::Softmax => {
                // Softmax gradient: s * (grad - sum(grad * s))
                if let Some(softmax_out) = inputs.first().and_then(|id| self.values.get(id)) {
                    let sum_grad_dot_s = grad_output.mul(softmax_out).sum(None);
                    let ones = DenseTensor::ones(softmax_out.shape().to_vec());
                    let ones_scaled = ones.scale(sum_grad_dot_s.data()[0]);
                    let diff = grad_output.sub(&ones_scaled);
                    let grad = softmax_out.mul(&diff);
                    grads.insert(0, grad);
                }
            }
            OpType::Transpose => {
                // Transpose gradient is just transpose of gradient
                if !inputs.is_empty() {
                    grads.insert(0, grad_output.transpose(None));
                }
            }
            OpType::LayerNorm | OpType::RMSNorm => {
                // Normalization gradients (simplified)
                if inputs.first().and_then(|id| self.values.get(id)).is_some() {
                    // Placeholder - actual implementation needs more careful handling
                    grads.insert(0, grad_output.clone());
                }
            }
            _ => {
                // For unimplemented operations, pass gradient through
                for (i, _) in inputs.iter().enumerate() {
                    grads.insert(i, grad_output.clone());
                }
            }
        }

        grads
    }

    /// Accumulate gradient for a tensor
    pub fn accumulate_gradient(&mut self, tensor_id: TensorId, gradient: DenseTensor) {
        self.gradients
            .entry(tensor_id)
            .and_modify(|existing| {
                *existing = existing.add(&gradient);
            })
            .or_insert(gradient);
    }

    /// Perform topological sort of operations
    pub fn topological_sort(&self) -> Vec<OpId> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn visit(
            node: &OpNode,
            nodes: &[OpNode],
            visited: &mut std::collections::HashSet<OpId>,
            result: &mut Vec<OpId>,
        ) {
            if visited.contains(&node.id) {
                return;
            }
            visited.insert(node.id);

            // Visit producers first
            for &input_id in &node.inputs {
                if let Some(producer) = nodes.iter().find(|n| n.output == input_id) {
                    visit(producer, nodes, visited, result);
                }
            }

            result.push(node.id);
        }

        for node in &self.nodes {
            visit(node, &self.nodes, &mut visited, &mut result);
        }

        result
    }

    /// Clear the compute graph (call after backward pass)
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.gradients.clear();
        self.values.clear();
        self.checkpoint = None;
    }

    /// Enable/disable operation recording
    pub fn set_recording(&mut self, recording: bool) {
        self.recording = recording;
    }

    /// Check if recording is enabled
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Get number of recorded operations
    pub fn num_ops(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_graph_basic() {
        let mut graph = ComputeGraph::new();

        // Create some tensors
        let x_id = graph.next_tensor_id();
        let w_id = graph.next_tensor_id();

        let x = DenseTensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let w = DenseTensor::new(vec![0.1, 0.2, 0.3], vec![3, 1]);

        graph.store_value(x_id, x);
        graph.store_value(w_id, w);

        // Record MatMul operation
        let out_id = graph.next_tensor_id();
        graph.record_op(OpType::MatMul, &[x_id, w_id], out_id);

        // Compute output
        if let (Some(x), Some(w)) = (graph.get_value(x_id), graph.get_value(w_id)) {
            let out = x.matmul(w);
            graph.store_value(out_id, out);
        }

        assert_eq!(graph.num_ops(), 1);
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = ComputeGraph::new();

        // Create a simple chain: x -> MatMul -> ReLU -> output
        let x_id = graph.next_tensor_id();
        let w_id = graph.next_tensor_id();
        let matmul_out = graph.next_tensor_id();
        let relu_out = graph.next_tensor_id();

        graph.store_value(x_id, DenseTensor::new(vec![1.0, 2.0], vec![1, 2]));
        graph.store_value(w_id, DenseTensor::new(vec![0.1, 0.2], vec![2, 1]));

        graph.record_op(OpType::MatMul, &[x_id, w_id], matmul_out);
        graph.record_op(OpType::ReLU, &[matmul_out], relu_out);

        let order = graph.topological_sort();
        assert_eq!(order.len(), 2);
        // MatMul should come before ReLU
        assert!(
            order
                .iter()
                .position(|&id| {
                    graph
                        .nodes
                        .iter()
                        .any(|n| n.id == id && matches!(n.op_type, OpType::MatMul))
                })
                .unwrap()
                < order
                    .iter()
                    .position(|&id| {
                        graph
                            .nodes
                            .iter()
                            .any(|n| n.id == id && matches!(n.op_type, OpType::ReLU))
                    })
                    .unwrap()
        );
    }
}
