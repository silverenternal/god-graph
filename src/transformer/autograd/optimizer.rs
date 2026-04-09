//! Optimizers for training (Adam, SGD)

use std::collections::HashMap;
use crate::tensor::DenseTensor;
use crate::tensor::traits::{TensorBase, TensorOps};
use super::compute_graph::TensorId;

/// Trait for optimization algorithms
pub trait Optimizer {
    /// Initialize optimizer state for a parameter
    fn init_param(&mut self, param_id: TensorId, param: &DenseTensor);
    
    /// Update a single parameter
    fn step_param(&mut self, param_id: TensorId, param: &mut DenseTensor, grad: &DenseTensor);
    
    /// Perform optimization step for all parameters
    fn step(&mut self, params: &mut HashMap<TensorId, DenseTensor>);
}

/// SGD optimizer with optional momentum and weight decay
#[derive(Debug, Clone)]
pub struct Sgd {
    /// Learning rate
    pub lr: f64,
    /// Momentum factor
    pub momentum: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Momentum buffers
    velocity: HashMap<TensorId, DenseTensor>,
}

impl Sgd {
    /// Create a new SGD optimizer
    pub fn new(lr: f64, momentum: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            velocity: HashMap::new(),
        }
    }
}

impl Default for Sgd {
    /// Create SGD with default parameters (lr=0.01, no momentum, no weight decay)
    fn default() -> Self {
        Self::new(0.01, 0.0, 0.0)
    }
}

impl Optimizer for Sgd {
    fn init_param(&mut self, param_id: TensorId, param: &DenseTensor) {
        if self.momentum > 0.0 {
            // Initialize velocity buffer with zeros
            let zeros = DenseTensor::zeros(param.shape().to_vec());
            self.velocity.insert(param_id, zeros);
        }
    }
    
    fn step_param(&mut self, param_id: TensorId, param: &mut DenseTensor, grad: &DenseTensor) {
        // Apply weight decay
        let mut effective_grad = grad.clone();
        if self.weight_decay > 0.0 {
            let decay_grad = param.scale(self.weight_decay);
            effective_grad = effective_grad.add(&decay_grad);
        }
        
        if self.momentum > 0.0 {
            // Update with momentum: v = momentum * v + grad
            if let Some(v) = self.velocity.get_mut(&param_id) {
                let scaled_v = v.scale(self.momentum);
                *v = scaled_v.add(&effective_grad);
                
                // param = param - lr * v
                let update = v.scale(self.lr);
                *param = param.sub(&update);
            }
        } else {
            // Simple SGD: param = param - lr * grad
            let update = effective_grad.scale(self.lr);
            *param = param.sub(&update);
        }
    }
    
    fn step(&mut self, params: &mut HashMap<TensorId, DenseTensor>) {
        for (param_id, param) in params.iter_mut() {
            // Initialize if not already done
            if !self.velocity.contains_key(param_id) && self.momentum > 0.0 {
                self.init_param(*param_id, param);
            }
            
            // Get gradient (placeholder - in real use, gradients come from compute graph)
            // This is a simplified version; actual implementation would fetch from compute graph
            let grad = DenseTensor::zeros(param.shape().to_vec());
            self.step_param(*param_id, param, &grad);
        }
    }
}

/// Adam optimizer
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate
    pub lr: f64,
    /// Beta1 for first moment (default: 0.9)
    pub beta1: f64,
    /// Beta2 for second moment (default: 0.999)
    pub beta2: f64,
    /// Epsilon for numerical stability (default: 1e-8)
    pub epsilon: f64,
    /// First moment estimates (m)
    m: HashMap<TensorId, DenseTensor>,
    /// Second moment estimates (v)
    v: HashMap<TensorId, DenseTensor>,
    /// Timestep
    t: usize,
}

impl Adam {
    /// Create a new Adam optimizer
    pub fn new(lr: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
    
    /// Set learning rate
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl Default for Adam {
    /// Create Adam with default parameters (lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8)
    }
}

impl Optimizer for Adam {
    fn init_param(&mut self, param_id: TensorId, param: &DenseTensor) {
        // Initialize first and second moment estimates with zeros
        let zeros = DenseTensor::zeros(param.shape().to_vec());
        self.m.insert(param_id, zeros.clone());
        self.v.insert(param_id, zeros);
    }
    
    fn step_param(&mut self, param_id: TensorId, param: &mut DenseTensor, grad: &DenseTensor) {
        // Initialize if not already done
        if !self.m.contains_key(&param_id) {
            self.init_param(param_id, param);
        }
        
        if let (Some(m), Some(v)) = (self.m.get_mut(&param_id), self.v.get_mut(&param_id)) {
            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            let grad_scaled = grad.scale(1.0 - self.beta1);
            let m_scaled = m.scale(self.beta1);
            *m = m_scaled.add(&grad_scaled);
            
            // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            let grad_squared = grad.mul(grad);
            let grad_squared_scaled = grad_squared.scale(1.0 - self.beta2);
            let v_scaled = v.scale(self.beta2);
            *v = v_scaled.add(&grad_squared_scaled);
            
            // Compute bias-corrected estimates
            let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
            
            let m_hat = m.scale(1.0 / bias_correction1);
            let v_hat = v.scale(1.0 / bias_correction2);
            
            // Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            let sqrt_v = v_hat.sqrt().add(&DenseTensor::full(v_hat.shape(), self.epsilon));
            let update = m_hat.div(&sqrt_v).scale(self.lr);
            
            *param = param.sub(&update);
        }
    }
    
    fn step(&mut self, params: &mut HashMap<TensorId, DenseTensor>) {
        // Increment timestep
        self.t += 1;
        
        for (param_id, param) in params.iter_mut() {
            // Initialize if not already done
            if !self.m.contains_key(param_id) {
                self.init_param(*param_id, param);
            }
            
            // Get gradient (placeholder - in real use, gradients come from compute graph)
            let grad = DenseTensor::zeros(param.shape().to_vec());
            self.step_param(*param_id, param, &grad);
        }
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
#[derive(Debug, Clone)]
pub struct AdamW {
    /// Base Adam optimizer
    pub adam: Adam,
    /// Weight decay
    pub weight_decay: f64,
}

impl AdamW {
    /// Create a new AdamW optimizer
    pub fn new(lr: f64, beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64) -> Self {
        Self {
            adam: Adam::new(lr, beta1, beta2, epsilon),
            weight_decay,
        }
    }
}

impl Default for AdamW {
    /// Create AdamW with default parameters (lr=0.001, weight_decay=0.01)
    fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, 0.01)
    }
}

impl Optimizer for AdamW {
    fn init_param(&mut self, param_id: TensorId, param: &DenseTensor) {
        self.adam.init_param(param_id, param);
    }
    
    fn step_param(&mut self, param_id: TensorId, param: &mut DenseTensor, grad: &DenseTensor) {
        // Apply decoupled weight decay directly to parameters
        if self.weight_decay > 0.0 {
            let decay = param.scale(self.weight_decay * self.adam.lr);
            *param = param.sub(&decay);
        }
        
        // Use Adam for gradient update
        self.adam.step_param(param_id, param, grad);
    }
    
    fn step(&mut self, params: &mut HashMap<TensorId, DenseTensor>) {
        self.adam.step(params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_basic() {
        let mut optimizer = Sgd::new(0.01, 0.0, 0.0);
        let mut params = HashMap::new();

        let param_id = TensorId(0);
        let param = DenseTensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        params.insert(param_id, param);

        optimizer.step(&mut params);

        // With zero gradient, parameter should not change
        let updated = params.get(&param_id).unwrap();
        assert_eq!(updated.data(), &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_adam_basic() {
        let mut optimizer = Adam::default();
        let mut params = HashMap::new();

        let param_id = TensorId(0);
        let param = DenseTensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        params.insert(param_id, param);

        optimizer.step(&mut params);

        // With zero gradient, parameter should not change
        let updated = params.get(&param_id).unwrap();
        assert_eq!(updated.data(), &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut optimizer = Sgd::new(0.01, 0.9, 0.0);
        
        let param_id = TensorId(0);
        let param = DenseTensor::new(vec![1.0, 2.0], vec![1, 2]);
        optimizer.init_param(param_id, &param);
        
        // Check that velocity buffer was initialized
        assert!(optimizer.velocity.contains_key(&param_id));
    }
}
