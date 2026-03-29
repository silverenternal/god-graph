//! CAD-LLM Lie Group Orthogonalization Example
//!
//! This example demonstrates how to:
//! 1. Create a Lie group optimizer configuration
//! 2. Orthogonalize weight matrices using QR decomposition
//! 3. Apply SO(k) block decomposition
//! 4. Verify orthogonality constraints

use god_gragh::tensor::{DenseTensor, TensorBase};
use god_gragh::transformer::optimization::{
    decompose_into_so_blocks, LieGroupConfig, LieGroupOptimizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CAD-LLM Lie Group Orthogonalization Example ===\n");

    // 1. Create optimizer configuration
    println!("Step 1: Creating Lie group optimizer configuration...");
    let config = LieGroupConfig::new()
        .with_block_size(64)
        .with_orthogonalize(true)
        .with_target_layers(vec!["q_proj".to_string(), "k_proj".to_string(), "v_proj".to_string()])
        .with_iterations(10);

    println!("  Block size: {}", config.block_size);
    println!("  Orthogonalize: {}", config.orthogonalize);
    println!("  Target layers: {:?}", config.target_layers);
    println!("  Iterations: {}\n", config.iterations);

    // 2. Create optimizer
    println!("Step 2: Creating Lie group optimizer...");
    let optimizer = LieGroupOptimizer::new(config.clone());
    println!("  Optimizer created successfully\n");

    // 3. Create a sample weight matrix
    println!("Step 3: Creating sample weight matrix...");
    let weight = DenseTensor::from_vec(
        (0..128 * 128)
            .map(|i| ((i % 50) as f64) / 50.0)
            .collect(),
        vec![128, 128],
    );
    println!("  Weight shape: {:?}", weight.shape());

    // Check initial orthogonality
    let is_ortho_before = check_orthogonality(&weight, 1e-5);
    println!("  Is orthogonal (before): {}\n", is_ortho_before);

    // 4. Apply orthogonalization
    println!("Step 4: Applying QR orthogonalization...");
    let orthogonalized = optimizer.cayley_transform(&weight)?;
    
    let is_ortho_after = check_orthogonality(&orthogonalized, 1e-5);
    println!("  Is orthogonal (after): {}", is_ortho_after);

    // Calculate orthogonality error
    let ortho_error = calculate_orthogonality_error(&orthogonalized);
    println!("  Orthogonality error: {:.2e}\n", ortho_error);

    // 5. Apply block decomposition
    println!("Step 5: Applying SO(k) block decomposition...");
    let blocks = decompose_into_so_blocks(&weight, config.block_size)?;
    
    println!("  Number of blocks: {}", blocks.len());
    println!("  Block size: {}x{}", blocks.first().map(|b| b.size).unwrap_or(0), 
                                    blocks.first().map(|b| b.size).unwrap_or(0));
    
    // Check block orthogonality
    let ortho_blocks = blocks.iter()
        .filter(|b: &&god_gragh::transformer::optimization::SOkBlock| b.is_orthogonal(1e-5))
        .count();
    println!("  Orthogonal blocks: {}/{}\n", ortho_blocks, blocks.len());

    // 6. Lie algebra regularization
    println!("Step 6: Applying Lie algebra regularization...");
    let regularized = optimizer.lie_algebra_regularize(&weight)?;
    
    let is_skew = check_skew_symmetric(&regularized, 1e-5);
    println!("  Is skew-symmetric: {}", is_skew);
    
    let is_ortho_regularized = check_orthogonality(&regularized, 1e-5);
    println!("  Is orthogonal (regularized): {}\n", is_ortho_regularized);

    // 7. Show statistics
    println!("Step 7: Optimizer statistics...");
    let stats = optimizer.statistics();
    for (key, value) in stats.iter() {
        println!("  {}: {:.4}", key, value);
    }

    println!("\n=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("  - QR decomposition orthogonalizes weight matrices");
    println!("  - SO(k) block decomposition enables structured optimization");
    println!("  - Lie algebra regularization improves numerical stability");
    println!("  - Orthogonal weights improve training dynamics");

    Ok(())
}

/// Check if a matrix is orthogonal (Q^T Q ≈ I)
fn check_orthogonality(tensor: &DenseTensor, tolerance: f64) -> bool {
    let shape = tensor.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return false;
    }

    let n = shape[0];
    let data = tensor.data();

    // Check Q^T Q = I
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
                dot += data[k * n + i] * data[k * n + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            if (dot - expected).abs() > tolerance {
                return false;
            }
        }
    }

    true
}

/// Calculate orthogonality error (||Q^T Q - I||_F)
fn calculate_orthogonality_error(tensor: &DenseTensor) -> f64 {
    let shape = tensor.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return f64::MAX;
    }

    let n = shape[0];
    let data = tensor.data();
    let mut error = 0.0;

    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
                dot += data[k * n + i] * data[k * n + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            error += (dot - expected).powi(2);
        }
    }

    error.sqrt()
}

/// Check if a matrix is skew-symmetric (A^T = -A)
fn check_skew_symmetric(tensor: &DenseTensor, tolerance: f64) -> bool {
    let shape = tensor.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return false;
    }

    let n = shape[0];
    let data = tensor.data();

    for i in 0..n {
        for j in 0..n {
            let idx_ij = i * n + j;
            let idx_ji = j * n + i;
            if (data[idx_ij] + data[idx_ji]).abs() > tolerance {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lie_group_optimizer() {
        let config = LieGroupConfig::new()
            .with_block_size(32)
            .with_orthogonalize(true);
        
        let optimizer = LieGroupOptimizer::new(config);
        
        let weight = DenseTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );

        let result = optimizer.cayley_transform(&weight);
        assert!(result.is_ok());
    }

    #[test]
    fn test_so_block_decomposition() {
        let weight = DenseTensor::from_vec(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
        );

        let blocks = decompose_into_so_blocks(&weight, 2).unwrap();
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].is_orthogonal(1e-5));
    }

    #[test]
    fn test_orthogonality_check() {
        // Identity matrix is orthogonal
        let identity = DenseTensor::from_vec(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
        );
        assert!(check_orthogonality(&identity, 1e-5));

        // Rotation matrix is orthogonal
        let theta = std::f64::consts::PI / 4.0;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let rotation = DenseTensor::from_vec(
            vec![cos_t, -sin_t, sin_t, cos_t],
            vec![2, 2],
        );
        assert!(check_orthogonality(&rotation, 1e-5));
    }
}
