//! CAD-LLM Tensor Ring Compression Example
//!
//! This example demonstrates how to:
//! 1. Create a compression configuration
//! 2. Decompose weight tensors using Tensor Ring
//! 3. Reconstruct and verify accuracy
//! 4. Query compression ratios
//!
//! Requires the `tensor` feature.

#[cfg(feature = "tensor")]
use god_graph::tensor::{DenseTensor, TensorBase};
#[cfg(feature = "tensor")]
use god_graph::transformer::optimization::{CompressionConfig, TensorRingCompressor};

#[cfg(feature = "tensor")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CAD-LLM Tensor Ring Compression Example ===\n");

    // 1. Create compression configuration
    println!("Step 1: Creating compression configuration...");
    let config = CompressionConfig::new()
        .with_target_ranks(vec![32, 64])
        .with_layers(vec!["qkv".to_string(), "mlp".to_string()])
        .with_min_rank(16)
        .with_max_rank(128);

    println!("  Target ranks: {:?}", config.target_ranks);
    println!("  Target layers: {:?}", config.layers);
    println!("  Rank range: [{}, {}]\n", config.min_rank, config.max_rank);

    // 2. Create a sample weight matrix
    println!("Step 2: Creating sample weight matrix...");
    let weight = DenseTensor::from_vec(
        (0..256 * 512).map(|i| ((i % 100) as f64) / 100.0).collect(),
        vec![256, 512],
    );
    println!("  Weight shape: {:?}", weight.shape());
    println!(
        "  Parameters: {}\n",
        weight.shape().iter().product::<usize>()
    );

    // 3. Create compressor and decompose
    println!("Step 3: Applying Tensor Ring decomposition...");
    let compressor = TensorRingCompressor::new(config);
    let ring = compressor.decompose(&weight)?;

    println!("  Number of cores: {}", ring.cores.len());
    println!("  TR ranks: {:?}", ring.ranks);
    println!("  Compression ratio: {:.2}x\n", ring.compression_ratio());

    // 4. Reconstruct and verify
    println!("Step 4: Reconstructing and verifying...");
    let reconstructed = ring.reconstruct()?;

    let original_data = weight.data();
    let recon_data = reconstructed.data();

    let mse: f64 = original_data
        .iter()
        .zip(recon_data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / original_data.len() as f64;

    let max_diff = original_data
        .iter()
        .zip(recon_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!("  Mean Squared Error: {:.6}", mse);
    println!("  Max Absolute Difference: {:.6}\n", max_diff);

    // 5. Show compression benefits
    println!("Step 5: Compression analysis...");
    let original_params = weight.shape().iter().product::<usize>();
    let compressed_params: usize = ring
        .cores
        .iter()
        .map(|c| c.shape().iter().product::<usize>())
        .sum();

    println!("  Original parameters: {}", original_params);
    println!("  Compressed parameters: {}", compressed_params);
    println!(
        "  Parameters saved: {}",
        original_params - compressed_params
    );
    println!(
        "  Memory reduction: {:.1}%",
        (1.0 - compressed_params as f64 / original_params as f64) * 100.0
    );

    println!("\n=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("  - Tensor Ring decomposition achieves lossy compression");
    println!("  - Higher ranks = better accuracy, less compression");
    println!("  - Lower ranks = more compression, lower accuracy");
    println!("  - Use adaptive rank selection for optimal results");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_ring_compression() {
        let config = CompressionConfig::new().with_target_ranks(vec![16]);
        let compressor = TensorRingCompressor::new(config);

        let weight = DenseTensor::from_vec(vec![1.0; 64 * 64], vec![64, 64]);

        let ring = compressor.decompose(&weight).unwrap();
        assert!(ring.compression_ratio() > 1.0);

        let reconstructed = ring.reconstruct().unwrap();
        assert_eq!(reconstructed.shape(), weight.shape());
    }

    #[test]
    fn test_adaptive_ranks() {
        // Test with different ranks
        for rank in [16, 32, 64] {
            let config = CompressionConfig::new().with_target_ranks(vec![rank]);
            let compressor = TensorRingCompressor::new(config);

            let weight = DenseTensor::from_vec(vec![1.0; 128 * 128], vec![128, 128]);

            let ring = compressor.decompose(&weight).unwrap();
            println!(
                "Rank {}: compression ratio = {:.2}x",
                rank,
                ring.compression_ratio()
            );
        }
    }
}
