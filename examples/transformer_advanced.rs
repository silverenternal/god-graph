//! Advanced Transformer Features Example
//!
//! This example demonstrates advanced transformer features including:
//! - KV Cache for efficient inference
//! - Sparse attention patterns
//! - Quantization for memory efficiency

#[cfg(feature = "transformer")]
fn main() {
    use god_graph::tensor::traits::TensorBase;
    use god_graph::tensor::DenseTensor;
    use god_graph::transformer::quantization::{QuantizationConfig, QuantizedTensor};
    use god_graph::transformer::sparse_attention::SparseMask;

    println!("=== Advanced Transformer Features Example ===\n");

    // Feature 1: Sparse Attention
    println!("=== Feature 1: Sparse Attention ===");
    let seq_len = 128;

    // Sliding window attention (used in Mistral)
    let window_size = 16;
    let sliding_mask = SparseMask::sliding_window(seq_len, window_size, true);
    println!("Sliding Window Attention:");
    println!("  Sequence length: {}", seq_len);
    println!("  Window size: {}", window_size);
    println!(
        "  Total non-zero blocks: {}",
        sliding_mask.col_indices.len()
    );
    println!(
        "  Sparsity: {:.1}% non-zero",
        100.0 * sliding_mask.col_indices.len() as f64 / (seq_len * seq_len) as f64
    );

    // Block sparse attention
    let block_size = 16;
    let num_blocks = 2;
    let block_sparse_mask = SparseMask::block_sparse(seq_len, block_size, num_blocks);
    println!("\nBlock Sparse Attention:");
    println!("  Block size: {}", block_size);
    println!("  Num blocks to attend: {}", num_blocks);
    println!(
        "  Total non-zero blocks: {}",
        block_sparse_mask.col_indices.len()
    );
    println!(
        "  Sparsity: {:.1}% non-zero",
        100.0 * block_sparse_mask.col_indices.len() as f64 / (seq_len * seq_len) as f64
    );
    println!();

    // Feature 2: Quantization
    println!("=== Feature 2: Quantization (INT8) ===");

    // Create a sample weight tensor
    let weight_data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.01).sin()).collect();
    let weight_tensor = DenseTensor::new(weight_data.clone(), vec![16, 16]);
    println!("Original weight tensor:");
    println!("  Shape: {:?}", weight_tensor.shape());
    println!("  Data size: {} bytes", weight_tensor.data().len() * 8); // f64 = 8 bytes

    // Quantize to INT8 (per-tensor)
    let config = QuantizationConfig::int8();
    let quantized = QuantizedTensor::from_tensor(&weight_tensor, config);
    println!("\nQuantized weight tensor (INT8):");
    println!("  Shape: {:?}", quantized.shape);
    println!("  Data size: {} bytes", quantized.memory_bytes()); // i8 = 1 byte
    println!("  Compression ratio: {:.2}x", quantized.compression_ratio());

    // Dequantize and check reconstruction
    let dequantized = quantized.dequantize();
    println!("\nDequantized weight tensor:");
    println!("  Shape: {:?}", dequantized.shape());

    // Calculate reconstruction error
    let orig_data = weight_tensor.data();
    let recon_data = dequantized.data();
    let mse: f64 = orig_data
        .iter()
        .zip(recon_data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / orig_data.len() as f64;

    println!("  Mean Squared Error: {:.6}", mse);
    println!();

    // Feature 3: Per-channel quantization
    println!("=== Feature 3: Per-Channel Quantization ===");
    let config_per_channel = QuantizationConfig::per_channel_int8(0);
    let quantized_pc = QuantizedTensor::from_tensor(&weight_tensor, config_per_channel);

    println!("Per-channel INT8 quantization:");
    println!("  Quantization axis: 0");
    println!(
        "  Compression ratio: {:.2}x",
        quantized_pc.compression_ratio()
    );

    let dequantized_pc = quantized_pc.dequantize();
    let recon_data_pc = dequantized_pc.data();
    let mse_pc: f64 = orig_data
        .iter()
        .zip(recon_data_pc.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / orig_data.len() as f64;

    println!("  Mean Squared Error: {:.6}", mse_pc);
    println!();

    println!("=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("- Sparse attention reduces memory and computation");
    println!("  - Sliding window: O(seq_len * window_size) instead of O(seq_len^2)");
    println!("  - Block sparse: Attend to local and selected global tokens");
    println!("- INT8 quantization achieves ~8x compression with minimal accuracy loss");
    println!("- Per-channel quantization provides better accuracy than per-tensor");
    println!("  - Per-tensor MSE: {:.6}", mse);
    println!("  - Per-channel MSE: {:.6} (typically lower)", mse_pc);
}

#[cfg(not(feature = "transformer"))]
fn main() {
    println!("This example requires the 'transformer' feature to be enabled.");
    println!("Run with: cargo run --example transformer_advanced --features transformer,safetensors,tensor-pool");
}
