//! Transformer Features Overview Example
//!
//! This example provides an overview of transformer features available in god-gragh.
//! For detailed usage, see the individual feature examples.

#[cfg(feature = "transformer")]
fn main() {
    use god_graph::tensor::DenseTensor;
    use god_graph::transformer::batch::BatchData;
    use god_graph::transformer::kv_cache::KVCache;
    use god_graph::transformer::quantization::{QuantizationConfig, QuantizedTensor};
    use god_graph::transformer::sparse_attention::SparseMask;

    println!("=== God-Graph Transformer Features Overview ===\n");

    // Feature 1: KV Cache
    println!("1. KV Cache for Efficient Inference");
    println!("   --------------------------------");
    let kv_cache = KVCache::new(2, 512, 8, 64);
    println!(
        "   Created KV Cache with capacity: {} tokens",
        kv_cache.max_seq_len()
    );
    println!("   Current usage: {} tokens\n", kv_cache.current_len());

    // Feature 2: Sparse Attention
    println!("2. Sparse Attention Patterns");
    println!("   -------------------------");
    let seq_len = 128;
    let window_size = 16;
    let sliding_mask = SparseMask::sliding_window(seq_len, window_size, true);
    println!("   Sliding Window Attention:");
    println!("     - Sequence length: {}", seq_len);
    println!("     - Window size: {}", window_size);
    println!(
        "     - Sparsity: {:.1}% non-zero",
        100.0 * sliding_mask.col_indices.len() as f64 / (seq_len * seq_len) as f64
    );

    let block_sparse = SparseMask::block_sparse(seq_len, 16, 2);
    println!("\n   Block Sparse Attention:");
    println!("     - Block size: 16");
    println!("     - Blocks to attend: 2");
    println!(
        "     - Sparsity: {:.1}% non-zero\n",
        100.0 * block_sparse.col_indices.len() as f64 / (seq_len * seq_len) as f64
    );

    // Feature 3: Quantization
    println!("3. Weight Quantization");
    println!("   -------------------");
    let weight_data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.01).sin()).collect();
    let weight = DenseTensor::new(weight_data, vec![16, 16]);

    let q_config = QuantizationConfig::int8();
    let quantized = QuantizedTensor::from_tensor(&weight, q_config);

    println!("   INT8 Per-Tensor Quantization:");
    println!("     - Original size: {} bytes", weight.data().len() * 8);
    println!("     - Quantized size: {} bytes", quantized.memory_bytes());
    println!(
        "     - Compression: {:.1}x\n",
        quantized.compression_ratio()
    );

    // Feature 4: Batch Processing
    println!("4. Batch Processing");
    println!("   ----------------");
    let sequences = vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8], vec![9, 10, 11, 12]];
    let batch = BatchData::new(sequences);
    println!("   Batch created with {} sequences", batch.batch_size());
    println!("   Max sequence length: {}", batch.max_seq_len());
    println!(
        "   Padded input shape: [{} x {}]",
        batch.batch_size(),
        batch.max_seq_len()
    );

    println!("\n=== Feature Overview Complete ===");
    println!("\nFor detailed examples, run:");
    println!(
        "  cargo run --example transformer_advanced --features transformer,safetensors,tensor-pool"
    );
}

#[cfg(not(feature = "transformer"))]
fn main() {
    println!("This example requires the 'transformer' feature.");
    println!("Run with: cargo run --example transformer_overview --features transformer");
}
