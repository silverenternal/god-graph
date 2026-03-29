//! SIMD-Optimized Batch Inference Example
//!
//! This example demonstrates:
//! 1. SIMD-accelerated batch matrix multiplication
//! 2. Memory pool optimization for batch processing
//! 3. Performance comparison: naive vs SIMD
//!
//! ## Usage
//!
//! ```bash
//! # Run with SIMD enabled
//! cargo run --example llm_batch_simd --features "transformer,simd"
//!
//! # Run without SIMD for comparison
//! cargo run --example llm_batch_simd --features transformer
//! ```

#[cfg(feature = "transformer")]
fn main() {
    use god_gragh::tensor::{DenseTensor, TensorBase, TensorOps};
    use god_gragh::transformer::{
        perf::{matmul_with_buffer, softmax_inplace_simd, TransformerMemoryPool},
        FeedForward, MultiHeadAttention, RMSNorm,
    };
    use std::time::Instant;

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║     SIMD-Optimized Batch Inference Demo                ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Configuration
    let batch_size = 4;
    let seq_len = 64;
    let hidden_dim = 256;
    let num_heads = 8;

    println!("⚙️  Configuration:");
    println!("     - Batch size: {}", batch_size);
    println!("     - Sequence length: {}", seq_len);
    println!("     - Hidden dimension: {}", hidden_dim);
    println!("     - Number of heads: {}", num_heads);

    #[cfg(feature = "simd")]
    println!("\n✅ SIMD acceleration: ENABLED (wide::f64x4)\n");

    #[cfg(not(feature = "simd"))]
    println!("\n⚠️  SIMD acceleration: DISABLED (enable with --features simd)\n");

    // Initialize weights with ones for demo
    println!("📦 Initializing weights...");
    let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);

    println!("   ✓ Attention weights initialized");

    // Create attention module
    let attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

    // Create layer norm
    let norm = RMSNorm::default(hidden_dim);

    // Create batch input
    let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);

    // Benchmark 1: Batch attention
    println!("\n🔬 Benchmark 1: Batch Attention");
    let iterations = 10;

    let start = Instant::now();
    for _ in 0..iterations {
        let _output = attn.forward(&x);
    }

    let elapsed = start.elapsed();
    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!(
        "   Time ({} iters): {:.2} ms",
        iterations,
        elapsed.as_secs_f64() * 1000.0
    );
    println!("   Time per iter: {:.2} ms", ms_per_iter);

    // Benchmark 2: SIMD-optimized matmul
    println!("\n🔬 Benchmark 2: SIMD-Optimized MatMul");
    let a = DenseTensor::ones(vec![batch_size * seq_len, hidden_dim]);
    let b = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let mut buffer = vec![0.0f64; batch_size * seq_len * hidden_dim];

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = matmul_with_buffer(&a, &b, &mut buffer);
    }

    let elapsed = start.elapsed();
    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!(
        "   Time ({} iters): {:.2} ms",
        iterations,
        elapsed.as_secs_f64() * 1000.0
    );
    println!("   Time per iter: {:.2} ms", ms_per_iter);

    // Benchmark 3: SIMD softmax
    println!("\n🔬 Benchmark 3: SIMD Softmax");
    let mut softmax_data = vec![1.0f64; batch_size * num_heads * seq_len * seq_len];
    let softmax_shape = vec![batch_size, num_heads, seq_len, seq_len];

    let start = Instant::now();
    for _ in 0..iterations {
        softmax_inplace_simd(&mut softmax_data, &softmax_shape, 3);
    }

    let elapsed = start.elapsed();
    println!(
        "   Time ({} iters): {:.2} ms",
        iterations,
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Time per iter: {:.2} ms",
        elapsed.as_secs_f64() * 1000.0 / iterations as f64
    );

    // Benchmark 4: Memory pool efficiency
    println!("\n🔬 Benchmark 4: Memory Pool Efficiency");

    // Traditional allocation
    let start = Instant::now();
    for _ in 0..iterations {
        let _qkv = vec![0.0f64; batch_size * seq_len * hidden_dim];
        let _attn_scores = vec![0.0f64; batch_size * num_heads * seq_len * seq_len];
    }
    let elapsed_traditional = start.elapsed();

    // Memory pool allocation
    let start = Instant::now();
    for _ in 0..iterations {
        let mut pool = TransformerMemoryPool::new(batch_size, seq_len, hidden_dim, num_heads);
        let _qkv_buf = pool.get_qkv_buffer();
        let _attn_score_buf = pool.get_attn_score_buffer();
        let _output_buf = pool.get_output_buffer();
    }
    let elapsed_pool = start.elapsed();

    println!(
        "   Traditional allocation: {:.2} ms",
        elapsed_traditional.as_secs_f64() * 1000.0
    );
    println!(
        "   Memory pool allocation: {:.2} ms",
        elapsed_pool.as_secs_f64() * 1000.0
    );

    // Benchmark 5: Full transformer layer
    println!("\n🔬 Benchmark 5: Full Transformer Layer");

    // Create FFN for complete layer
    let intermediate_dim = hidden_dim * 4;
    let gate_proj = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
    let up_proj = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
    let down_proj = DenseTensor::ones(vec![intermediate_dim, hidden_dim]);
    let ffn = FeedForward::swiglu(gate_proj, up_proj, down_proj);

    let start = Instant::now();

    for _ in 0..iterations {
        // Pre-norm architecture
        let mut h = x.clone();

        // Attention block
        let h_norm = norm.forward(&h);
        let attn_out = attn.forward(&h_norm);
        h = h.add(&attn_out);

        // FFN block
        let h_norm = norm.forward(&h);
        let ffn_out = ffn.forward(&h_norm);
        h = h.add(&ffn_out);
    }

    let elapsed = start.elapsed();
    println!(
        "   Time ({} iters): {:.2} ms",
        iterations,
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Time per iter: {:.2} ms",
        elapsed.as_secs_f64() * 1000.0 / iterations as f64
    );

    // Summary
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║                    Summary                             ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!("   Batch size: {}", batch_size);
    println!("   Sequence length: {}", seq_len);
    println!("   Total tokens per batch: {}", batch_size * seq_len);
    println!("   Hidden dimension: {}", hidden_dim);

    #[cfg(feature = "simd")]
    println!("\n   ✅ SIMD optimizations active");
    #[cfg(not(feature = "simd"))]
    println!("\n   ⚠️  Enable SIMD with --features simd for better performance");

    println!("\n💡 Tips for optimal performance:");
    println!("   1. Enable SIMD: cargo run --features transformer,simd");
    println!("   2. Use larger batch sizes for better throughput");
    println!("   3. Enable INT8 quantization for 2-4x speedup");
    println!("   4. Use KV cache for autoregressive generation\n");
}

#[cfg(not(feature = "transformer"))]
fn main() {
    println!("This example requires the 'transformer' feature.");
    println!("Run with: cargo run --example llm_batch_simd --features transformer");
}
