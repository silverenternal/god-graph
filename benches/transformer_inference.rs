//! Transformer inference performance benchmarks
//!
//! This benchmark suite measures:
//! - INT8/INT4 quantization speedup
//! - SIMD acceleration for attention and matmul
//! - Memory pool efficiency
//! - Batch inference throughput
//! - KV cache update efficiency
//! - End-to-end LLaMA forward pass
//!
//! ## Usage
//!
//! ```bash
//! # Run all benchmarks with SIMD
//! cargo bench --features "transformer,simd" --bench transformer_inference
//!
//! # Run specific benchmark
//! cargo bench --features "transformer,simd" --bench transformer_inference -- bench_simd_matmul
//!
//! # Compare SIMD vs naive
//! cargo bench --features transformer --bench transformer_inference  # naive
//! cargo bench --features "transformer,simd" --bench transformer_inference  # SIMD
//! ```

#![feature(test)]

extern crate test;

use god_graph::tensor::{DenseTensor, TensorBase, TensorOps};
use god_graph::transformer::{
    batch::{BatchData, BatchInference},
    kv_cache::KVCache,
    perf::{matmul_with_buffer, softmax_inplace_simd, TransformerMemoryPool},
    quantization::{QuantizationConfig, QuantizedMatMul, QuantizedTensor},
    FeedForward, LlamaConfig, LlamaModel, MultiHeadAttention, RMSNorm, RoPE,
};
use test::Bencher;

// ============================================================================
// Quantization Benchmarks
// ============================================================================

/// Benchmark INT8 quantization compression ratio
#[bench]
fn bench_int8_compression_ratio(b: &mut Bencher) {
    let tensor = DenseTensor::new(vec![0.0f64; 4096 * 4096], vec![4096, 4096]);

    b.iter(|| {
        let quantized = QuantizedTensor::from_tensor(&tensor, QuantizationConfig::int8());
        let ratio = quantized.compression_ratio();
        assert!((ratio - 4.0).abs() < 0.1, "INT8 should be 4x smaller");
    });
}

/// Benchmark INT8 quantization speed
#[bench]
fn bench_int8_quantize_speed(b: &mut Bencher) {
    let tensor = DenseTensor::new(vec![1.0f64; 1024 * 1024], vec![1024, 1024]);

    b.iter(|| {
        let _quantized = QuantizedTensor::from_tensor(&tensor, QuantizationConfig::int8());
    });
}

/// Benchmark INT8 dequantization speed
#[bench]
fn bench_int8_dequantize_speed(b: &mut Bencher) {
    let tensor = DenseTensor::new(vec![1.0f64; 1024 * 1024], vec![1024, 1024]);
    let quantized = QuantizedTensor::from_tensor(&tensor, QuantizationConfig::int8());

    b.iter(|| {
        let _dequantized = quantized.dequantize();
    });
}

/// Benchmark INT8 GEMM vs dense GEMM
#[bench]
fn bench_int8_gemm(b: &mut Bencher) {
    let a = DenseTensor::new(vec![1.0f64; 512 * 1024], vec![512, 1024]);
    let b = DenseTensor::new(vec![0.5f64; 1024 * 512], vec![1024, 512]);

    let a_q = QuantizedTensor::from_tensor(&a, QuantizationConfig::int8());
    let b_q = QuantizedTensor::from_tensor(&b, QuantizationConfig::int8());

    b.iter(|| {
        let _result = QuantizedMatMul::gemm_int8(&a_q, &b_q);
    });
}

/// Benchmark dense GEMM for comparison
#[bench]
fn bench_dense_gemm(b: &mut Bencher) {
    let a = DenseTensor::new(vec![1.0f64; 512 * 1024], vec![512, 1024]);
    let b = DenseTensor::new(vec![0.5f64; 1024 * 512], vec![1024, 512]);

    b.iter(|| {
        let _result = a.matmul(&b);
    });
}

/// Benchmark INT4 quantization (2-bit per weight)
#[bench]
fn bench_int4_quantize_speed(b: &mut Bencher) {
    let tensor = DenseTensor::new(vec![1.0f64; 1024 * 1024], vec![1024, 1024]);

    b.iter(|| {
        let _quantized = QuantizedTensor::from_tensor(&tensor, QuantizationConfig::int4());
    });
}

// ============================================================================
// SIMD Benchmarks
// ============================================================================

/// Benchmark SIMD softmax
#[bench]
fn bench_softmax_simd(b: &mut Bencher) {
    let mut data = vec![1.0f64; 512 * 512];
    let shape = vec![512, 512];

    b.iter(|| {
        softmax_inplace_simd(&mut data, &shape, 1);
    });
}

/// Benchmark SIMD matrix multiplication
#[bench]
fn bench_simd_matmul(b: &mut Bencher) {
    let a = DenseTensor::new(vec![1.0f64; 256 * 512], vec![256, 512]);
    let b = DenseTensor::new(vec![0.5f64; 512 * 256], vec![512, 256]);
    let mut buffer = vec![0.0f64; 256 * 256];

    b.iter(|| {
        let _result = matmul_with_buffer(&a, &b, &mut buffer);
    });
}

/// Benchmark naive matrix multiplication (for comparison)
#[bench]
fn bench_naive_matmul(b: &mut Bencher) {
    let a = DenseTensor::new(vec![1.0f64; 256 * 512], vec![256, 512]);
    let b = DenseTensor::new(vec![0.5f64; 512 * 256], vec![512, 256]);

    b.iter(|| {
        let _result = a.matmul(&b);
    });
}

// ============================================================================
// Memory Pool Benchmarks
// ============================================================================

/// Benchmark memory pool allocation
#[bench]
fn bench_memory_pool_allocation(b: &mut Bencher) {
    b.iter(|| {
        let mut pool = TransformerMemoryPool::new(4, 512, 4096, 32);

        let _qkv_buf = pool.get_qkv_buffer();
        let _attn_score_buf = pool.get_attn_score_buffer();
        let _attn_weight_buf = pool.get_attn_weight_buffer();
        let _output_buf = pool.get_output_buffer();
    });
}

/// Benchmark traditional allocation (for comparison)
#[bench]
fn bench_traditional_allocation(b: &mut Bencher) {
    b.iter(|| {
        let _qkv = vec![0.0f64; 4 * 512 * 4096];
        let _attn_scores = vec![0.0f64; 4 * 32 * 512 * 512];
        let _attn_weights = vec![0.0f64; 4 * 32 * 512 * 512];
        let _output = vec![0.0f64; 4 * 512 * 4096];
    });
}

/// Benchmark matmul with buffer reuse
#[bench]
fn bench_matmul_with_buffer(b: &mut Bencher) {
    let a = DenseTensor::new(vec![1.0f64; 256 * 512], vec![256, 512]);
    let b = DenseTensor::new(vec![0.5f64; 512 * 256], vec![512, 256]);
    let mut buffer = vec![0.0f64; 256 * 256];

    b.iter(|| {
        let _result = matmul_with_buffer(&a, &b, &mut buffer);
    });
}

// ============================================================================
// Component Benchmarks
// ============================================================================

/// Benchmark multi-head attention forward pass
#[bench]
fn bench_multi_head_attention_forward(b: &mut Bencher) {
    let batch_size = 4;
    let seq_len = 128;
    let hidden_dim = 768;
    let num_heads = 12;

    let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);

    let attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);
    let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);

    b.iter(|| {
        let _output = attn.forward(&x);
    });
}

/// Benchmark RMSNorm forward pass
#[bench]
fn bench_rmsnorm_forward(b: &mut Bencher) {
    let batch_size = 4;
    let seq_len = 128;
    let hidden_dim = 768;

    let weight = DenseTensor::ones(vec![hidden_dim]);
    let norm = RMSNorm::new(weight, 1e-6);
    let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);

    b.iter(|| {
        let _output = norm.forward(&x);
    });
}

/// Benchmark RoPE (Rotary Positional Embedding)
#[bench]
fn bench_rope_apply(b: &mut Bencher) {
    let batch_size = 4;
    let seq_len = 128;
    let num_heads = 12;
    let head_dim = 64;

    let rope = RoPE::new(head_dim, 512, 10000.0);
    let x = DenseTensor::ones(vec![batch_size, num_heads, seq_len, head_dim]);

    b.iter(|| {
        let _rotated = rope.apply(&x, 0);
    });
}

/// Benchmark feed-forward network (SwiGLU)
#[bench]
fn bench_ffn_swiglu_forward(b: &mut Bencher) {
    let batch_size = 4;
    let seq_len = 128;
    let hidden_dim = 768;
    let intermediate_dim = 3072;

    let gate_proj = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
    let up_proj = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
    let down_proj = DenseTensor::ones(vec![intermediate_dim, hidden_dim]);

    let ffn = FeedForward::swiglu(gate_proj, up_proj, down_proj);
    let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);

    b.iter(|| {
        let _output = ffn.forward(&x);
    });
}

/// Benchmark full transformer layer (Attention + FFN)
#[bench]
fn bench_transformer_layer_forward(b: &mut Bencher) {
    let batch_size = 2;
    let seq_len = 64;
    let hidden_dim = 512;
    let num_heads = 8;
    let intermediate_dim = 2048;

    let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

    let gate_proj = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
    let up_proj = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
    let down_proj = DenseTensor::ones(vec![intermediate_dim, hidden_dim]);
    let ffn = FeedForward::swiglu(gate_proj, up_proj, down_proj);

    let norm_weight = DenseTensor::ones(vec![hidden_dim]);
    let norm = RMSNorm::new(norm_weight, 1e-6);
    let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);

    b.iter(|| {
        let mut h = x.clone();
        let h_norm = norm.forward(&h);
        let attn_out = attn.forward(&h_norm);
        h = h.add(&attn_out);
        let h_norm = norm.forward(&h);
        let ffn_out = ffn.forward(&h_norm);
        h = h.add(&ffn_out);
    });
}

// ============================================================================
// Batch Inference Benchmarks
// ============================================================================

/// Benchmark batched inference throughput
#[bench]
fn bench_batched_inference_throughput(b: &mut Bencher) {
    let batch_size = 8;
    let seq_len = 256;
    let hidden_dim = 1024;
    let num_heads = 16;

    let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

    let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);

    b.iter(|| {
        let _output = attn.forward(&x);
    });
}

/// Benchmark KV cache update efficiency
#[bench]
fn bench_kv_cache_update(b: &mut Bencher) {
    let num_layers = 12;
    let max_seq_len = 512;
    let hidden_dim = 768;
    let num_kv_heads = 12;

    let mut cache = KVCache::new(num_layers, max_seq_len, hidden_dim, num_kv_heads);
    let k = DenseTensor::ones(vec![1, 1, hidden_dim]);
    let v = DenseTensor::ones(vec![1, 1, hidden_dim]);

    b.iter(|| {
        cache.update(0, &k, &v);
    });
}

/// Benchmark batch data creation
#[bench]
fn bench_batch_data_creation(b: &mut Bencher) {
    let batch_size = 8;
    let seq_len = 128;

    b.iter(|| {
        let input_ids: Vec<Vec<usize>> = (0..batch_size)
            .map(|i| {
                (0..seq_len)
                    .map(|j| ((i * seq_len + j) % 100) as usize)
                    .collect()
            })
            .collect();
        let _batch = BatchData::new(input_ids);
    });
}

// ============================================================================
// End-to-End Model Benchmarks
// ============================================================================

/// Benchmark end-to-end LLaMA forward pass
#[bench]
fn bench_llama_forward(b: &mut Bencher) {
    let config = LlamaConfig {
        vocab_size: 1000,
        hidden_size: 512,
        intermediate_size: 2048,
        num_hidden_layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 8,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
    };

    let mut model = LlamaModel::new(&config);
    model.embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);

    let input_ids = vec![0usize; 4 * 64];

    b.iter(|| {
        let _output = model.forward(&input_ids, 4, 64);
    });
}

/// Benchmark LLaMA forward pass with batch
#[bench]
fn bench_llama_batch_forward(b: &mut Bencher) {
    let config = LlamaConfig {
        vocab_size: 1000,
        hidden_size: 512,
        intermediate_size: 2048,
        num_hidden_layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 8,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
    };

    let mut model = LlamaModel::new(&config);
    model.embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);

    let batch_size = 8;
    let seq_len = 64;
    let input_ids: Vec<Vec<usize>> = (0..batch_size)
        .map(|_| (0..seq_len).map(|_| 0).collect())
        .collect();
    let batch = BatchData::new(input_ids);

    let mut batch_infer = BatchInference::new(&model, batch_size, 512);

    b.iter(|| {
        let _logits = batch_infer.forward(&batch);
    });
}

/// Benchmark LLaMA forward pass with KV cache (autoregressive)
#[bench]
fn bench_llama_autoregressive(b: &mut Bencher) {
    let config = LlamaConfig {
        vocab_size: 1000,
        hidden_size: 256,
        intermediate_size: 512,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
    };

    let mut model = LlamaModel::new(&config);
    model.embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);

    b.iter(|| {
        // Simulate autoregressive generation
        for seq_len in 1..10 {
            let input_ids = vec![vec![0usize; seq_len]];
            let _output = model.forward(&input_ids, 1, seq_len);
        }
    });
}

// ============================================================================
// Quantization vs Dense Comparison
// ============================================================================

/// Benchmark INT8 attention forward (quantized weights)
#[bench]
fn bench_int8_attention_forward(b: &mut Bencher) {
    let batch_size = 4;
    let seq_len = 128;
    let hidden_dim = 768;
    let num_heads = 12;

    let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);

    // Quantize weights
    let w_q_q = QuantizedTensor::from_tensor(&w_q, QuantizationConfig::int8());
    let w_k_q = QuantizedTensor::from_tensor(&w_k, QuantizationConfig::int8());
    let w_v_q = QuantizedTensor::from_tensor(&w_v, QuantizationConfig::int8());
    let w_o_q = QuantizedTensor::from_tensor(&w_o, QuantizationConfig::int8());

    let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);

    b.iter(|| {
        // Quantized attention forward
        let q = QuantizedMatMul::gemm_int8(
            &QuantizedTensor::from_tensor(&x, QuantizationConfig::int8()),
            &w_q_q,
        );
        let _k = QuantizedMatMul::gemm_int8(
            &QuantizedTensor::from_tensor(&x, QuantizationConfig::int8()),
            &w_k_q,
        );
        let _v = QuantizedMatMul::gemm_int8(
            &QuantizedTensor::from_tensor(&x, QuantizationConfig::int8()),
            &w_v_q,
        );
        let _out = QuantizedMatMul::gemm_int8(&q, &w_o_q);
    });
}

/// Benchmark dense attention forward (for comparison)
#[bench]
fn bench_dense_attention_forward(b: &mut Bencher) {
    let batch_size = 4;
    let seq_len = 128;
    let hidden_dim = 768;
    let num_heads = 12;

    let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);

    let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);

    b.iter(|| {
        let _q = x.matmul(&w_q);
        let _k = x.matmul(&w_k);
        let _v = x.matmul(&w_v);
    });
}
