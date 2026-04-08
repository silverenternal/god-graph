//! Transformer performance benchmarks
//!
//! This benchmark suite measures:
//! - INT8 quantization speedup
//! - SIMD acceleration for attention
//! - Memory pool efficiency
//! - Overall inference throughput

#![feature(test)]

extern crate test;

use god_graph::tensor::{DenseTensor, TensorBase, TensorOps};
use god_graph::transformer::{
    perf::{benchmark, matmul_with_buffer, softmax_inplace_simd, TransformerMemoryPool},
    quantization::{QuantizationConfig, QuantizedMatMul, QuantizedTensor},
    FeedForward, MultiHeadAttention, RMSNorm, RoPE,
};
use test::Bencher;

/// Benchmark INT8 quantization compression ratio
#[bench]
fn bench_int8_compression_ratio(b: &mut Bencher) {
    let tensor = DenseTensor::new(vec![0.0f64; 4096 * 4096], vec![4096, 4096]); // 16M elements ~ 128MB

    b.iter(|| {
        let quantized = QuantizedTensor::from_tensor(&tensor, QuantizationConfig::int8());
        let ratio = quantized.compression_ratio();
        assert!((ratio - 4.0).abs() < 0.1, "INT8 should be 4x smaller");
    });
}

/// Benchmark INT8 quantization speed
#[bench]
fn bench_int8_quantize_speed(b: &mut Bencher) {
    let tensor = DenseTensor::new(vec![1.0f64; 1024 * 1024], vec![1024, 1024]); // 1M elements

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

/// Benchmark SIMD softmax
#[bench]
fn bench_softmax_simd(b: &mut Bencher) {
    let mut data = vec![1.0f64; 512 * 512]; // [batch, seq_len]
    let shape = vec![512, 512];

    b.iter(|| {
        softmax_inplace_simd(&mut data, &shape, 1);
    });
}

/// Benchmark memory pool allocation
#[bench]
fn bench_memory_pool_allocation(b: &mut Bencher) {
    b.iter(|| {
        let mut pool = TransformerMemoryPool::new(4, 512, 4096, 32);

        // Simulate a forward pass
        let _qkv_buf = pool.get_qkv_buffer();
        let _attn_score_buf = pool.get_attn_score_buffer();
        let _attn_weight_buf = pool.get_attn_weight_buffer();
        let _output_buf = pool.get_output_buffer();
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

/// Benchmark multi-head attention forward pass
#[bench]
fn bench_multi_head_attention_forward(b: &mut Bencher) {
    let batch_size = 4;
    let seq_len = 128;
    let hidden_dim = 768;
    let num_heads = 12;

    // Initialize weights
    let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);

    let attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

    // Create input
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
    let intermediate_dim = 3072; // 4x hidden_dim

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

    // Attention
    let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

    // FFN
    let gate_proj = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
    let up_proj = DenseTensor::ones(vec![hidden_dim, intermediate_dim]);
    let down_proj = DenseTensor::ones(vec![intermediate_dim, hidden_dim]);
    let ffn = FeedForward::swiglu(gate_proj, up_proj, down_proj);

    // Norm
    let norm_weight = DenseTensor::ones(vec![hidden_dim]);
    let norm = RMSNorm::new(norm_weight, 1e-6);

    let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);

    b.iter(|| {
        // Pre-norm architecture (like LLaMA)
        let mut h = x.clone();

        // Attention block with residual
        let h_norm = norm.forward(&h);
        let attn_out = attn.forward(&h_norm);
        h = h.add(&attn_out);

        // FFN block with residual
        let h_norm = norm.forward(&h);
        let ffn_out = ffn.forward(&h_norm);
        h = h.add(&ffn_out);
    });
}

/// Benchmark batched inference throughput
#[bench]
fn bench_batched_inference_throughput(b: &mut Bencher) {
    let batch_size = 8;
    let seq_len = 256;
    let hidden_dim = 1024;
    let num_heads = 16;

    // Initialize model
    let w_q = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_k = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_v = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let w_o = DenseTensor::ones(vec![hidden_dim, hidden_dim]);
    let attn = MultiHeadAttention::standard(w_q, w_k, w_v, w_o, num_heads);

    let x = DenseTensor::ones(vec![batch_size, seq_len, hidden_dim]);

    b.iter(|| {
        let _output = attn.forward(&x);
    });

    // Calculate throughput
    let tokens_per_batch = batch_size * seq_len;
    benchmark::tokens_per_second(tokens_per_batch, 0.1); // Placeholder
}

/// Benchmark KV cache update efficiency
#[bench]
fn bench_kv_cache_update(b: &mut Bencher) {
    use god_graph::transformer::kv_cache::KVCache;

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

/// Benchmark end-to-end LLaMA forward pass
#[bench]
fn bench_llama_forward(b: &mut Bencher) {
    use god_graph::transformer::model::{LlamaConfig, LlamaModel};

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

    // Initialize with small weights for faster testing
    model.embed_tokens = DenseTensor::ones(vec![config.vocab_size, config.hidden_size]);

    let input_ids = vec![0usize; 4 * 64]; // batch_size=4, seq_len=64

    b.iter(|| {
        let _output = model.forward(&input_ids, 4, 64);
    });
}
