#!/bin/bash
# Download TinyLlama model from HuggingFace
# This script downloads the TinyLlama-1.1B model weights for validation testing
#
# Note: Some models require authentication. If you encounter authentication errors,
# please:
# 1. Install huggingface-cli: pip install huggingface_hub
# 2. Login: huggingface-cli login
# 3. Or use the synthetic model generator for testing

set -e

MODEL_DIR="${1:-models/tinyllama}"

# Function to generate synthetic TinyLlama-like model
generate_synthetic_model() {
    local target_dir="$1"
    echo "Generating synthetic TinyLlama-like model in $target_dir..."
    
    # Use Python to generate config
    python3 << PYTHON_SCRIPT
import json
import struct
import os
import random

def generate_synthetic_model(output_dir):
    """Generate synthetic TinyLlama-like weights for testing."""
    
    # TinyLlama-1.1B config (simplified)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_attention_heads": 32,
        "num_hidden_layers": 22,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "model_type": "llama"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated config.json with {config['num_hidden_layers']} layers")
    
    # Create a minimal safetensors file for testing the loading infrastructure
    create_minimal_safetensors(f"{output_dir}/model.safetensors")
    print("Note: Synthetic model generated for testing purposes")
    print("For real model validation, please use huggingface-cli with authentication")

def create_minimal_safetensors(output_path):
    """Create a minimal valid safetensors file for testing."""
    
    # Create multiple test tensors to simulate a real model structure
    tensors = {
        "model.embed_tokens.weight": {"shape": [100, 64], "dtype": "F32"},
        "model.layers.0.self_attn.q_proj.weight": {"shape": [64, 64], "dtype": "F32"},
        "model.layers.0.self_attn.k_proj.weight": {"shape": [64, 64], "dtype": "F32"},
        "model.layers.0.self_attn.v_proj.weight": {"shape": [64, 64], "dtype": "F32"},
        "model.layers.0.self_attn.o_proj.weight": {"shape": [64, 64], "dtype": "F32"},
        "model.layers.0.mlp.gate_proj.weight": {"shape": [128, 64], "dtype": "F32"},
        "model.layers.0.mlp.up_proj.weight": {"shape": [128, 64], "dtype": "F32"},
        "model.layers.0.mlp.down_proj.weight": {"shape": [64, 128], "dtype": "F32"},
        "model.norm.weight": {"shape": [64], "dtype": "F32"},
    }
    
    # Build header and data
    header = {}
    all_data = b''
    
    random.seed(42)
    
    for tensor_name, info in tensors.items():
        shape = info["shape"]
        dtype = info["dtype"]
        
        # Generate random data
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        
        data_list = [random.gauss(0, 0.1) for _ in range(num_elements)]
        
        # Pack data as f32
        data_bytes = struct.pack(f'{len(data_list)}f', *data_list)
        
        # Record offsets
        start_offset = len(all_data)
        header[tensor_name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start_offset, start_offset + len(data_bytes)]
        }
        
        all_data += data_bytes
    
    header_json = json.dumps(header).encode('utf-8')
    header_len = len(header_json)
    
    # Write file: 8 bytes header length + header JSON + data
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', header_len))
        f.write(header_json)
        f.write(all_data)
    
    print(f"Created safetensors file: {output_path}")
    print(f"  Tensors: {len(tensors)}, total size={len(all_data)} bytes")

if __name__ == "__main__":
    output_dir = "${target_dir}"
    generate_synthetic_model(output_dir)
PYTHON_SCRIPT
}

echo "Downloading TinyLlama model to: $MODEL_DIR"

# Create directory
mkdir -p "$MODEL_DIR"

# Check if huggingface-cli is available
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli to download model..."
    huggingface-cli download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-5T \
        --include "model.safetensors" \
        --include "config.json" \
        --local-dir "$MODEL_DIR"
    
    # Verify download
    if [ -f "$MODEL_DIR/model.safetensors" ] && [ -s "$MODEL_DIR/model.safetensors" ]; then
        echo "Model downloaded successfully!"
    else
        echo "Download failed or incomplete. Falling back to synthetic model generator..."
        generate_synthetic_model "$MODEL_DIR"
    fi
else
    echo "huggingface-cli not found."
    echo ""
    echo "To download the real model, please:"
    echo "  1. Install huggingface_hub: pip install huggingface_hub"
    echo "  2. Login: huggingface-cli login"
    echo "  3. Re-run this script"
    echo ""
    generate_synthetic_model "$MODEL_DIR"
fi

echo ""
echo "Files in $MODEL_DIR:"
ls -lh "$MODEL_DIR"
