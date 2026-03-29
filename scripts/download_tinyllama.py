#!/usr/bin/env python3
"""
Download TinyLlama model from HuggingFace

This script downloads the TinyLlama-1.1B model weights from HuggingFace Hub.
The downloaded model will be stored in the models/ directory.

Usage:
    python scripts/download_tinyllama.py

Requirements:
    pip install huggingface_hub
"""

import os
import sys
from huggingface_hub import hf_hub_download

def download_tinyllama():
    """Download TinyLlama model files"""
    
    # Model repository on HuggingFace
    repo_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-5T"
    
    # Files to download
    files_to_download = [
        "config.json",
        "model.safetensors",
    ]
    
    # Local directory to store models
    local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading TinyLlama model from {repo_id}")
    print(f"Local directory: {local_dir}")
    print()
    
    downloaded_files = []
    
    for filename in files_to_download:
        print(f"Downloading {filename}...")
        try:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            downloaded_files.append(file_path)
            print(f"  ✓ Downloaded to: {file_path}")
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
            print("  Make sure you have huggingface_hub installed: pip install huggingface_hub")
            sys.exit(1)
    
    print()
    print("Download complete!")
    print(f"Downloaded {len(downloaded_files)} files:")
    for f in downloaded_files:
        print(f"  - {f}")
    
    return downloaded_files

if __name__ == "__main__":
    download_tinyllama()
