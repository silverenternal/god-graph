//! Safetensors format loader
//!
//! Safetensors is a safe tensor format developed by HuggingFace for storing deep learning weights.
//! Format structure:
//! - First 8 bytes: header length (little endian u64)
//! - Header: JSON string containing tensor metadata (name, dtype, shape, offsets)
//! - Data: Binary tensor data arranged according to header offsets

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use crate::tensor::DenseTensor;
use crate::errors::{GraphError, GraphResult};

/// Supported data types in Safetensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (half precision)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
}

impl Dtype {
    /// Get the size in bytes for a dtype
    pub fn size(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
            Dtype::BF16 => 2,
            Dtype::I32 => 4,
            Dtype::I64 => 8,
        }
    }

    /// Parse dtype from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "F32" | "f32" | "FLOAT32" => Some(Dtype::F32),
            "F16" | "f16" | "FLOAT16" => Some(Dtype::F16),
            "BF16" | "bf16" | "BFLOAT16" => Some(Dtype::BF16),
            "I32" | "i32" | "INT32" => Some(Dtype::I32),
            "I64" | "i64" | "INT64" => Some(Dtype::I64),
            _ => None,
        }
    }
}

/// Tensor metadata from Safetensors header
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Data type
    pub dtype: Dtype,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Byte offsets in the file [start, end)
    pub offsets: [usize; 2],
}

/// Safetensors loader
#[derive(Debug)]
pub struct SafetensorsLoader {
    /// Tensor metadata
    tensors: HashMap<String, TensorInfo>,
    /// File path (stored for reference)
    #[allow(dead_code)]
    path: String,
    /// File handle (kept open for lazy loading)
    file: Option<File>,
}

impl SafetensorsLoader {
    /// Load a Safetensors file
    ///
    /// # Arguments
    /// * `path` - Path to the .safetensors file
    ///
    /// # Returns
    /// A loader that can be used to access individual tensors
    pub fn load<P: AsRef<Path>>(path: P) -> GraphResult<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)
            .map_err(|e| GraphError::IoError(e.to_string()))?;

        // Read first 8 bytes to get header length
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)
            .map_err(|e| GraphError::IoError(e.to_string()))?;

        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        // Read header JSON
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .map_err(|e| GraphError::IoError(e.to_string()))?;
        
        let header_str = String::from_utf8(header_bytes)
            .map_err(|e| GraphError::InvalidFormat(e.to_string()))?;

        // Parse header JSON
        let header: serde_json::Value = serde_json::from_str(&header_str)
            .map_err(|e| GraphError::InvalidFormat(e.to_string()))?;

        // Parse tensor metadata
        let mut tensors = HashMap::new();

        if let Some(obj) = header.as_object() {
            for (name, value) in obj {
                if name == "__metadata__" {
                    continue; // Skip metadata
                }

                if let Some(tensor_info) = value.as_object() {
                    let dtype_str = tensor_info.get("dtype")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| GraphError::InvalidFormat(format!("Missing dtype for tensor {}", name)))?;

                    let dtype = Dtype::from_str(dtype_str)
                        .ok_or_else(|| GraphError::InvalidFormat(format!("Unknown dtype: {}", dtype_str)))?;

                    let shape = tensor_info.get("shape")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| GraphError::InvalidFormat(format!("Missing shape for tensor {}", name)))?;

                    let shape: Vec<usize> = shape.iter()
                        .map(|v| v.as_u64().map(|x| x as usize).ok_or_else(|| GraphError::InvalidFormat("Invalid shape value".to_string())))
                        .collect::<Result<_, _>>()?;

                    let data_offsets = tensor_info.get("data_offsets")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| GraphError::InvalidFormat(format!("Missing data_offsets for tensor {}", name)))?;

                    if data_offsets.len() != 2 {
                        return Err(GraphError::InvalidFormat(format!("Invalid data_offsets for tensor {}", name)));
                    }

                    let offsets = [
                        data_offsets[0].as_u64().ok_or_else(|| GraphError::InvalidFormat("Invalid offset".to_string()))? as usize,
                        data_offsets[1].as_u64().ok_or_else(|| GraphError::InvalidFormat("Invalid offset".to_string()))? as usize,
                    ];

                    let info = TensorInfo {
                        dtype,
                        shape,
                        offsets,
                    };

                    tensors.insert(name.clone(), info);
                }
            }
        }
        
        Ok(Self {
            tensors,
            path: path.to_string_lossy().to_string(),
            file: Some(file),
        })
    }

    /// Get tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Get tensor info
    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Load a specific tensor
    ///
    /// # Arguments
    /// * `name` - Tensor name
    ///
    /// # Returns
    /// DenseTensor with the loaded data
    pub fn get_tensor(&mut self, name: &str) -> GraphResult<DenseTensor> {
        let info = self.tensors.get(name)
            .ok_or_else(|| GraphError::NotFound(name.to_string()))?
            .clone();

        // Seek to tensor data
        let file = self.file.as_mut()
            .ok_or_else(|| GraphError::IoError("File not open".to_string()))?;

        // Offset includes 8 bytes for header length
        let data_offset = 8 + info.offsets[0];
        let data_size = info.offsets[1] - info.offsets[0];

        file.seek(std::io::SeekFrom::Start(data_offset as u64))
            .map_err(|e: std::io::Error| GraphError::IoError(e.to_string()))?;

        // Read tensor data
        let mut buffer = vec![0u8; data_size];
        file.read_exact(&mut buffer)
            .map_err(|e: std::io::Error| GraphError::IoError(e.to_string()))?;

        // Convert to f64 based on dtype
        // Use try_cast_slice for unaligned data, or manual conversion as fallback
        let data = match info.dtype {
            Dtype::F32 => {
                // Use try_cast_slice to handle unaligned data
                match bytemuck::try_cast_slice::<u8, f32>(&buffer) {
                    Ok(f32_data) => f32_data.iter().map(|&x| x as f64).collect(),
                    Err(_) => {
                        // Fallback: manual conversion for unaligned data
                        buffer.chunks_exact(4)
                            .map(|chunk| {
                                let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                                f32::from_le_bytes(bytes) as f64
                            })
                            .collect()
                    }
                }
            }
            Dtype::F16 => {
                match bytemuck::try_cast_slice::<u8, u16>(&buffer) {
                    Ok(f16_data) => f16_data.iter().map(|&x| half::f16::from_bits(x).to_f64()).collect(),
                    Err(_) => {
                        buffer.chunks_exact(2)
                            .map(|chunk| {
                                let bytes: [u8; 2] = [chunk[0], chunk[1]];
                                half::f16::from_bits(u16::from_le_bytes(bytes)).to_f64()
                            })
                            .collect()
                    }
                }
            }
            Dtype::BF16 => {
                match bytemuck::try_cast_slice::<u8, u16>(&buffer) {
                    Ok(bf16_data) => bf16_data.iter().map(|&x| half::bf16::from_bits(x).to_f64()).collect(),
                    Err(_) => {
                        buffer.chunks_exact(2)
                            .map(|chunk| {
                                let bytes: [u8; 2] = [chunk[0], chunk[1]];
                                half::bf16::from_bits(u16::from_le_bytes(bytes)).to_f64()
                            })
                            .collect()
                    }
                }
            }
            Dtype::I32 => {
                match bytemuck::try_cast_slice::<u8, i32>(&buffer) {
                    Ok(i32_data) => i32_data.iter().map(|&x| x as f64).collect(),
                    Err(_) => {
                        buffer.chunks_exact(4)
                            .map(|chunk| {
                                let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                                i32::from_le_bytes(bytes) as f64
                            })
                            .collect()
                    }
                }
            }
            Dtype::I64 => {
                match bytemuck::try_cast_slice::<u8, i64>(&buffer) {
                    Ok(i64_data) => i64_data.iter().map(|&x| x as f64).collect(),
                    Err(_) => {
                        buffer.chunks_exact(8)
                            .map(|chunk| {
                                let bytes: [u8; 8] = [chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]];
                                i64::from_le_bytes(bytes) as f64
                            })
                            .collect()
                    }
                }
            }
        };
        
        Ok(DenseTensor::new(data, info.shape))
    }

    /// Load all tensors
    pub fn get_all_tensors(&mut self) -> GraphResult<HashMap<String, DenseTensor>> {
        let mut tensors = HashMap::new();
        
        for name in self.tensors.keys().cloned().collect::<Vec<_>>() {
            let tensor = self.get_tensor(&name)?;
            tensors.insert(name, tensor);
        }
        
        Ok(tensors)
    }

    /// Get number of tensors
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Close the file handle
    pub fn close(&mut self) {
        self.file = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_parsing() {
        assert_eq!(Dtype::from_str("F32"), Some(Dtype::F32));
        assert_eq!(Dtype::from_str("f16"), Some(Dtype::F16));
        assert_eq!(Dtype::from_str("BF16"), Some(Dtype::BF16));
        assert_eq!(Dtype::from_str("I32"), Some(Dtype::I32));
        assert_eq!(Dtype::from_str("i64"), Some(Dtype::I64));
        assert_eq!(Dtype::from_str("UNKNOWN"), None);
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(Dtype::F32.size(), 4);
        assert_eq!(Dtype::F16.size(), 2);
        assert_eq!(Dtype::BF16.size(), 2);
        assert_eq!(Dtype::I32.size(), 4);
        assert_eq!(Dtype::I64.size(), 8);
    }
}
