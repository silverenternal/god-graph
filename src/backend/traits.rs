//! 后端 trait 重新导出
//!
//! Backend trait 已移至 `vgi::traits`，此处仅重新导出以保持向后兼容性

pub use crate::vgi::traits::{Backend, BackendBuilder, BackendConfig, BackendType};
