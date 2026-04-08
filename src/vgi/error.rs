//! VGI 错误处理
//!
//! 定义虚拟图接口层的错误类型
//!
//! ## 结构化错误信息（AI 友好）
//!
//! 每个错误都提供：
//! - `error_code()`: 错误代码（如 "VG_001"）
//! - `error_type()`: 错误类型分类
//! - `context()`: 错误上下文（可选）
//!
//! 示例：
//! ```rust
//! use god_graph::vgi::error::VgiError;
//!
//! let err = VgiError::PluginNotFound { plugin_name: "pagerank".to_string() };
//! assert_eq!(err.error_code(), "VG_004");
//! assert_eq!(err.error_type(), "PluginError");
//! ```

use std::fmt;

/// VGI 操作结果
pub type VgiResult<T> = Result<T, VgiError>;

/// 错误上下文信息（AI 友好）
///
/// 提供结构化的错误上下文，便于 AI 自动处理和恢复
#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    /// 实体类型（如 "node", "edge", "graph", "plugin"）
    pub entity_type: &'static str,
    /// 实体 ID
    pub entity_id: Option<usize>,
    /// 操作类型（如 "add", "remove", "get", "execute"）
    pub operation: &'static str,
    /// 图 ID（如果适用）
    pub graph_id: Option<String>,
    /// 额外信息
    pub details: Option<String>,
}

impl ErrorContext {
    /// 创建新的错误上下文
    pub fn new(entity_type: &'static str, operation: &'static str) -> Self {
        Self {
            entity_type,
            operation,
            entity_id: None,
            graph_id: None,
            details: None,
        }
    }

    /// 设置实体 ID
    pub fn with_entity_id(mut self, id: usize) -> Self {
        self.entity_id = Some(id);
        self
    }

    /// 设置图 ID
    pub fn with_graph_id(mut self, id: impl Into<String>) -> Self {
        self.graph_id = Some(id.into());
        self
    }

    /// 设置详细信息
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

/// VGI 错误类型
///
/// 涵盖虚拟图接口层可能遇到的各种错误情况
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VgiError {
    /// 后端不支持请求的能力
    UnsupportedCapability {
        /// 请求的能力
        capability: String,
        /// 后端名称
        backend: String,
    },

    /// 元数据不兼容
    MetadataMismatch {
        /// 期望的元数据
        expected: String,
        /// 实际的元数据
        actual: String,
    },

    /// 插件注册失败
    PluginRegistrationFailed {
        /// 插件名称
        plugin_name: String,
        /// 失败原因
        reason: String,
    },

    /// 插件未找到
    PluginNotFound {
        /// 插件名称
        plugin_name: String,
    },

    /// 插件执行失败
    PluginExecutionFailed {
        /// 插件名称
        plugin_name: String,
        /// 错误信息
        message: String,
    },

    /// 后端初始化失败
    BackendInitializationFailed {
        /// 后端类型
        backend_type: String,
        /// 错误信息
        message: String,
    },

    /// 图分区失败
    PartitionFailed {
        /// 错误信息
        message: String,
    },

    /// 分布式执行失败
    DistributedExecutionFailed {
        /// 错误信息
        message: String,
    },

    /// 内部错误
    Internal {
        /// 错误描述
        message: String,
    },

    /// 验证错误
    ValidationError {
        /// 错误信息
        message: String,
    },
}

impl VgiError {
    /// 获取错误代码（AI 友好）
    ///
    /// 错误代码格式：`VG_XXX`，其中 `VG` 表示 VGI 层，`XXX` 是三位数字
    pub fn error_code(&self) -> &'static str {
        match self {
            VgiError::UnsupportedCapability { .. } => "VG_001",
            VgiError::MetadataMismatch { .. } => "VG_002",
            VgiError::PluginRegistrationFailed { .. } => "VG_003",
            VgiError::PluginNotFound { .. } => "VG_004",
            VgiError::PluginExecutionFailed { .. } => "VG_005",
            VgiError::BackendInitializationFailed { .. } => "VG_006",
            VgiError::PartitionFailed { .. } => "VG_007",
            VgiError::DistributedExecutionFailed { .. } => "VG_008",
            VgiError::Internal { .. } => "VG_009",
            VgiError::ValidationError { .. } => "VG_010",
        }
    }

    /// 获取错误类型分类（AI 友好）
    pub fn error_type(&self) -> &'static str {
        match self {
            VgiError::UnsupportedCapability { .. } => "CapabilityError",
            VgiError::MetadataMismatch { .. } => "MetadataError",
            VgiError::PluginRegistrationFailed { .. }
            | VgiError::PluginNotFound { .. }
            | VgiError::PluginExecutionFailed { .. } => "PluginError",
            VgiError::BackendInitializationFailed { .. } => "BackendError",
            VgiError::PartitionFailed { .. } => "PartitionError",
            VgiError::DistributedExecutionFailed { .. } => "ExecutionError",
            VgiError::Internal { .. } => "InternalError",
            VgiError::ValidationError { .. } => "ValidationError",
        }
    }

    /// 获取错误恢复指南（AI 友好）
    ///
    /// 提供结构化的恢复步骤，便于用户和 AI Agent 自动修复
    ///
    /// # 示例
    ///
    /// ```rust
    /// use god_graph::vgi::error::VgiError;
    ///
    /// let err = VgiError::UnsupportedCapability {
    ///     capability: "Parallel".to_string(),
    ///     backend: "single_machine".to_string(),
    /// };
    /// println!("Recovery guide:\n{}", err.recovery_guide());
    /// ```
    pub fn recovery_guide(&self) -> &'static str {
        match self {
            VgiError::UnsupportedCapability { .. } => {
                "【恢复指南】不支持的能力错误\n\
                 1. 检查当前后端是否支持该能力：backend.has_capability(capability)\n\
                 2. 如果不支持，尝试使用其他后端（如 DistributedBackend）\n\
                 3. 或者使用不需要该能力的算法版本\n\
                 4. 参考：docs/VGI_GUIDE.md#能力发现"
            }
            VgiError::MetadataMismatch { .. } => {
                "【恢复指南】元数据不匹配错误\n\
                 1. 检查期望的元数据：expected\n\
                 2. 检查实际的元数据：actual\n\
                 3. 确保图类型和能力列表匹配\n\
                 4. 使用 GraphMetadata::new() 重新配置"
            }
            VgiError::PluginRegistrationFailed { .. } => {
                "【恢复指南】插件注册失败错误\n\
                 1. 检查插件名称是否唯一：registry.list_plugins()\n\
                 2. 检查插件信息是否完整（name, version, description）\n\
                 3. 确保插件实现了 GraphAlgorithm trait\n\
                 4. 使用不同的插件名称重试"
            }
            VgiError::PluginNotFound { .. } => {
                "【恢复指南】插件未找到错误\n\
                 1. 确认插件名称拼写正确（区分大小写）\n\
                 2. 检查插件是否已注册：registry.is_registered(name)\n\
                 3. 列出所有已注册插件：registry.list_plugins()\n\
                 4. 确保插件在正确的模块中导入"
            }
            VgiError::PluginExecutionFailed { .. } => {
                "【恢复指南】插件执行失败错误\n\
                 1. 查看错误消息：message\n\
                 2. 检查图是否满足算法要求（类型、能力）\n\
                 3. 验证配置参数是否正确\n\
                 4. 检查图数据是否有效（无孤立节点、边权重合法）"
            }
            VgiError::BackendInitializationFailed { .. } => {
                "【恢复指南】后端初始化失败错误\n\
                 1. 检查 BackendConfig 配置是否正确\n\
                 2. 确认内存充足（检查 initial_node_capacity）\n\
                 3. 查看错误消息获取详细信息\n\
                 4. 尝试使用默认配置：BackendConfig::default()"
            }
            VgiError::PartitionFailed { .. } => {
                "【恢复指南】图分区失败错误\n\
                 1. 检查图是否足够大（小图不需要分区）\n\
                 2. 验证分区算法参数\n\
                 3. 确保节点和边数据可序列化\n\
                 4. 考虑使用单分区模式"
            }
            VgiError::DistributedExecutionFailed { .. } => {
                "【恢复指南】分布式执行失败错误\n\
                 1. 检查网络连接和 worker 状态\n\
                 2. 增加超时时间：ExecutionConfig::with_timeout()\n\
                 3. 启用重试机制\n\
                 4. 查看详细错误日志定位问题节点"
            }
            VgiError::Internal { .. } => {
                "【恢复指南】内部错误\n\
                 1. 查看详细错误消息：message\n\
                 2. 检查节点/边索引是否有效\n\
                 3. 确保图操作顺序正确（先添加后访问）\n\
                 4. 如果问题持续，请在 GitHub 报告 issue"
            }
            VgiError::ValidationError { .. } => {
                "【恢复指南】验证错误\n\
                 1. 检查验证失败的具体原因：message\n\
                 2. 确保输入数据符合格式要求\n\
                 3. 验证配置参数在合法范围内\n\
                 4. 参考 API 文档中的参数约束"
            }
        }
    }

    /// 获取错误上下文（AI 友好）
    pub fn context(&self) -> ErrorContext {
        match self {
            VgiError::UnsupportedCapability { capability, backend } => ErrorContext::new(
                "capability",
                "check",
            )
            .with_details(format!("Backend '{}' doesn't support '{}'", backend, capability)),

            VgiError::MetadataMismatch { expected, actual } => ErrorContext::new(
                "metadata",
                "validate",
            )
            .with_details(format!("Expected '{}', got '{}'", expected, actual)),

            VgiError::PluginRegistrationFailed { plugin_name, reason } => ErrorContext::new(
                "plugin",
                "register",
            )
            .with_entity_id(plugin_name.len())
            .with_details(format!("Plugin '{}': {}", plugin_name, reason)),

            VgiError::PluginNotFound { plugin_name } => ErrorContext::new(
                "plugin",
                "find",
            )
            .with_details(format!("Plugin '{}' not found", plugin_name)),

            VgiError::PluginExecutionFailed { plugin_name, message } => ErrorContext::new(
                "plugin",
                "execute",
            )
            .with_details(format!("Plugin '{}': {}", plugin_name, message)),

            VgiError::BackendInitializationFailed { backend_type, message } => ErrorContext::new(
                "backend",
                "initialize",
            )
            .with_details(format!("Backend '{}': {}", backend_type, message)),

            VgiError::PartitionFailed { message } => ErrorContext::new(
                "graph",
                "partition",
            )
            .with_details(message.clone()),

            VgiError::DistributedExecutionFailed { message } => ErrorContext::new(
                "execution",
                "run",
            )
            .with_details(message.clone()),

            VgiError::Internal { message } => ErrorContext::new(
                "internal",
                "unknown",
            )
            .with_details(message.clone()),

            VgiError::ValidationError { message } => ErrorContext::new(
                "validation",
                "check",
            )
            .with_details(message.clone()),
        }
    }
}

impl fmt::Display for VgiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VgiError::UnsupportedCapability {
                capability,
                backend,
            } => {
                write!(
                    f,
                    "Backend '{}' does not support capability '{}'",
                    backend, capability
                )
            }
            VgiError::MetadataMismatch { expected, actual } => {
                write!(
                    f,
                    "Metadata mismatch: expected '{}', got '{}'",
                    expected, actual
                )
            }
            VgiError::PluginRegistrationFailed {
                plugin_name,
                reason,
            } => {
                write!(f, "Failed to register plugin '{}': {}", plugin_name, reason)
            }
            VgiError::PluginNotFound { plugin_name } => {
                write!(f, "Plugin '{}' not found", plugin_name)
            }
            VgiError::PluginExecutionFailed {
                plugin_name,
                message,
            } => {
                write!(f, "Plugin '{}' execution failed: {}", plugin_name, message)
            }
            VgiError::BackendInitializationFailed {
                backend_type,
                message,
            } => {
                write!(
                    f,
                    "Failed to initialize backend '{}': {}",
                    backend_type, message
                )
            }
            VgiError::PartitionFailed { message } => {
                write!(f, "Graph partition failed: {}", message)
            }
            VgiError::DistributedExecutionFailed { message } => {
                write!(f, "Distributed execution failed: {}", message)
            }
            VgiError::Internal { message } => {
                write!(f, "Internal error: {}", message)
            }
            VgiError::ValidationError { message } => {
                write!(f, "Validation error: {}", message)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VgiError {}

impl From<VgiError> for crate::errors::GraphError {
    fn from(err: VgiError) -> Self {
        crate::errors::GraphError::Internal(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vgi_error_display() {
        let err = VgiError::UnsupportedCapability {
            capability: "distributed".to_string(),
            backend: "single_machine".to_string(),
        };
        assert!(err.to_string().contains("does not support capability"));

        let err = VgiError::PluginExecutionFailed {
            plugin_name: "pagerank".to_string(),
            message: "convergence failed".to_string(),
        };
        assert!(err.to_string().contains("execution failed"));
    }
}
