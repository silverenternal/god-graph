//! 图算法插件 trait 定义
//!
//! 定义图算法插件的标准接口

use crate::vgi::VirtualGraph;
use crate::vgi::{Capability, GraphType};
use crate::vgi::{VgiError, VgiResult};
use std::any::Any;
use std::time::Duration;

#[cfg(feature = "mcp")]
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// Fast hash map type - uses FxHashMap when mcp feature is enabled, otherwise std HashMap
#[cfg(feature = "mcp")]
pub type FastHashMap<K, V> = FxHashMap<K, V>;
/// Fast hash map type - uses FxHashMap when mcp feature is enabled, otherwise std HashMap
#[cfg(not(feature = "mcp"))]
pub type FastHashMap<K, V> = HashMap<K, V>;

/// 插件优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum PluginPriority {
    /// 低优先级
    Low,
    /// 普通优先级（默认）
    #[default]
    Normal,
    /// 高优先级
    High,
    /// 紧急优先级
    Critical,
}

impl PluginPriority {
    /// 获取优先级的数值表示（数值越大优先级越高）
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }
}

/// 插件信息
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// 插件名称
    pub name: String,
    /// 插件版本
    pub version: String,
    /// 插件描述
    pub description: String,
    /// 插件作者
    pub author: Option<String>,
    /// 需要的能力列表
    pub required_capabilities: Vec<Capability>,
    /// 支持的图类型
    pub supported_graph_types: Vec<GraphType>,
    /// 插件标签
    pub tags: Vec<String>,
    /// 插件优先级
    pub priority: PluginPriority,
    /// 配置模式（用于验证）
    pub config_schema: FastHashMap<String, ConfigField>,
}

/// 配置字段定义
#[derive(Debug, Clone)]
pub struct ConfigField {
    /// 字段名称
    pub name: String,
    /// 字段类型
    pub field_type: ConfigFieldType,
    /// 是否必填
    pub required: bool,
    /// 默认值
    pub default_value: Option<String>,
    /// 描述
    pub description: String,
}

/// 配置字段类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFieldType {
    /// 字符串
    String,
    /// 整数
    Integer,
    /// 浮点数
    Float,
    /// 布尔值
    Boolean,
    /// 列表
    List,
}

impl ConfigField {
    /// 创建新的配置字段
    pub fn new(name: impl Into<String>, field_type: ConfigFieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            required: false,
            default_value: None,
            description: String::new(),
        }
    }

    /// 设置是否必填
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// 设置默认值
    pub fn default_value(mut self, value: impl Into<String>) -> Self {
        self.default_value = Some(value.into());
        self
    }

    /// 设置描述
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

impl PluginInfo {
    /// 创建新的插件信息
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            description: description.into(),
            author: None,
            required_capabilities: Vec::new(),
            supported_graph_types: Vec::new(),
            tags: Vec::new(),
            priority: PluginPriority::Normal,
            config_schema: FastHashMap::default(),
        }
    }

    /// 设置作者
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// 设置需要的能力
    pub fn with_required_capabilities(mut self, caps: &[Capability]) -> Self {
        self.required_capabilities = caps.to_vec();
        self
    }

    /// 设置支持的图类型
    pub fn with_supported_graph_types(mut self, types: &[GraphType]) -> Self {
        self.supported_graph_types = types.to_vec();
        self
    }

    /// 设置标签
    pub fn with_tags(mut self, tags: &[&str]) -> Self {
        self.tags = tags.iter().map(|s| s.to_string()).collect();
        self
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: PluginPriority) -> Self {
        self.priority = priority;
        self
    }

    /// 添加配置字段
    pub fn with_config_field(mut self, field: ConfigField) -> Self {
        self.config_schema.insert(field.name.clone(), field);
        self
    }

    /// 批量添加配置字段
    pub fn with_config_fields(mut self, fields: Vec<ConfigField>) -> Self {
        for field in fields {
            self.config_schema.insert(field.name.clone(), field);
        }
        self
    }
}

/// 插件上下文
///
/// 提供插件执行时的环境和工具
///
/// 使用泛型设计，避免 trait object 的 dyn 兼容性问题
pub struct PluginContext<'a, G>
where
    G: VirtualGraph + ?Sized,
{
    /// 图引用
    pub graph: &'a G,
    /// 插件配置
    pub config: FastHashMap<String, String>,
    /// 取消标志
    pub cancelled: bool,
    /// 进度回调
    pub progress_callback: Option<Box<dyn Fn(f32) + Send + 'a>>,
    /// 执行超时时间
    pub timeout: Option<Duration>,
    /// 执行开始时间
    pub start_time: Option<std::time::Instant>,
    /// 执行 ID
    pub execution_id: Option<String>,
}

/// 执行配置
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// 超时时间
    pub timeout: Option<Duration>,
    /// 优先级
    pub priority: PluginPriority,
    /// 自定义参数
    pub parameters: FastHashMap<String, String>,
    /// 执行 ID
    pub execution_id: Option<String>,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            timeout: Some(Duration::from_secs(60)), // 默认 60 秒超时
            priority: PluginPriority::Normal,
            parameters: FastHashMap::default(),
            execution_id: None,
        }
    }
}

impl ExecutionConfig {
    /// 创建新的执行配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: PluginPriority) -> Self {
        self.priority = priority;
        self
    }

    /// 设置参数
    pub fn with_parameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// 设置执行 ID
    pub fn with_execution_id(mut self, id: impl Into<String>) -> Self {
        self.execution_id = Some(id.into());
        self
    }
}

impl<'a, G> PluginContext<'a, G>
where
    G: VirtualGraph + ?Sized,
{
    /// 创建新的插件上下文
    pub fn new(graph: &'a G) -> Self {
        Self {
            graph,
            config: FastHashMap::default(),
            cancelled: false,
            progress_callback: None,
            timeout: None,
            start_time: Some(std::time::Instant::now()),
            execution_id: None,
        }
    }

    /// 创建带执行配置的插件上下文
    pub fn with_config_ctx(graph: &'a G, exec_config: &ExecutionConfig) -> Self {
        Self {
            graph,
            config: exec_config.parameters.clone(),
            cancelled: false,
            progress_callback: None,
            timeout: exec_config.timeout,
            start_time: Some(std::time::Instant::now()),
            execution_id: exec_config.execution_id.clone(),
        }
    }

    /// 设置配置项
    pub fn with_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }

    /// 获取配置项
    pub fn get_config(&self, key: &str) -> Option<&String> {
        self.config.get(key)
    }

    /// 获取配置项（带默认值）
    pub fn get_config_or<'b>(&'b self, key: &str, default: &'b str) -> &'b str {
        self.config.get(key).map(|s| s.as_str()).unwrap_or(default)
    }

    /// 解析配置项为指定类型
    pub fn get_config_as<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.config
            .get(key)
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    /// 设置进度回调
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(f32) + Send + 'a,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// 报告进度
    pub fn report_progress(&self, progress: f32) {
        if let Some(callback) = &self.progress_callback {
            callback(progress);
        }
    }

    /// 检查是否被取消
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// 取消执行
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// 检查是否超时
    pub fn is_timeout(&self) -> bool {
        if let (Some(timeout), Some(start_time)) = (self.timeout, self.start_time) {
            start_time.elapsed() > timeout
        } else {
            false
        }
    }

    /// 检查是否可以继续执行（未取消且未超时）
    pub fn can_continue(&self) -> bool {
        !self.is_cancelled() && !self.is_timeout()
    }

    /// 检查图是否支持所需能力
    pub fn check_capability(&self, capability: Capability) -> bool {
        self.graph.has_capability(capability)
    }

    /// 检查图是否支持所有所需能力
    pub fn check_capabilities(&self, capabilities: &[Capability]) -> bool {
        self.graph.has_capabilities(capabilities)
    }

    /// 验证配置是否符合模式
    pub fn validate_config(&self, schema: &FastHashMap<String, ConfigField>) -> VgiResult<()> {
        for (field_name, field) in schema {
            let has_value = self.config.contains_key(field_name);

            if !has_value && field.required {
                return Err(VgiError::ValidationError {
                    message: format!("Required config field '{}' is missing", field_name),
                });
            }

            if let Some(value) = self.config.get(field_name) {
                match field.field_type {
                    ConfigFieldType::Integer => {
                        if value.parse::<i64>().is_err() {
                            return Err(VgiError::ValidationError {
                                message: format!("Field '{}' must be an integer", field_name),
                            });
                        }
                    }
                    ConfigFieldType::Float => {
                        if value.parse::<f64>().is_err() {
                            return Err(VgiError::ValidationError {
                                message: format!("Field '{}' must be a float", field_name),
                            });
                        }
                    }
                    ConfigFieldType::Boolean => {
                        if !["true", "false", "1", "0"].contains(&value.to_lowercase().as_str()) {
                            return Err(VgiError::ValidationError {
                                message: format!("Field '{}' must be a boolean", field_name),
                            });
                        }
                    }
                    _ => {} // String 和 List 不需要特殊验证
                }
            }
        }
        Ok(())
    }
}

/// 算法执行结果
#[derive(Debug, Clone)]
pub struct AlgorithmResult {
    /// 结果名称
    pub name: String,
    /// 结果数据
    pub data: AlgorithmData,
    /// 元数据
    pub metadata: FastHashMap<String, String>,
}

impl AlgorithmResult {
    /// 创建新的结果
    pub fn new(name: impl Into<String>, data: AlgorithmData) -> Self {
        Self {
            name: name.into(),
            data,
            metadata: FastHashMap::default(),
        }
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// 算法数据类型
///
/// 支持多种算法结果类型
#[derive(Debug, Clone)]
pub enum AlgorithmData {
    /// 节点值映射（如 PageRank 分数）
    NodeValues(FastHashMap<usize, f64>),
    /// 节点索引列表（如路径）
    NodeList(Vec<usize>),
    /// 边索引列表
    EdgeList(Vec<usize>),
    /// 社区划分结果
    Communities(Vec<usize>),
    /// 标量值
    Scalar(f64),
    /// 布尔值
    Boolean(bool),
    /// 字符串值
    String(String),
    /// 自定义数据
    Custom(String),
}

impl AlgorithmData {
    /// 尝试获取节点值映射
    pub fn as_node_values(&self) -> Option<&FastHashMap<usize, f64>> {
        match self {
            AlgorithmData::NodeValues(values) => Some(values),
            _ => None,
        }
    }

    /// 尝试获取节点列表
    pub fn as_node_list(&self) -> Option<&Vec<usize>> {
        match self {
            AlgorithmData::NodeList(nodes) => Some(nodes),
            _ => None,
        }
    }

    /// 尝试获取标量值
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            AlgorithmData::Scalar(value) => Some(*value),
            _ => None,
        }
    }

    /// 尝试获取布尔值
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            AlgorithmData::Boolean(value) => Some(*value),
            _ => None,
        }
    }
}

/// 图算法插件 trait
///
/// 所有图算法插件必须实现此 trait
///
/// # 注意
///
/// 由于此 trait 包含泛型方法，它不能直接用于 `dyn Trait`。
/// 如需动态分发，请使用 `PluginRegistry` 来管理插件。
///
/// # 示例
///
/// ```
/// use god_graph::plugins::{GraphAlgorithm, PluginInfo, PluginContext, AlgorithmResult, AlgorithmData};
/// use god_graph::vgi::{VirtualGraph, Capability, GraphType};
/// use std::collections::HashMap;
///
/// struct PageRankPlugin;
///
/// impl GraphAlgorithm for PageRankPlugin {
///     fn info(&self) -> PluginInfo {
///         PluginInfo::new("pagerank", "1.0.0", "PageRank 算法")
///             .with_author("God-Graph Team")
///             .with_required_capabilities(&[Capability::Parallel])
///             .with_supported_graph_types(&[GraphType::Directed, GraphType::Undirected])
///             .with_tags(&["centrality", "ranking"])
///     }
///
///     fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
///     where
///         G: VirtualGraph + ?Sized,
///     {
///         // 实现 PageRank 算法
///         let damping = ctx.get_config_as("damping", 0.85);
///         let max_iter = ctx.get_config_as("max_iter", 20);
///
///         let mut scores = HashMap::new();
///         // ... 计算 PageRank
///
///         Ok(AlgorithmResult::new("pagerank", AlgorithmData::NodeValues(scores)))
///     }
/// }
/// ```
pub trait GraphAlgorithm: Send + Sync {
    /// 获取插件信息
    fn info(&self) -> PluginInfo;

    /// 验证图是否满足算法要求
    ///
    /// 默认实现检查：
    /// 1. 图类型是否支持
    /// 2. 所需能力是否满足
    /// 3. 配置是否符合模式
    ///
    /// 子类可以重写此方法添加额外验证
    fn validate<G>(&self, ctx: &PluginContext<G>) -> VgiResult<()>
    where
        G: VirtualGraph + ?Sized,
    {
        let info = self.info();

        // 检查图类型
        let graph_type = ctx.graph.graph_type();
        if !info.supported_graph_types.is_empty()
            && !info.supported_graph_types.contains(&graph_type)
        {
            return Err(VgiError::MetadataMismatch {
                expected: format!(
                    "One of {:?}",
                    info.supported_graph_types
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                ),
                actual: graph_type.to_string(),
            });
        }

        // 检查能力
        if !info.required_capabilities.is_empty()
            && !ctx.check_capabilities(&info.required_capabilities)
        {
            let first_capability = info.required_capabilities
                .first()
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            return Err(VgiError::UnsupportedCapability {
                capability: first_capability.to_string(),
                backend: "unknown".to_string(),
            });
        }

        // 验证配置
        if !info.config_schema.is_empty() {
            ctx.validate_config(&info.config_schema)?;
        }

        Ok(())
    }

    /// 执行算法
    ///
    /// # 注意
    ///
    /// 算法执行时应定期检查 `ctx.can_continue()` 以支持取消和超时
    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized;

    /// 执行前回调
    ///
    /// 在算法执行前调用，可用于预处理和日志记录
    fn before_execute<G>(&self, _ctx: &PluginContext<G>) -> VgiResult<()>
    where
        G: VirtualGraph + ?Sized,
    {
        Ok(())
    }

    /// 执行后回调
    ///
    /// 在算法执行后调用，可用于后处理和结果验证
    fn after_execute<G>(&self, _ctx: &PluginContext<G>, _result: &AlgorithmResult) -> VgiResult<()>
    where
        G: VirtualGraph + ?Sized,
    {
        Ok(())
    }

    /// 清理资源
    ///
    /// 在算法执行完成后调用，可用于释放资源
    fn cleanup(&self) {}

    /// 获取插件的 Any 引用，用于向下转型
    fn as_any(&self) -> &dyn Any;
}

/// 算法结果类型辅助函数
impl AlgorithmResult {
    /// 创建节点值结果
    pub fn node_values(values: FastHashMap<usize, f64>) -> Self {
        Self::new("node_values", AlgorithmData::NodeValues(values))
    }

    /// 创建节点列表结果
    pub fn node_list(nodes: Vec<usize>) -> Self {
        Self::new("node_list", AlgorithmData::NodeList(nodes))
    }

    /// 创建标量结果
    pub fn scalar(value: f64) -> Self {
        Self::new("scalar", AlgorithmData::Scalar(value))
    }

    /// 创建布尔结果
    pub fn boolean(value: bool) -> Self {
        Self::new("boolean", AlgorithmData::Boolean(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_info() {
        let info = PluginInfo::new("test_plugin", "1.0.0", "Test Plugin Description")
            .with_author("Test Author")
            .with_required_capabilities(&[Capability::Parallel])
            .with_supported_graph_types(&[GraphType::Directed])
            .with_tags(&["test", "demo"]);

        assert_eq!(info.name, "test_plugin");
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.author, Some("Test Author".to_string()));
        assert_eq!(info.required_capabilities, vec![Capability::Parallel]);
        assert_eq!(info.supported_graph_types, vec![GraphType::Directed]);
        assert_eq!(info.tags, vec!["test", "demo"]);
    }

    #[test]
    fn test_algorithm_data() {
        use core::f64::consts::PI;

        let mut values = FastHashMap::default();
        values.insert(0, 1.0);
        values.insert(1, 2.0);

        let data = AlgorithmData::NodeValues(values.clone());
        assert_eq!(data.as_node_values(), Some(&values));
        assert_eq!(data.as_node_list(), None);
        assert_eq!(data.as_scalar(), None);

        let data = AlgorithmData::Scalar(PI);
        assert_eq!(data.as_scalar(), Some(PI));
        assert_eq!(data.as_node_values(), None);
    }

    #[test]
    fn test_algorithm_result() {
        let mut values = FastHashMap::default();
        values.insert(0, 1.0);

        let result = AlgorithmResult::node_values(values)
            .with_metadata("iterations", "10")
            .with_metadata("converged", "true");

        assert_eq!(result.name, "node_values");
        assert!(result.data.as_node_values().is_some());
        assert_eq!(result.metadata.get("iterations"), Some(&"10".to_string()));
    }

    #[test]
    fn test_plugin_priority() {
        assert_eq!(PluginPriority::Low.as_u8(), 0);
        assert_eq!(PluginPriority::Normal.as_u8(), 1);
        assert_eq!(PluginPriority::High.as_u8(), 2);
        assert_eq!(PluginPriority::Critical.as_u8(), 3);

        assert!(PluginPriority::High > PluginPriority::Normal);
        assert!(PluginPriority::Critical > PluginPriority::High);
    }

    #[test]
    fn test_plugin_info_with_priority() {
        let info = PluginInfo::new("test", "1.0.0", "Test").with_priority(PluginPriority::High);

        assert_eq!(info.priority, PluginPriority::High);
    }

    #[test]
    fn test_config_field() {
        let field = ConfigField::new("damping", ConfigFieldType::Float)
            .required(true)
            .default_value("0.85")
            .description("Damping factor");

        assert_eq!(field.name, "damping");
        assert_eq!(field.field_type, ConfigFieldType::Float);
        assert!(field.required);
        assert_eq!(field.default_value, Some("0.85".to_string()));
    }

    #[test]
    fn test_execution_config() {
        use std::time::Duration;

        let config = ExecutionConfig::new()
            .with_timeout(Duration::from_secs(120))
            .with_priority(PluginPriority::High)
            .with_parameter("key", "value")
            .with_execution_id("test-123");

        assert_eq!(config.timeout, Some(Duration::from_secs(120)));
        assert_eq!(config.priority, PluginPriority::High);
        assert_eq!(config.parameters.get("key"), Some(&"value".to_string()));
        assert_eq!(config.execution_id, Some("test-123".to_string()));
    }

    #[test]
    fn test_plugin_context_timeout() {
        use crate::graph::Graph;
    use crate::graph::traits::GraphOps;
        use std::time::Duration;

        let graph = Graph::<String, f64>::directed();
        let mut ctx = PluginContext::new(&graph);

        // 默认没有超时
        assert!(!ctx.is_timeout());

        // 设置超时
        ctx.timeout = Some(Duration::from_millis(10));
        std::thread::sleep(Duration::from_millis(50));
        assert!(ctx.is_timeout());
    }

    #[test]
    fn test_plugin_context_validate_config() {
        use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

        let graph = Graph::<String, f64>::directed();
        let mut ctx = PluginContext::new(&graph);

        let mut schema = FastHashMap::default();
        schema.insert(
            "damping".to_string(),
            ConfigField::new("damping", ConfigFieldType::Float),
        );
        schema.insert(
            "max_iter".to_string(),
            ConfigField::new("max_iter", ConfigFieldType::Integer),
        );

        // 没有配置应该通过验证（非必填）
        assert!(ctx.validate_config(&schema).is_ok());

        // 添加必填字段
        schema.insert(
            "required_field".to_string(),
            ConfigField::new("required_field", ConfigFieldType::String).required(true),
        );

        // 缺少必填字段应该失败
        assert!(ctx.validate_config(&schema).is_err());

        // 添加必填字段
        ctx.config
            .insert("required_field".to_string(), "value".to_string());
        assert!(ctx.validate_config(&schema).is_ok());

        // 类型验证
        ctx.config
            .insert("damping".to_string(), "not_a_float".to_string());
        assert!(ctx.validate_config(&schema).is_err());

        ctx.config.insert("damping".to_string(), "0.85".to_string());
        ctx.config
            .insert("max_iter".to_string(), "not_an_int".to_string());
        assert!(ctx.validate_config(&schema).is_err());
    }
}
