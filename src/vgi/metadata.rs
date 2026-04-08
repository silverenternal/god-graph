//! 图元数据系统
//!
//! 定义图的元数据结构和能力标识

use std::collections::HashMap;
use std::str::FromStr;

/// 图类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphType {
    /// 有向图
    Directed,
    /// 无向图
    Undirected,
    /// 混合图（同时包含有向边和无向边）
    Mixed,
}

impl GraphType {
    /// 检查是否为有向图
    pub fn is_directed(self) -> bool {
        matches!(self, GraphType::Directed | GraphType::Mixed)
    }

    /// 检查是否为无向图
    pub fn is_undirected(self) -> bool {
        matches!(self, GraphType::Undirected)
    }

    /// 检查是否为混合图
    pub fn is_mixed(self) -> bool {
        matches!(self, GraphType::Mixed)
    }
}

impl std::fmt::Display for GraphType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphType::Directed => write!(f, "Directed"),
            GraphType::Undirected => write!(f, "Undirected"),
            GraphType::Mixed => write!(f, "Mixed"),
        }
    }
}

/// 后端能力标识
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// 支持并行执行
    Parallel,
    /// 支持分布式执行
    Distributed,
    /// 支持增量更新
    IncrementalUpdate,
    /// 支持事务操作
    Transactions,
    /// 支持持久化存储
    Persistence,
    /// 支持图分区
    Partitioning,
    /// 支持动态模式（运行时添加/删除节点）
    DynamicMode,
    /// 支持静态模式（预分配，高性能）
    StaticMode,
    /// 支持加权边
    WeightedEdges,
    /// 支持自环
    SelfLoops,
    /// 支持多重边（两个节点之间多条边）
    MultiEdges,
    /// 支持节点属性
    NodeAttributes,
    /// 支持边属性
    EdgeAttributes,
    /// 支持时间戳（时变图）
    Temporal,
    /// 支持流式处理
    Streaming,
    // ============================================
    // 高级能力（DifferentiableGraph / LLM 优化）
    // ============================================
    /// 支持可微图结构（梯度引导的架构优化）
    DifferentiableStructure,
    /// 支持李群正交化（权重矩阵数值稳定性）
    LieGroupOrthogonalization,
    /// 支持张量环压缩（参数压缩）
    TensorRingCompression,
    /// 支持拓扑缺陷检测（孤立节点、梯度阻断检测）
    TopologyDefectDetection,
    /// 支持注意力机制分析（注意力权重可视化/剪枝）
    AttentionAnalysis,
}

impl Capability {
    /// 获取能力的字符串表示
    pub fn as_str(&self) -> &'static str {
        match self {
            Capability::Parallel => "parallel",
            Capability::Distributed => "distributed",
            Capability::IncrementalUpdate => "incremental_update",
            Capability::Transactions => "transactions",
            Capability::Persistence => "persistence",
            Capability::Partitioning => "partitioning",
            Capability::DynamicMode => "dynamic_mode",
            Capability::StaticMode => "static_mode",
            Capability::WeightedEdges => "weighted_edges",
            Capability::SelfLoops => "self_loops",
            Capability::MultiEdges => "multi_edges",
            Capability::NodeAttributes => "node_attributes",
            Capability::EdgeAttributes => "edge_attributes",
            Capability::Temporal => "temporal",
            Capability::Streaming => "streaming",
            // Advanced capabilities
            Capability::DifferentiableStructure => "differentiable_structure",
            Capability::LieGroupOrthogonalization => "lie_group_orthogonalization",
            Capability::TensorRingCompression => "tensor_ring_compression",
            Capability::TopologyDefectDetection => "topology_defect_detection",
            Capability::AttentionAnalysis => "attention_analysis",
        }
    }
}

impl FromStr for Capability {
    type Err = ();

    /// 从字符串解析能力
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "parallel" => Ok(Capability::Parallel),
            "distributed" => Ok(Capability::Distributed),
            "incremental_update" => Ok(Capability::IncrementalUpdate),
            "transactions" => Ok(Capability::Transactions),
            "persistence" => Ok(Capability::Persistence),
            "partitioning" => Ok(Capability::Partitioning),
            "dynamic_mode" => Ok(Capability::DynamicMode),
            "static_mode" => Ok(Capability::StaticMode),
            "weighted_edges" => Ok(Capability::WeightedEdges),
            "self_loops" => Ok(Capability::SelfLoops),
            "multi_edges" => Ok(Capability::MultiEdges),
            "node_attributes" => Ok(Capability::NodeAttributes),
            "edge_attributes" => Ok(Capability::EdgeAttributes),
            "temporal" => Ok(Capability::Temporal),
            "streaming" => Ok(Capability::Streaming),
            // Advanced capabilities
            "differentiable_structure" => Ok(Capability::DifferentiableStructure),
            "lie_group_orthogonalization" => Ok(Capability::LieGroupOrthogonalization),
            "tensor_ring_compression" => Ok(Capability::TensorRingCompression),
            "topology_defect_detection" => Ok(Capability::TopologyDefectDetection),
            "attention_analysis" => Ok(Capability::AttentionAnalysis),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// 图元数据
///
/// 描述图的基本信息和能力
#[derive(Debug, Clone)]
pub struct GraphMetadata {
    /// 图的唯一标识符
    pub id: String,
    /// 图的名称
    pub name: Option<String>,
    /// 图的类型
    pub graph_type: GraphType,
    /// 节点数量（如果已知）
    pub node_count: Option<usize>,
    /// 边数量（如果已知）
    pub edge_count: Option<usize>,
    /// 后端支持的能力
    pub capabilities: Vec<Capability>,
    /// 自定义元数据键值对
    pub custom: HashMap<String, String>,
}

impl GraphMetadata {
    /// 创建新的元数据
    pub fn new(id: impl Into<String>, graph_type: GraphType) -> Self {
        Self {
            id: id.into(),
            name: None,
            graph_type,
            node_count: None,
            edge_count: None,
            capabilities: Vec::new(),
            custom: HashMap::new(),
        }
    }

    /// 设置名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// 设置节点数量
    pub fn with_node_count(mut self, count: usize) -> Self {
        self.node_count = Some(count);
        self
    }

    /// 设置边数量
    pub fn with_edge_count(mut self, count: usize) -> Self {
        self.edge_count = Some(count);
        self
    }

    /// 添加能力
    pub fn with_capability(mut self, capability: Capability) -> Self {
        self.capabilities.push(capability);
        self
    }

    /// 添加多个能力
    pub fn with_capabilities(mut self, capabilities: impl IntoIterator<Item = Capability>) -> Self {
        self.capabilities.extend(capabilities);
        self
    }

    /// 添加自定义元数据
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// 检查是否支持特定能力
    pub fn supports(&self, capability: Capability) -> bool {
        self.capabilities.contains(&capability)
    }

    /// 检查是否支持所有指定能力
    pub fn supports_all(&self, capabilities: &[Capability]) -> bool {
        capabilities.iter().all(|&c| self.supports(c))
    }

    /// 检查是否支持任一指定能力
    pub fn supports_any(&self, capabilities: &[Capability]) -> bool {
        capabilities.iter().any(|&c| self.supports(c))
    }
}

impl std::fmt::Display for GraphMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GraphMetadata {{ ")?;
        write!(f, "id: {}, ", self.id)?;
        if let Some(name) = &self.name {
            write!(f, "name: {}, ", name)?;
        }
        write!(f, "type: {}", self.graph_type)?;
        if let Some(count) = self.node_count {
            write!(f, ", nodes: {}", count)?;
        }
        if let Some(count) = self.edge_count {
            write!(f, ", edges: {}", count)?;
        }
        if !self.capabilities.is_empty() {
            write!(f, ", capabilities: [")?;
            for (i, cap) in self.capabilities.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", cap)?;
            }
            write!(f, "]")?;
        }
        write!(f, " }}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_type() {
        assert!(GraphType::Directed.is_directed());
        assert!(!GraphType::Directed.is_undirected());
        assert!(GraphType::Undirected.is_undirected());
        assert!(!GraphType::Undirected.is_directed());
        assert!(GraphType::Mixed.is_directed());
        assert!(!GraphType::Mixed.is_undirected());
    }

    #[test]
    fn test_capability_from_str() {
        assert_eq!(Capability::from_str("parallel"), Ok(Capability::Parallel));
        assert_eq!(
            Capability::from_str("distributed"),
            Ok(Capability::Distributed)
        );
        assert!(Capability::from_str("invalid").is_err());
    }

    #[test]
    fn test_graph_metadata() {
        let metadata = GraphMetadata::new("test_graph", GraphType::Directed)
            .with_name("Test Graph")
            .with_node_count(100)
            .with_edge_count(500)
            .with_capability(Capability::Parallel)
            .with_capability(Capability::IncrementalUpdate)
            .with_custom("version".to_string(), "1.0".to_string());

        assert_eq!(metadata.id, "test_graph");
        assert_eq!(metadata.name, Some("Test Graph".to_string()));
        assert_eq!(metadata.node_count, Some(100));
        assert_eq!(metadata.edge_count, Some(500));
        assert!(metadata.supports(Capability::Parallel));
        assert!(metadata.supports(Capability::IncrementalUpdate));
        assert!(!metadata.supports(Capability::Distributed));
        assert_eq!(metadata.custom.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_metadata_supports_all() {
        let metadata = GraphMetadata::new("test", GraphType::Undirected)
            .with_capability(Capability::Parallel)
            .with_capability(Capability::IncrementalUpdate);

        assert!(metadata.supports_all(&[Capability::Parallel, Capability::IncrementalUpdate]));
        assert!(!metadata.supports_all(&[Capability::Parallel, Capability::Distributed]));
    }

    #[test]
    fn test_metadata_supports_any() {
        let metadata =
            GraphMetadata::new("test", GraphType::Undirected).with_capability(Capability::Parallel);

        assert!(metadata.supports_any(&[Capability::Parallel, Capability::Distributed]));
        assert!(!metadata.supports_any(&[Capability::Distributed, Capability::Persistence]));
    }
}
