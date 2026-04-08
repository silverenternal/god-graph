# 并行图处理指南（原分布式指南）

**版本**: v0.6.0-beta
**日期**: 2026-03-31
**状态**: 已完成 ✅

---

## 📖 概述

God-Graph 的并行图处理模块提供了将大型图分割成多个分区并多线程并行执行算法的能力。

### ⚠️ 重要说明

**当前实现是单机多线程并行，不是分布式集群**：

- ✅ **支持**: 图分区、多线程并行执行、结果聚合
- ✅ **使用**: Rayon 线程池实现并行
- ❌ **不支持**: 跨机器分布式执行、网络通信、故障恢复
- ❌ **未来计划**: 真正的分布式集群支持（v1.0+）

**为什么模块名叫 `distributed`？**

历史原因。该模块最初设计时计划支持真正的分布式集群，但当前实现聚焦于单机多线程并行。未来版本计划添加：
- 跨机器通信（gRPC/消息队列）
- 故障恢复机制
- 分布式一致性协议

**适用场景**：
- ✅ 单机处理大规模图数据（百万级节点）
- ✅ 利用多核 CPU 加速图算法
- ✅ 需要图分区和并行计算的场景

**不适用场景**：
- ❌ 超大规模图（十亿级节点）需要跨机器分布
- ❌ 需要高可用性和故障恢复的生产环境

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                  Application Layer                          │
│              (Parallel Algorithms)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Parallel Executor Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Executor  │  │  Scheduler  │  │  Aggregator │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Partitioning Layer                             │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │    Hash     │  │    Range    │                          │
│  │ Partitioner │  │ Partitioner │                          │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Worker Threads (Rayon)                         │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │ Partition │  │ Partition │  │ Partition │               │
│  └───────────┘  └───────────┘  └───────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 创建图分区器

```rust
use god_graph::distributed::partitioner::{HashPartitioner, Partitioner};
use god_graph::graph::Graph;
use god_graph::vgi::VirtualGraph;

// 创建图
let mut graph = Graph::<String, f64>::undirected();
for i in 0..100 {
    graph.add_node(format!("node_{}", i)).unwrap();
}

// 创建 Hash 分区器（4 个分区）
let partitioner = HashPartitioner::new(4);

// 执行分区
let partitions = partitioner.partition_graph(&graph);

println!("Created {} partitions", partitions.len());
for (i, partition) in partitions.iter().enumerate() {
    println!("Partition {}: {} nodes, {} boundary nodes", 
             i, partition.size(), partition.boundary_nodes.len());
}
```

### 2. 使用不同的分区策略

#### Hash 分区器

```rust
use god_graph::distributed::partitioner::HashPartitioner;

// 基本用法
let partitioner = HashPartitioner::new(8);

// 带种子的分区器（用于可重复的分区结果）
let partitioner = HashPartitioner::with_seed(8, 42);

// 从配置创建
use god_graph::distributed::PartitionerConfig;
let config = PartitionerConfig::new(8)
    .with_property("seed", "12345");
let partitioner = HashPartitioner::from_config(&config);
```

#### Range 分区器

```rust
use god_graph::distributed::partitioner::RangePartitioner;

// 基本用法：按索引范围分区
let partitioner = RangePartitioner::new(4);

// 指定每分区节点数
let partitioner = RangePartitioner::with_nodes_per_partition(4, 100);
// 节点 0-99 → 分区 0, 节点 100-199 → 分区 1, ...
```

### 3. 创建分布式执行器

```rust
use god_graph::distributed::{
    SingleMachineExecutor, ExecutorConfig, DistributedExecutor
};
use std::time::Duration;

// 配置执行器
let config = ExecutorConfig::new()
    .with_workers(4)
    .with_timeout(Duration::from_secs(300))
    .with_retry_count(3);

// 创建单机执行器（用于测试）
let mut executor = SingleMachineExecutor::new(config);

// 初始化
executor.initialize().unwrap();

// 使用完毕后关闭
executor.shutdown().unwrap();
```

---

## 📦 核心组件

### 分区器 (Partitioner)

分区器负责将图分割成多个子图，每个子图可以在不同的工作节点上独立处理。

#### Partitioner Trait

```rust
pub trait Partitioner: Send + Sync {
    /// 获取分区器名称
    fn name(&self) -> &'static str;
    
    /// 获取分区数量
    fn num_partitions(&self) -> usize;
    
    /// 根据节点索引计算分区 ID
    fn partition_node(&self, node: NodeIndex) -> PartitionId;
    
    /// 执行图分区
    fn partition_graph<G>(&self, graph: &G) -> Vec<Partition>
    where
        G: VirtualGraph;
    
    /// 获取分区统计信息
    fn partition_stats<G>(&self, graph: &G) -> PartitionStats;
}
```

#### Partition 结构

```rust
pub struct Partition {
    /// 分区 ID
    pub id: PartitionId,
    /// 分区中的节点索引列表
    pub nodes: Vec<NodeIndex>,
    /// 分区中的边索引列表
    pub edges: Vec<usize>,
    /// 边界节点（与其他分区相连的节点）
    pub boundary_nodes: Vec<NodeIndex>,
}
```

#### PartitionStats 统计

```rust
pub struct PartitionStats {
    /// 分区数量
    pub num_partitions: usize,
    /// 总节点数
    pub total_nodes: usize,
    /// 最小区大小
    pub min_partition_size: usize,
    /// 最大分区大小
    pub max_partition_size: usize,
    /// 平均分区大小
    pub avg_partition_size: usize,
    /// 边界节点总数
    pub total_boundary_nodes: usize,
    /// 平衡比率（最大/最小）
    pub balance_ratio: f64,
}

// 检查分区是否平衡
if stats.is_balanced(1.5) {
    println!("Partitions are well balanced");
}
```

### 执行器 (Executor)

执行器负责在分布式环境中调度算法执行。

#### DistributedExecutor Trait

```rust
pub trait DistributedExecutor: Send + Sync {
    fn name(&self) -> &'static str;
    fn config(&self) -> &ExecutorConfig;
    fn workers(&self) -> &[WorkerInfo];
    fn initialize(&mut self) -> Result<(), String>;
    fn shutdown(&mut self) -> Result<(), String>;
    
    fn execute<G>(
        &self,
        graph: &G,
        partitions: Vec<Partition>,
        algorithm_name: &str,
        ctx: &mut PluginContext<G>,
    ) -> Result<ExecutionResult, String>
    where
        G: VirtualGraph;
    
    fn aggregate_results(
        &self,
        partition_results: HashMap<usize, AlgorithmResult>,
    ) -> Result<AlgorithmResult, String>;
}
```

### 通信层 (Communication)

通信层提供工作节点间的消息传递机制。

#### Message 类型

```rust
pub struct Message {
    pub id: MessageId,
    pub from: NodeId,
    pub to: Option<NodeId>, // None 表示广播
    pub message_type: MessageType,
    pub payload: MessagePayload,
    pub timestamp: u64,
    pub status: MessageStatus,
}

pub enum MessageType {
    Request,
    Response,
    Broadcast,
    Heartbeat,
    Barrier,
    DataExchange,
}

pub enum MessagePayload {
    Text(String),
    Binary(Vec<u8>),
    Json(String),
    NodeValues(Vec<(usize, f64)>),
    BoundaryValues(HashMap<usize, f64>),
    // ...
}
```

#### Channel Trait

```rust
pub trait Channel: Send + Sync {
    fn send(&self, message: Message) -> Result<(), String>;
    fn recv(&self, timeout: Option<Duration>) -> Option<Message>;
    fn broadcast(&self, from: NodeId, payload: MessagePayload) -> Result<usize, String>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}
```

---

## 🔧 高级用法

### 自定义分区器

实现自定义分区策略：

```rust
use god_graph::distributed::partitioner::{Partitioner, Partition, PartitionerConfig};
use god_graph::node::NodeIndex;
use god_graph::vgi::VirtualGraph;

struct CustomPartitioner {
    num_partitions: usize,
}

impl CustomPartitioner {
    fn new(num_partitions: usize) -> Self {
        Self { num_partitions }
    }
}

impl Partitioner for CustomPartitioner {
    fn name(&self) -> &'static str {
        "custom"
    }

    fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    fn partition_node(&self, node: NodeIndex) -> usize {
        // 自定义分区逻辑
        node.index() % self.num_partitions
    }
}
```

### 消息路由

```rust
use god_graph::distributed::communication::{
    MessageRouter, Message, MessagePayload, InMemoryChannel
};
use std::sync::Arc;

let mut router = MessageRouter::new();

// 创建并注册节点通道
for node_id in 0..4 {
    let channel = Arc::new(InMemoryChannel::new(node_id));
    router.register_channel(node_id, channel);
}

// 广播消息
let payload = MessagePayload::text("Hello, workers!");
router.broadcast(0, payload).unwrap();

// 发送消息到指定节点
let msg = Message::request(0, 1, MessagePayload::json("{\"cmd\": \"start\"}"));
router.send_to(1, msg).unwrap();
```

---

## 📊 性能优化建议

### 1. 选择合适的分区策略

- **Hash 分区**: 适用于节点分布均匀的场景
- **Range 分区**: 适用于节点有自然顺序的场景
- **自定义分区**: 适用于有特殊图结构的场景

### 2. 平衡分区大小

```rust
let stats = partitioner.partition_stats(&graph);

// 检查平衡比率
if stats.balance_ratio > 1.5 {
    println!("Warning: Unbalanced partitions (ratio: {})", stats.balance_ratio);
}

// 目标：平衡比率 < 1.2
assert!(stats.is_balanced(1.2));
```

### 3. 最小化边界节点

边界节点需要跨分区通信，应尽量减少：

```rust
let boundary_ratio = stats.total_boundary_nodes as f64 / stats.total_nodes as f64;
println!("Boundary ratio: {:.2}%", boundary_ratio * 100.0);
// 目标：边界比率 < 20%
```

### 4. 配置合适的超时时间

```rust
let config = ExecutorConfig::new()
    .with_workers(8)
    .with_timeout(Duration::from_secs(600)) // 根据图大小调整
    .with_retry_count(3);
```

---

## 🧪 测试

### 单元测试示例

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use god_graph::distributed::partitioner::HashPartitioner;

    #[test]
    fn test_partitioner_basic() {
        let partitioner = HashPartitioner::new(4);
        assert_eq!(partitioner.num_partitions(), 4);
    }

    #[test]
    fn test_partition_distribution() {
        let partitioner = HashPartitioner::new(4);
        let mut counts = vec![0; 4];

        for i in 0..1000 {
            let node = NodeIndex::new_public(i);
            let partition = partitioner.partition_node(node);
            counts[partition] += 1;
        }

        // 验证分布均匀
        let avg = 250;
        for count in counts {
            assert!(count >= avg / 2 && count <= avg * 2);
        }
    }
}
```

---

## 📋 检查清单

在使用分布式图处理前，请确认：

- [ ] 选择了合适的分区策略
- [ ] 分区大小平衡（balance_ratio < 1.5）
- [ ] 边界节点比例合理（< 20%）
- [ ] 配置了合适的超时时间
- [ ] 实现了结果聚合逻辑
- [ ] 添加了错误处理和重试机制

---

## 🔜 未来计划

Phase 3 后续将添加：

1. **METIS 分区器**: 更智能的图分区算法
2. **分布式 PageRank**: 完整的分布式实现
3. **分布式 BFS**: 跨分区广度优先搜索
4. **容错机制**: 工作节点失败恢复
5. **动态负载均衡**: 运行时调整分区

---

## 📚 参考资料

- [VGI Phase 2 进度报告](./VGI_PHASE2_PROGRESS.md)
- [插件开发指南](./PLUGIN_DEVELOPMENT_GUIDE.md)
- [VGI 实施计划](./VGI_IMPLEMENTATION_PLAN.md)

---

**最后更新**: 2026-03-31
**维护者**: God-Graph Team
