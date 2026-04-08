//! 分布式执行引擎
//!
//! 提供分布式图算法执行的框架

use crate::parallel::partitioner::Partition;
use crate::plugins::algorithm::{AlgorithmResult, PluginContext};
use crate::vgi::VirtualGraph;
use std::collections::HashMap;
use std::time::Duration;

/// 执行器配置
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// 工作节点数量
    pub num_workers: usize,
    /// 超时时间
    pub timeout: Option<Duration>,
    /// 重试次数
    pub retry_count: usize,
    /// 自定义配置
    pub properties: HashMap<String, String>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            timeout: Some(Duration::from_secs(300)),
            retry_count: 3,
            properties: HashMap::new(),
        }
    }
}

impl ExecutorConfig {
    /// 创建新的执行器配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置工作节点数量
    pub fn with_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// 设置重试次数
    pub fn with_retry_count(mut self, retry_count: usize) -> Self {
        self.retry_count = retry_count;
        self
    }
}

/// 工作节点信息
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    /// 工作节点 ID
    pub id: usize,
    /// 工作节点地址
    pub address: String,
    /// 分配的分区 ID 列表
    pub partition_ids: Vec<usize>,
    /// 状态
    pub status: WorkerStatus,
}

/// 工作节点状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStatus {
    /// 空闲
    Idle,
    /// 正在执行
    Running,
    /// 已完成
    Completed,
    /// 失败
    Failed(String),
}

/// 分布式执行结果
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// 是否成功
    pub success: bool,
    /// 执行时间（毫秒）
    pub execution_time_ms: u64,
    /// 各分区的结果
    pub partition_results: HashMap<usize, AlgorithmResult>,
    /// 聚合后的最终结果
    pub aggregated_result: Option<AlgorithmResult>,
    /// 错误信息
    pub error_message: Option<String>,
}

impl ExecutionResult {
    /// 创建成功结果
    pub fn success(execution_time_ms: u64) -> Self {
        Self {
            success: true,
            execution_time_ms,
            partition_results: HashMap::new(),
            aggregated_result: None,
            error_message: None,
        }
    }

    /// 创建失败结果
    pub fn failure(error_message: String) -> Self {
        Self {
            success: false,
            execution_time_ms: 0,
            partition_results: HashMap::new(),
            aggregated_result: None,
            error_message: Some(error_message),
        }
    }
}

/// 分布式执行器 trait
///
/// 用于在分布式环境中执行图算法
pub trait DistributedExecutor: Send + Sync {
    /// 获取执行器名称
    fn name(&self) -> &'static str;

    /// 获取执行器配置
    fn config(&self) -> &ExecutorConfig;

    /// 获取工作节点信息
    fn workers(&self) -> &[WorkerInfo];

    /// 初始化执行器
    fn initialize(&mut self) -> Result<(), String>;

    /// 关闭执行器
    fn shutdown(&mut self) -> Result<(), String>;

    /// 执行分布式算法
    ///
    /// # Arguments
    ///
    /// * `graph` - 要执行的图
    /// * `partitions` - 图分区
    /// * `algorithm_name` - 算法名称
    /// * `ctx` - 算法上下文
    ///
    /// # Returns
    ///
    /// 返回执行结果
    fn execute<G>(
        &self,
        graph: &G,
        partitions: Vec<Partition>,
        algorithm_name: &str,
        ctx: &mut PluginContext<G>,
    ) -> Result<ExecutionResult, String>
    where
        G: VirtualGraph + ?Sized;

    /// 聚合分区结果
    fn aggregate_results(
        &self,
        partition_results: HashMap<usize, AlgorithmResult>,
    ) -> Result<AlgorithmResult, String>;

    /// 获取执行统计信息
    fn get_stats(&self) -> ExecutorStats;
}

/// 执行器统计信息
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    /// 总执行次数
    pub total_executions: usize,
    /// 成功执行次数
    pub successful_executions: usize,
    /// 失败执行次数
    pub failed_executions: usize,
    /// 平均执行时间（毫秒）
    pub avg_execution_time_ms: f64,
    /// 当前活跃工作节点数
    pub active_workers: usize,
}

impl ExecutorStats {
    /// 创建新的统计信息
    pub fn new() -> Self {
        Self::default()
    }

    /// 获取成功率
    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.successful_executions as f64 / self.total_executions as f64
        }
    }
}

/// 简单的单机模拟分布式执行器（用于测试和本地开发）
pub struct SingleMachineExecutor {
    config: ExecutorConfig,
    workers: Vec<WorkerInfo>,
    stats: ExecutorStats,
    initialized: bool,
}

impl SingleMachineExecutor {
    /// 创建新的单机执行器
    pub fn new(config: ExecutorConfig) -> Self {
        let workers: Vec<WorkerInfo> = (0..config.num_workers)
            .map(|id| WorkerInfo {
                id,
                address: format!("localhost:{}", 8000 + id),
                partition_ids: Vec::new(),
                status: WorkerStatus::Idle,
            })
            .collect();

        Self {
            config,
            workers,
            stats: ExecutorStats::new(),
            initialized: false,
        }
    }
}

impl DistributedExecutor for SingleMachineExecutor {
    fn name(&self) -> &'static str {
        "single_machine"
    }

    fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    fn workers(&self) -> &[WorkerInfo] {
        &self.workers
    }

    fn initialize(&mut self) -> Result<(), String> {
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), String> {
        self.initialized = false;
        for worker in &mut self.workers {
            worker.status = WorkerStatus::Idle;
        }
        Ok(())
    }

    fn execute<G>(
        &self,
        _graph: &G,
        _partitions: Vec<Partition>,
        _algorithm_name: &str,
        _ctx: &mut PluginContext<G>,
    ) -> Result<ExecutionResult, String>
    where
        G: VirtualGraph + ?Sized,
    {
        if !self.initialized {
            return Err("Executor not initialized".to_string());
        }

        // 单机模式：简单返回成功
        Ok(ExecutionResult::success(0))
    }

    fn aggregate_results(
        &self,
        _partition_results: HashMap<usize, AlgorithmResult>,
    ) -> Result<AlgorithmResult, String> {
        Ok(AlgorithmResult::scalar(0.0))
    }

    fn get_stats(&self) -> ExecutorStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_config() {
        use std::time::Duration;

        let config = ExecutorConfig::new()
            .with_workers(8)
            .with_timeout(Duration::from_secs(600))
            .with_retry_count(5);

        assert_eq!(config.num_workers, 8);
        assert_eq!(config.timeout, Some(Duration::from_secs(600)));
        assert_eq!(config.retry_count, 5);
    }

    #[test]
    fn test_worker_status() {
        let status = WorkerStatus::Running;
        assert_ne!(status, WorkerStatus::Idle);

        let failed = WorkerStatus::Failed("error".to_string());
        if let WorkerStatus::Failed(msg) = failed {
            assert_eq!(msg, "error");
        }
    }

    #[test]
    fn test_execution_result() {
        let success = ExecutionResult::success(100);
        assert!(success.success);
        assert_eq!(success.execution_time_ms, 100);
        assert!(success.error_message.is_none());

        let failure = ExecutionResult::failure("test error".to_string());
        assert!(!failure.success);
        assert_eq!(failure.error_message, Some("test error".to_string()));
    }

    #[test]
    fn test_executor_stats() {
        let stats = ExecutorStats {
            total_executions: 10,
            successful_executions: 8,
            failed_executions: 2,
            avg_execution_time_ms: 50.0,
            active_workers: 4,
        };

        assert_eq!(stats.success_rate(), 0.8);
    }

    #[test]
    fn test_single_machine_executor() {
        use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

        let config = ExecutorConfig::new().with_workers(2);
        let mut executor = SingleMachineExecutor::new(config);

        // 初始化
        assert!(executor.initialize().is_ok());

        // 创建测试图
        let graph = Graph::<(), ()>::undirected();
        let partitions = vec![];
        let mut ctx = PluginContext::new(&graph);

        // 执行
        let result = executor.execute(&graph, partitions, "test", &mut ctx);
        assert!(result.is_ok());

        // 关闭
        assert!(executor.shutdown().is_ok());
    }
}
