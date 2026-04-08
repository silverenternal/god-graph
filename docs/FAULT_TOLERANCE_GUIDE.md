# 分布式容错机制使用指南

**模块**: `god_graph::distributed::fault_tolerance`

**版本**: v0.6.0-alpha

---

## 概述

分布式容错机制提供了在分布式图处理系统中处理故障、重试和恢复的完整框架。

## 核心组件

### 1. 重试策略 (RetryPolicy)

处理临时故障的自动重试机制。

#### 特性
- 可配置的最大重试次数
- 固定延迟或指数退避
- 延迟抖动 (Jitter) 防止雪崩
- 可重试错误类型识别

#### 使用示例

```rust
use god_graph::distributed::fault_tolerance::{RetryPolicy, execute_with_retry, DistributedLogger};
use std::time::Duration;

// 创建重试策略
let retry_policy = RetryPolicy::builder()
    .with_max_retries(3)                    // 最多重试 3 次
    .with_delay(Duration::from_millis(100)) // 基础延迟 100ms
    .with_max_delay(Duration::from_secs(5)) // 最大延迟 5 秒
    .with_exponential_backoff(true)         // 启用指数退避
    .with_jitter_factor(0.1)                // 10% 抖动
    .with_retryable_error("timeout")        // 可重试的错误类型
    .with_retryable_error("connection")
    .build();

// 创建日志记录器
let logger = DistributedLogger::new()
    .with_min_level(LogLevel::Info)
    .with_max_entries(1000);

// 执行带重试的操作
let result = execute_with_retry(&retry_policy, || {
    // 可能失败的操作
    perform_network_request()
}, Some(&logger));

match result {
    Ok(value) => println!("Success: {}", value),
    Err(e) => println!("Failed after retries: {}", e),
}
```

#### 延迟计算

指数退避公式：
```
delay = min(base_delay * 2^attempt, max_delay)
final_delay = delay ± jitter
```

示例（base_delay=100ms, exponential_backoff=true）:
- Attempt 0: 100ms
- Attempt 1: 200ms
- Attempt 2: 400ms
- Attempt 3: 800ms
- ...

### 2. 熔断器 (CircuitBreaker)

防止系统过载的保护机制。

#### 状态转换

```
        +----------+
        |  Closed  |  正常状态
        +----+-----+
             |
             | 失败次数 >= threshold
             v
        +----+-----+
        |   Open   |  熔断状态（拒绝请求）
        +----+-----+
             |
             | timeout 到期
             v
        +----+-----+
        | HalfOpen |  测试恢复
        +----+-----+
             |
             | 成功次数 >= threshold  → Closed
             | 失败                 → Open
```

#### 使用示例

```rust
use god_graph::distributed::fault_tolerance::{CircuitBreaker, CircuitState};
use std::time::Duration;

// 创建熔断器
let circuit_breaker = CircuitBreaker::builder()
    .with_failure_threshold(5)      // 5 次失败后熔断
    .with_success_threshold(2)      // 2 次成功后恢复
    .with_timeout(Duration::from_secs(30)) // 30 秒后尝试恢复
    .build();

// 使用熔断器
fn execute_with_circuit_breaker<T, F>(
    cb: &CircuitBreaker,
    operation: F
) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    // 检查是否允许执行
    if !cb.is_allowed() {
        return Err("Circuit breaker is open".to_string());
    }
    
    // 执行操作
    match operation() {
        Ok(result) => {
            cb.record_success();
            Ok(result)
        }
        Err(e) => {
            cb.record_failure(&e);
            Err(e)
        }
    }
}

// 监控状态
match circuit_breaker.state() {
    CircuitState::Closed => println!("Normal operation"),
    CircuitState::Open => println!("Circuit broken, waiting for recovery"),
    CircuitState::HalfOpen => println!("Testing recovery"),
}

// 获取统计信息
let stats = circuit_breaker.get_stats();
println!("Success rate: {:.2}%", stats.success_rate() * 100.0);
println!("Circuit breaks: {}", stats.circuit_breaks);
```

### 3. 健康检查器 (HealthChecker)

监控节点健康状态。

#### 使用示例

```rust
use god_graph::distributed::fault_tolerance::HealthChecker;
use std::time::Duration;

// 创建健康检查器
let health_checker = HealthChecker::new()
    .with_interval(Duration::from_secs(5))   // 每 5 秒检查一次
    .with_timeout(Duration::from_secs(2));   // 2 秒超时

// 注册节点
health_checker.register_node(1);
health_checker.register_node(2);
health_checker.register_node(3);

// 记录检查结果
health_checker.record_node_success(1);
health_checker.record_node_failure(2);
health_checker.record_node_failure(3);

// 检查节点健康状态
if health_checker.is_node_healthy(1) {
    println!("Node 1 is healthy");
}

// 获取健康节点列表
let healthy_nodes = health_checker.get_healthy_nodes();
println!("Healthy nodes: {:?}", healthy_nodes);

// 获取所有节点健康状态
let all_health = health_checker.get_all_health();
for (node_id, health) in &all_health {
    println!(
        "Node {}: healthy={}, score={:.1}",
        node_id, health.healthy, health.health_score
    );
}
```

#### 健康评分

- 初始分数：100.0
- 成功：+10 分（最高 100）
- 失败：-20 分（最低 0）
- 连续 3 次失败标记为不健康

### 4. 故障检测器 (FailureDetector)

检测节点故障。

#### 使用示例

```rust
use god_graph::distributed::fault_tolerance::FailureDetector;
use std::time::Duration;

// 创建故障检测器
let detector = FailureDetector::new()
    .with_suspicion_threshold(3)         // 3 次怀疑后标记
    .with_failure_timeout(Duration::from_secs(60)); // 60 秒超时

// 记录节点响应
detector.record_response(1);
detector.record_response(2);

// 增加怀疑计数
if !detector.is_failed(2) {
    let is_suspected = detector.suspect_node(2);
    if is_suspected {
        println!("Node 2 is suspected!");
    }
}

// 检查是否故障
if detector.is_failed(2) {
    println!("Node 2 has failed!");
    
    // 获取故障节点列表
    let failed_nodes = detector.get_failed_nodes();
    println!("Failed nodes: {:?}", failed_nodes);
}

// 移除故障节点（修复后）
detector.remove_failed_node(2);
```

### 5. 检查点恢复 (CheckpointRecovery)

数据持久化和恢复。

#### 使用示例

```rust
use god_graph::distributed::fault_tolerance::CheckpointRecovery;
use std::time::Duration;

// 创建检查点恢复策略
let recovery = CheckpointRecovery::new("/tmp/checkpoints")
    .with_interval(Duration::from_secs(60)); // 每 60 秒检查点

// 创建检查点
let node_id = 1;
let data = vec![1, 2, 3, 4, 5]; // 节点状态数据

match recovery.create_checkpoint(node_id, &data) {
    Ok(path) => println!("Checkpoint created: {}", path),
    Err(e) => println!("Failed to create checkpoint: {}", e),
}

// 恢复节点状态
if recovery.needs_recovery(node_id) {
    match recovery.load_latest_checkpoint(node_id) {
        Ok(checkpoint_data) => {
            println!("Loaded checkpoint, restoring state...");
            // 恢复逻辑
        }
        Err(e) => println!("Failed to load checkpoint: {}", e),
    }
}

// 获取恢复进度
let progress = recovery.get_recovery_progress(node_id);
println!("Recovery progress: {:.1}%", progress * 100.0);
```

### 6. 分布式日志系统 (DistributedLogger)

统一的日志记录。

#### 使用示例

```rust
use god_graph::distributed::fault_tolerance::{
    DistributedLogger, LogLevel, LogEntry
};
use std::time::Instant;

// 创建日志系统
let logger = DistributedLogger::new()
    .with_min_level(LogLevel::Info)    // 最低日志级别
    .with_max_entries(10000);          // 最多保留 10000 条

// 记录日志
logger.debug("algorithm", "Starting distributed PageRank");
logger.info("algorithm", "Processing 1000 nodes");
logger.warn("network", "High latency detected");
logger.error("node", "Node 3 failed");

// 带节点 ID 的日志
let entry = LogEntry::new(LogLevel::Info, "partition", "Processing partition")
    .with_node_id(1);
logger.log(entry);

// 查询日志
let error_logs = logger.get_entries(Some(LogLevel::Error));
for entry in &error_logs {
    println!(
        "[{:?}] {}: {} (node: {:?})",
        entry.timestamp.elapsed(),
        entry.level,
        entry.message,
        entry.node_id
    );
}

// 清空日志
logger.clear();
```

#### 日志级别

| 级别 | 说明 | 使用场景 |
|------|------|----------|
| Debug | 调试信息 | 开发调试 |
| Info | 信息 | 正常运行状态 |
| Warn | 警告 | 可恢复的异常 |
| Error | 错误 | 不可恢复的故障 |

---

## 综合示例：容错分布式执行

```rust
use god_graph::distributed::fault_tolerance::*;
use std::time::Duration;
use std::sync::Arc;

struct FaultTolerantExecutor {
    retry_policy: RetryPolicy,
    circuit_breaker: CircuitBreaker,
    health_checker: HealthChecker,
    logger: DistributedLogger,
}

impl FaultTolerantExecutor {
    fn new() -> Self {
        Self {
            retry_policy: RetryPolicy::builder()
                .with_max_retries(3)
                .with_delay(Duration::from_millis(100))
                .with_exponential_backoff(true)
                .build(),
            
            circuit_breaker: CircuitBreaker::builder()
                .with_failure_threshold(5)
                .with_success_threshold(2)
                .with_timeout(Duration::from_secs(30))
                .build(),
            
            health_checker: HealthChecker::new()
                .with_interval(Duration::from_secs(5))
                .with_timeout(Duration::from_secs(2)),
            
            logger: DistributedLogger::new()
                .with_min_level(LogLevel::Info),
        }
    }
    
    fn register_node(&self, node_id: usize) {
        self.health_checker.register_node(node_id);
        self.logger.info("executor", format!("Registered node {}", node_id));
    }
    
    fn execute_task<T, F>(&self, node_id: usize, task: F) -> Result<T, String>
    where
        F: FnMut() -> Result<T, String>,
    {
        // 检查节点健康
        if !self.health_checker.is_node_healthy(node_id) {
            return Err(format!("Node {} is not healthy", node_id));
        }
        
        // 检查熔断器
        if !self.circuit_breaker.is_allowed() {
            return Err("Circuit breaker is open".to_string());
        }
        
        // 带重试执行
        let result = execute_with_retry(&self.retry_policy, task, Some(&self.logger));
        
        match &result {
            Ok(_) => {
                self.circuit_breaker.record_success();
                self.health_checker.record_node_success(node_id);
            }
            Err(e) => {
                self.circuit_breaker.record_failure(e);
                self.health_checker.record_node_failure(node_id);
            }
        }
        
        result
    }
}

// 使用示例
fn main() {
    let executor = FaultTolerantExecutor::new();
    
    // 注册节点
    executor.register_node(1);
    executor.register_node(2);
    
    // 执行任务
    let result = executor.execute_task(1, || {
        // 模拟可能失败的操作
        println!("Executing task on node 1");
        Ok::<_, String>("Task completed".to_string())
    });
    
    match result {
        Ok(msg) => println!("{}", msg),
        Err(e) => println!("Error: {}", e),
    }
}
```

---

## 最佳实践

### 1. 重试策略配置

```rust
// 网络请求：较长延迟，较多重试
let network_retry = RetryPolicy::builder()
    .with_max_retries(5)
    .with_delay(Duration::from_millis(500))
    .with_max_delay(Duration::from_secs(30))
    .with_exponential_backoff(true)
    .build();

// 本地操作：较短延迟，较少重试
let local_retry = RetryPolicy::builder()
    .with_max_retries(2)
    .with_delay(Duration::from_millis(10))
    .with_exponential_backoff(false)
    .build();
```

### 2. 熔断器调优

```rust
// 关键服务：低阈值，快速熔断
let critical_cb = CircuitBreaker::builder()
    .with_failure_threshold(3)
    .with_timeout(Duration::from_secs(10))
    .build();

// 非关键服务：高阈值，容忍故障
let non_critical_cb = CircuitBreaker::builder()
    .with_failure_threshold(10)
    .with_timeout(Duration::from_secs(60))
    .build();
```

### 3. 健康检查频率

```rust
// 高可用环境：频繁检查
let ha_checker = HealthChecker::new()
    .with_interval(Duration::from_secs(2))
    .with_timeout(Duration::from_secs(1));

// 稳定环境：降低频率
let stable_checker = HealthChecker::new()
    .with_interval(Duration::from_secs(10))
    .with_timeout(Duration::from_secs(5));
```

### 4. 日志级别选择

```rust
// 开发环境
let dev_logger = DistributedLogger::new()
    .with_min_level(LogLevel::Debug);

// 生产环境
let prod_logger = DistributedLogger::new()
    .with_min_level(LogLevel::Warn);
```

---

## 故障排查

### 常见问题

#### 1. 重试风暴

**症状**: 系统负载突然升高，大量重试请求

**解决**:
- 增加基础延迟
- 启用指数退避
- 增加抖动因子

```rust
let safe_retry = RetryPolicy::builder()
    .with_delay(Duration::from_secs(1))  // 增加基础延迟
    .with_exponential_backoff(true)      // 启用指数退避
    .with_jitter_factor(0.3)             // 增加抖动
    .build();
```

#### 2. 熔断器频繁打开

**症状**: 服务频繁不可用

**解决**:
- 增加失败阈值
- 缩短超时时间（更快尝试恢复）
- 检查根本原因（网络？负载？）

```rust
let tolerant_cb = CircuitBreaker::builder()
    .with_failure_threshold(10)  // 提高阈值
    .with_timeout(Duration::from_secs(15)) // 更快尝试恢复
    .build();
```

#### 3. 健康检查误报

**症状**: 健康节点被标记为不健康

**解决**:
- 增加检查间隔
- 增加超时时间
- 调整健康评分算法

```rust
let lenient_checker = HealthChecker::new()
    .with_interval(Duration::from_secs(10))  // 降低频率
    .with_timeout(Duration::from_secs(5));   // 增加超时
```

---

## 性能考虑

### 内存使用

- `DistributedLogger`: 限制 `max_entries` 防止内存溢出
- `HealthChecker`: 定期清理不活跃节点
- `FailureDetector`: 限制故障历史记录

### CPU 使用

- 重试延迟避免忙等待（使用 `thread::sleep`）
- 健康检查异步执行
- 日志批量写入

### 线程安全

所有组件使用 `Arc<Mutex<>>` 或 `Atomic` 类型保证线程安全，可在多线程环境中直接使用。

---

## 参考资料

- [分布式系统容错设计模式](https://docs.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)
- [重试策略最佳实践](https://github.com/grpc/grpc/blob/master/doc/retry-policy.md)
- [熔断器实现](https://github.com/resilience4j/resilience4j)
