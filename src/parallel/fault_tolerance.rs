//! 分布式容错机制
//!
//! 提供分布式系统中的故障检测、恢复和重试机制
//!
//! # 架构
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  Fault Tolerance Layer                      │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
//! │  │   Retry     │  │   Circuit   │  │   Health    │         │
//! │  │   Policy    │  │   Breaker   │  │   Checker   │         │
//! │  └─────────────┘  └─────────────┘  └─────────────┘         │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!              ┌───────────────┼───────────────┐
//!              ▼               ▼               ▼
//! ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
//! │  Failure        │ │  Recovery       │ │  Logging        │
//! │  Detector       │ │  Strategy       │ │  System         │
//! └─────────────────┘ └─────────────────┘ └─────────────────┘
//! ```
//!
//! # 使用示例
//!
//! ```
//! use god_graph::parallel::fault_tolerance::{
//!     FaultTolerance, RetryPolicy, CircuitBreaker, HealthChecker
//! };
//! use std::time::Duration;
//!
//! // 创建重试策略
//! let retry_policy = RetryPolicy::builder()
//!     .with_max_retries(3)
//!     .with_delay(Duration::from_millis(100))
//!     .with_exponential_backoff(true)
//!     .build();
//!
//! // 创建熔断器
//! let circuit_breaker = CircuitBreaker::builder()
//!     .with_failure_threshold(5)
//!     .with_success_threshold(2)
//!     .with_timeout(Duration::from_secs(30))
//!     .build();
//!
//! // 创建健康检查器
//! let health_checker = HealthChecker::new()
//!     .with_interval(Duration::from_secs(5))
//!     .with_timeout(Duration::from_secs(2));
//! ```

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};
use std::collections::HashMap;

#[cfg(feature = "distributed")]
use dashmap::DashMap;

/// 容错 trait
///
/// 定义了分布式系统容错的基本接口
pub trait FaultTolerance: Send + Sync {
    /// 检查是否允许执行
    fn is_allowed(&self) -> bool;

    /// 记录成功
    fn record_success(&self);

    /// 记录失败
    fn record_failure(&self, error: &str);

    /// 重置状态
    fn reset(&self);

    /// 获取统计信息
    fn get_stats(&self) -> FaultToleranceStats;
}

/// 容错统计信息
#[derive(Debug, Clone, Default)]
pub struct FaultToleranceStats {
    /// 总尝试次数
    pub total_attempts: usize,
    /// 成功次数
    pub successes: usize,
    /// 失败次数
    pub failures: usize,
    /// 重试次数
    pub retries: usize,
    /// 熔断次数
    pub circuit_breaks: usize,
    /// 最后成功时间
    pub last_success: Option<Instant>,
    /// 最后失败时间
    pub last_failure: Option<Instant>,
}

impl FaultToleranceStats {
    /// 创建新的统计信息
    pub fn new() -> Self {
        Self::default()
    }

    /// 获取成功率
    pub fn success_rate(&self) -> f64 {
        let total = self.successes + self.failures;
        if total == 0 {
            0.0
        } else {
            self.successes as f64 / total as f64
        }
    }

    /// 获取失败率
    pub fn failure_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }
}

/// 重试策略构建器
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// 最大重试次数
    pub max_retries: usize,
    /// 基础延迟时间
    pub base_delay: Duration,
    /// 最大延迟时间
    pub max_delay: Duration,
    /// 是否使用指数退避
    pub exponential_backoff: bool,
    /// 延迟抖动因子 (0.0-1.0)
    pub jitter_factor: f64,
    /// 可重试的错误类型
    pub retryable_errors: Vec<String>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            exponential_backoff: true,
            jitter_factor: 0.1,
            retryable_errors: vec![
                "timeout".to_string(),
                "connection".to_string(),
                "temporary".to_string(),
            ],
        }
    }
}

impl RetryPolicy {
    /// 创建构建器
    pub fn builder() -> RetryPolicyBuilder {
        RetryPolicyBuilder::default()
    }

    /// 计算下次重试延迟
    pub fn next_delay(&self, attempt: usize) -> Duration {
        let delay = if self.exponential_backoff {
            let exp_delay = self.base_delay.as_millis() as u64 * (1u64 << attempt);
            Duration::from_millis(exp_delay.min(self.max_delay.as_millis() as u64))
        } else {
            self.base_delay
        };

        // 添加抖动
        if self.jitter_factor > 0.0 {
            self.add_jitter(delay)
        } else {
            delay
        }
    }

    /// 检查错误是否可重试
    pub fn is_retryable(&self, error: &str) -> bool {
        let error_lower = error.to_lowercase();
        self.retryable_errors
            .iter()
            .any(|pattern| error_lower.contains(pattern))
    }

    /// 添加抖动
    fn add_jitter(&self, delay: Duration) -> Duration {
        use std::time::Duration;

        let jitter_range = (delay.as_millis() as f64 * self.jitter_factor) as u128;
        let jitter = if jitter_range > 0 {
            // 简单的伪随机
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos()
                % (jitter_range + 1);
            seed as i128 - (jitter_range / 2) as i128
        } else {
            0
        };

        let new_delay = (delay.as_millis() as i128 + jitter).max(0) as u64;
        Duration::from_millis(new_delay)
    }
}

/// 重试策略构建器
#[derive(Debug, Default)]
pub struct RetryPolicyBuilder {
    max_retries: Option<usize>,
    base_delay: Option<Duration>,
    max_delay: Option<Duration>,
    exponential_backoff: Option<bool>,
    jitter_factor: Option<f64>,
    retryable_errors: Option<Vec<String>>,
}

impl RetryPolicyBuilder {
    /// 设置最大重试次数
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// 设置基础延迟
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.base_delay = Some(delay);
        self
    }

    /// 设置最大延迟
    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = Some(max_delay);
        self
    }

    /// 设置是否使用指数退避
    pub fn with_exponential_backoff(mut self, enabled: bool) -> Self {
        self.exponential_backoff = Some(enabled);
        self
    }

    /// 设置抖动因子
    pub fn with_jitter_factor(mut self, factor: f64) -> Self {
        self.jitter_factor = Some(factor);
        self
    }

    /// 添加可重试错误类型
    pub fn with_retryable_error(mut self, error: impl Into<String>) -> Self {
        self.retryable_errors
            .get_or_insert_with(Vec::new)
            .push(error.into());
        self
    }

    /// 构建重试策略
    pub fn build(self) -> RetryPolicy {
        let mut policy = RetryPolicy::default();
        if let Some(v) = self.max_retries {
            policy.max_retries = v;
        }
        if let Some(v) = self.base_delay {
            policy.base_delay = v;
        }
        if let Some(v) = self.max_delay {
            policy.max_delay = v;
        }
        if let Some(v) = self.exponential_backoff {
            policy.exponential_backoff = v;
        }
        if let Some(v) = self.jitter_factor {
            policy.jitter_factor = v;
        }
        if let Some(v) = self.retryable_errors {
            policy.retryable_errors = v;
        }
        policy
    }
}

/// 熔断器状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// 闭合状态（正常）
    Closed,
    /// 打开状态（熔断）
    Open,
    /// 半开状态（测试恢复）
    HalfOpen,
}

/// 熔断器
pub struct CircuitBreaker {
    /// 失败阈值
    failure_threshold: AtomicUsize,
    /// 成功阈值（半开状态）
    success_threshold: AtomicUsize,
    /// 超时时间
    timeout: Duration,
    /// 当前状态（使用 RwLock 提升读性能）
    state: RwLock<CircuitState>,
    /// 失败计数
    failure_count: AtomicUsize,
    /// 成功计数（半开状态）
    success_count: AtomicUsize,
    /// 最后失败时间
    last_failure_time: RwLock<Option<Instant>>,
    /// 统计信息
    stats: RwLock<FaultToleranceStats>,
}

impl CircuitBreaker {
    /// 创建构建器
    pub fn builder() -> CircuitBreakerBuilder {
        CircuitBreakerBuilder::default()
    }

    /// 检查是否允许执行（读多写少，使用 read lock）
    pub fn is_allowed(&self) -> bool {
        let state_guard = self.state.read().unwrap_or_else(|e| e.into_inner());

        match *state_guard {
            CircuitState::Closed => true,
            CircuitState::Open => {
                drop(state_guard); // 释放读锁
                // 检查是否超时
                if let Some(last_failure) = *self
                    .last_failure_time
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                {
                    if last_failure.elapsed() >= self.timeout {
                        // 切换到半开状态（需要写锁）
                        let mut state_guard = self.state.write().unwrap_or_else(|e| e.into_inner());
                        if matches!(*state_guard, CircuitState::Open) {
                            *state_guard = CircuitState::HalfOpen;
                            self.success_count.store(0, Ordering::Relaxed);
                        }
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// 记录成功
    pub fn record_success(&self) {
        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
        stats.successes += 1;
        stats.last_success = Some(Instant::now());
        drop(stats);

        let mut state_guard = self.state.write().unwrap_or_else(|e| e.into_inner());
        if matches!(*state_guard, CircuitState::HalfOpen) {
            let count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
            if count >= self.success_threshold.load(Ordering::Relaxed) {
                // 恢复到闭合状态
                *state_guard = CircuitState::Closed;
                self.failure_count.store(0, Ordering::Relaxed);
            }
        }
    }

    /// 记录失败
    pub fn record_failure(&self, _error: &str) {
        *self
            .last_failure_time
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());

        let mut state_guard = self.state.write().unwrap_or_else(|e| e.into_inner());
        match *state_guard {
            CircuitState::Closed => {
                let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count >= self.failure_threshold.load(Ordering::Relaxed) {
                    // 熔断
                    *state_guard = CircuitState::Open;
                    let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
                    stats.failures += 1;
                    stats.last_failure = Some(Instant::now());
                    stats.circuit_breaks += 1;
                }
            }
            CircuitState::HalfOpen => {
                // 半开状态失败，重新熔断
                *state_guard = CircuitState::Open;
                let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
                stats.failures += 1;
                stats.last_failure = Some(Instant::now());
                stats.circuit_breaks += 1;
            }
            CircuitState::Open => {
                let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
                stats.failures += 1;
                stats.last_failure = Some(Instant::now());
            }
        }
    }

    /// 重置熔断器
    pub fn reset(&self) {
        *self.state.write().unwrap_or_else(|e| e.into_inner()) = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        *self
            .last_failure_time
            .write()
            .unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// 获取当前状态
    pub fn state(&self) -> CircuitState {
        *self.state.read().unwrap_or_else(|e| e.into_inner())
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> FaultToleranceStats {
        self.stats.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl FaultTolerance for CircuitBreaker {
    fn is_allowed(&self) -> bool {
        self.is_allowed()
    }

    fn record_success(&self) {
        self.record_success()
    }

    fn record_failure(&self, error: &str) {
        self.record_failure(error)
    }

    fn reset(&self) {
        self.reset()
    }

    fn get_stats(&self) -> FaultToleranceStats {
        self.get_stats()
    }
}

/// 熔断器构建器
#[derive(Debug, Default)]
pub struct CircuitBreakerBuilder {
    failure_threshold: Option<usize>,
    success_threshold: Option<usize>,
    timeout: Option<Duration>,
}

impl CircuitBreakerBuilder {
    /// 设置失败阈值
    pub fn with_failure_threshold(mut self, threshold: usize) -> Self {
        self.failure_threshold = Some(threshold);
        self
    }

    /// 设置成功阈值
    pub fn with_success_threshold(mut self, threshold: usize) -> Self {
        self.success_threshold = Some(threshold);
        self
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// 构建熔断器
    pub fn build(self) -> CircuitBreaker {
        CircuitBreaker {
            failure_threshold: AtomicUsize::new(self.failure_threshold.unwrap_or(5)),
            success_threshold: AtomicUsize::new(self.success_threshold.unwrap_or(2)),
            timeout: self.timeout.unwrap_or(Duration::from_secs(30)),
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            last_failure_time: RwLock::new(None),
            stats: RwLock::new(FaultToleranceStats::new()),
        }
    }
}

/// 健康检查器（读多写少，使用 RwLock 优化）
pub struct HealthChecker {
    /// 检查间隔
    interval: Duration,
    /// 超时时间
    timeout: Duration,
    /// 节点健康状态
    #[cfg(feature = "distributed")]
    node_health: DashMap<usize, NodeHealth>,
    #[cfg(not(feature = "distributed"))]
    node_health: RwLock<HashMap<usize, NodeHealth>>,
    /// 是否启用
    enabled: AtomicBool,
}

/// 节点健康状态
#[derive(Debug, Clone)]
pub struct NodeHealth {
    /// 节点 ID
    pub node_id: usize,
    /// 是否健康
    pub healthy: bool,
    /// 连续失败次数
    pub consecutive_failures: usize,
    /// 最后检查时间
    pub last_check: Option<Instant>,
    /// 最后成功时间
    pub last_success: Option<Instant>,
    /// 健康评分 (0-100)
    pub health_score: f64,
}

impl NodeHealth {
    /// 创建新的节点健康状态
    pub fn new(node_id: usize) -> Self {
        Self {
            node_id,
            healthy: true,
            consecutive_failures: 0,
            last_check: None,
            last_success: None,
            health_score: 100.0,
        }
    }

    /// 记录成功检查
    pub fn record_success(&mut self) {
        self.healthy = true;
        self.consecutive_failures = 0;
        self.last_check = Some(Instant::now());
        self.last_success = Some(Instant::now());
        self.health_score = (self.health_score + 10.0).min(100.0);
    }

    /// 记录失败检查
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.last_check = Some(Instant::now());

        // 连续失败 3 次标记为不健康
        if self.consecutive_failures >= 3 {
            self.healthy = false;
        }

        // 降低健康评分
        self.health_score = (self.health_score - 20.0).max(0.0);
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthChecker {
    /// 创建新的健康检查器
    #[cfg(feature = "distributed")]
    pub fn new() -> Self {
        Self {
            interval: Duration::from_secs(5),
            timeout: Duration::from_secs(2),
            node_health: DashMap::new(),
            enabled: AtomicBool::new(true),
        }
    }

    /// 创建新的健康检查器（无 distributed 特性）
    #[cfg(not(feature = "distributed"))]
    pub fn new() -> Self {
        Self {
            interval: Duration::from_secs(5),
            timeout: Duration::from_secs(2),
            node_health: RwLock::new(HashMap::new()),
            enabled: AtomicBool::new(true),
        }
    }

    /// 设置检查间隔
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// 启用/禁用健康检查
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// 检查是否启用
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// 注册节点
    #[cfg(feature = "distributed")]
    pub fn register_node(&self, node_id: usize) {
        self.node_health.insert(node_id, NodeHealth::new(node_id));
    }

    /// 注册节点（无 distributed 特性，使用 write lock）
    #[cfg(not(feature = "distributed"))]
    pub fn register_node(&self, node_id: usize) {
        let mut health = self.node_health.write().unwrap_or_else(|e| e.into_inner());
        health.insert(node_id, NodeHealth::new(node_id));
    }

    /// 记录节点成功
    #[cfg(feature = "distributed")]
    pub fn record_node_success(&self, node_id: usize) {
        if let Some(mut node) = self.node_health.get_mut(&node_id) {
            node.record_success();
        }
    }

    /// 记录节点成功（无 distributed 特性，使用 write lock）
    #[cfg(not(feature = "distributed"))]
    pub fn record_node_success(&self, node_id: usize) {
        let mut health = self.node_health.write().unwrap_or_else(|e| e.into_inner());
        if let Some(node) = health.get_mut(&node_id) {
            node.record_success();
        }
    }

    /// 记录节点失败
    #[cfg(feature = "distributed")]
    pub fn record_node_failure(&self, node_id: usize) {
        if let Some(mut node) = self.node_health.get_mut(&node_id) {
            node.record_failure();
        }
    }

    /// 记录节点失败（无 distributed 特性，使用 write lock）
    #[cfg(not(feature = "distributed"))]
    pub fn record_node_failure(&self, node_id: usize) {
        let mut health = self.node_health.write().unwrap_or_else(|e| e.into_inner());
        if let Some(node) = health.get_mut(&node_id) {
            node.record_failure();
        }
    }

    /// 检查节点是否健康（无 distributed 特性，使用 read lock）
    #[cfg(not(feature = "distributed"))]
    pub fn is_node_healthy(&self, node_id: usize) -> bool {
        let health = self.node_health.read().unwrap_or_else(|e| e.into_inner());
        health.get(&node_id).map(|n| n.healthy).unwrap_or(false)
    }

    /// 获取所有节点健康状态（无 distributed 特性，使用 read lock）
    #[cfg(not(feature = "distributed"))]
    pub fn get_all_health(&self) -> HashMap<usize, NodeHealth> {
        self.node_health
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// 获取健康节点列表
    #[cfg(feature = "distributed")]
    pub fn get_healthy_nodes(&self) -> Vec<usize> {
        self.node_health.iter().filter_map(|kv| if kv.value().healthy { Some(*kv.key()) } else { None }).collect()
    }

    /// 获取健康节点列表（无 distributed 特性，使用 read lock）
    #[cfg(not(feature = "distributed"))]
    pub fn get_healthy_nodes(&self) -> Vec<usize> {
        let health = self.node_health.read().unwrap_or_else(|e| e.into_inner());
        health
            .iter()
            .filter(|(_, n)| n.healthy)
            .map(|(id, _)| *id)
            .collect()
    }

    /// 获取检查间隔
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// 获取超时时间
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// 故障检测器
///
/// P0 OPTIMIZATION: Uses Vec<Option<T>> instead of HashMap for dense node IDs
/// This provides O(1) access without hashing overhead and better cache locality
pub struct FailureDetector {
    /// 怀疑阈值
    suspicion_threshold: AtomicUsize,
    /// 节点最大 ID (用于 Vec 预分配)
    max_node_id: AtomicUsize,
    /// 节点怀疑计数 (使用 Vec<Option> 替代 HashMap)
    #[cfg(feature = "distributed")]
    suspicion_counts: DashMap<usize, usize>,
    #[cfg(not(feature = "distributed"))]
    suspicion_counts: RwLock<Vec<Option<usize>>>,
    /// 节点最后响应时间 (使用 Vec<Option> 替代 HashMap)
    #[cfg(feature = "distributed")]
    last_response: DashMap<usize, Instant>,
    #[cfg(not(feature = "distributed"))]
    last_response: RwLock<Vec<Option<Instant>>>,
    /// 故障节点列表 (使用 Vec<Option> 替代 HashMap)
    #[cfg(feature = "distributed")]
    failed_nodes: DashMap<usize, Instant>,
    #[cfg(not(feature = "distributed"))]
    failed_nodes: RwLock<Vec<Option<Instant>>>,
    /// 故障超时
    failure_timeout: Duration,
}

impl Default for FailureDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FailureDetector {
    /// 创建新的故障检测器
    #[cfg(feature = "distributed")]
    pub fn new() -> Self {
        Self {
            suspicion_threshold: AtomicUsize::new(3),
            max_node_id: AtomicUsize::new(0),
            suspicion_counts: DashMap::new(),
            last_response: DashMap::new(),
            failed_nodes: DashMap::new(),
            failure_timeout: Duration::from_secs(60),
        }
    }

    /// 创建新的故障检测器（无 distributed 特性）
    /// P0 OPTIMIZATION: Pre-allocate Vecs for dense node ID access
    #[cfg(not(feature = "distributed"))]
    pub fn new() -> Self {
        Self {
            suspicion_threshold: AtomicUsize::new(3),
            max_node_id: AtomicUsize::new(256), // Pre-allocate for 256 nodes
            suspicion_counts: RwLock::new(vec![None; 256]),
            last_response: RwLock::new(vec![None; 256]),
            failed_nodes: RwLock::new(vec![None; 256]),
            failure_timeout: Duration::from_secs(60),
        }
    }

    /// 创建新的故障检测器（带容量）
    #[cfg(not(feature = "distributed"))]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            suspicion_threshold: AtomicUsize::new(3),
            max_node_id: AtomicUsize::new(capacity),
            suspicion_counts: RwLock::new(vec![None; capacity]),
            last_response: RwLock::new(vec![None; capacity]),
            failed_nodes: RwLock::new(vec![None; capacity]),
            failure_timeout: Duration::from_secs(60),
        }
    }

    /// 设置怀疑阈值
    pub fn with_suspicion_threshold(self, threshold: usize) -> Self {
        self.suspicion_threshold.store(threshold, Ordering::Relaxed);
        self
    }

    /// 设置故障超时
    pub fn with_failure_timeout(mut self, timeout: Duration) -> Self {
        self.failure_timeout = timeout;
        self
    }

    /// 记录节点响应（无 distributed 特性）
    /// P0 OPTIMIZATION: O(1) Vec access with single RwLock write
    #[cfg(not(feature = "distributed"))]
    pub fn record_response(&self, node_id: usize) {
        // Ensure capacity
        self.ensure_capacity(node_id + 1);

        // Single write lock for all updates
        let mut last_response = self.last_response.write().unwrap_or_else(|e| e.into_inner());
        let mut suspicion_counts = self.suspicion_counts.write().unwrap_or_else(|e| e.into_inner());
        let mut failed_nodes = self.failed_nodes.write().unwrap_or_else(|e| e.into_inner());

        last_response[node_id] = Some(Instant::now());
        suspicion_counts[node_id] = Some(0);
        failed_nodes[node_id] = None;
    }

    /// Ensure Vec capacity for node_id
    #[cfg(not(feature = "distributed"))]
    fn ensure_capacity(&self, min_size: usize) {
        let current = self.max_node_id.load(Ordering::Relaxed);
        if min_size > current {
            let new_size = (min_size * 2).max(256);

            // Resize all Vecs with single lock acquisitions
            let mut suspicion_counts = self.suspicion_counts.write().unwrap_or_else(|e| e.into_inner());
            suspicion_counts.resize(new_size, None);

            let mut last_response = self.last_response.write().unwrap_or_else(|e| e.into_inner());
            last_response.resize(new_size, None);

            let mut failed_nodes = self.failed_nodes.write().unwrap_or_else(|e| e.into_inner());
            failed_nodes.resize(new_size, None);

            self.max_node_id.store(new_size, Ordering::Relaxed);
        }
    }

    /// 增加怀疑计数
    #[cfg(feature = "distributed")]
    pub fn suspect_node(&self, node_id: usize) -> bool {
        let mut count = self.suspicion_counts.entry(node_id).or_insert(0);
        *count += 1;
        *count >= self.suspicion_threshold.load(Ordering::Relaxed)
    }

    /// 增加怀疑计数（无 distributed 特性）
    /// P0 OPTIMIZATION: O(1) Vec access with single write lock
    #[cfg(not(feature = "distributed"))]
    pub fn suspect_node(&self, node_id: usize) -> bool {
        self.ensure_capacity(node_id + 1);

        // Direct Vec access - no hashing, single lock
        let mut suspicion_counts = self.suspicion_counts.write().unwrap_or_else(|e| e.into_inner());
        let count = suspicion_counts[node_id].get_or_insert(0);
        *count += 1;

        // 检查是否达到阈值
        *count >= self.suspicion_threshold.load(Ordering::Relaxed)
    }

    /// 检查节点是否故障
    /// P0 OPTIMIZATION: O(1) Vec access instead of HashMap lookup
    #[cfg(feature = "distributed")]
    pub fn is_failed(&self, node_id: usize) -> bool {
        // 检查是否在故障列表中
        if let Some(failure_time) = self.failed_nodes.get(&node_id) {
            if failure_time.elapsed() < self.failure_timeout {
                return true;
            }
        }

        // 检查是否超时
        if let Some(last_time) = self.last_response.get(&node_id) {
            if last_time.elapsed() > self.failure_timeout {
                // 标记为故障
                self.failed_nodes.insert(node_id, Instant::now());
                return true;
            }
        }

        false
    }

    /// 检查节点是否故障（无 distributed 特性）
    /// P0 OPTIMIZATION: O(1) Vec access, single RwLock read lock
    #[cfg(not(feature = "distributed"))]
    pub fn is_failed(&self, node_id: usize) -> bool {
        let max_id = self.max_node_id.load(Ordering::Relaxed);
        if node_id >= max_id {
            return false;
        }

        // Read locks for checking
        let failed_nodes = self.failed_nodes.read().unwrap_or_else(|e| e.into_inner());
        let last_response = self.last_response.read().unwrap_or_else(|e| e.into_inner());

        // 检查是否在故障列表中
        if let Some(failure_time) = &failed_nodes[node_id] {
            if failure_time.elapsed() < self.failure_timeout {
                return true;
            }
        }

        // 检查是否超时
        if let Some(last_time) = &last_response[node_id] {
            if last_time.elapsed() > self.failure_timeout {
                // 标记为故障 - need write lock
                drop(failed_nodes);
                drop(last_response);
                let mut failed = self.failed_nodes.write().unwrap_or_else(|e| e.into_inner());
                if node_id < failed.len() {
                    failed[node_id] = Some(Instant::now());
                }
                return true;
            }
        }

        false
    }

    /// 获取故障节点列表
    /// P0 OPTIMIZATION: Iterate Vec instead of HashMap
    #[cfg(feature = "distributed")]
    pub fn get_failed_nodes(&self) -> Vec<usize> {
        self.failed_nodes.iter().map(|kv| *kv.key()).collect()
    }

    /// 获取故障节点列表（无 distributed 特性）
    #[cfg(not(feature = "distributed"))]
    pub fn get_failed_nodes(&self) -> Vec<usize> {
        let failed_nodes = self.failed_nodes.read().unwrap_or_else(|e| e.into_inner());
        failed_nodes
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.map(|_| i))
            .collect()
    }

    /// 移除故障节点
    /// P0 OPTIMIZATION: O(1) Vec access instead of HashMap + Mutex
    #[cfg(feature = "distributed")]
    pub fn remove_failed_node(&self, node_id: usize) {
        self.failed_nodes.remove(&node_id);
        self.suspicion_counts.remove(&node_id);
        self.last_response.remove(&node_id);
    }

    /// 移除故障节点（无 distributed 特性）
    #[cfg(not(feature = "distributed"))]
    pub fn remove_failed_node(&self, node_id: usize) {
        if node_id >= self.max_node_id.load(Ordering::Relaxed) {
            return;
        }

        // Direct Vec access - no hashing, single lock scope
        let mut failed = self.failed_nodes.write().unwrap_or_else(|e| e.into_inner());
        let mut suspicion = self.suspicion_counts.write().unwrap_or_else(|e| e.into_inner());
        let mut last_resp = self.last_response.write().unwrap_or_else(|e| e.into_inner());

        if node_id < failed.len() {
            failed[node_id] = None;
        }
        if node_id < suspicion.len() {
            suspicion[node_id] = None;
        }
        if node_id < last_resp.len() {
            last_resp[node_id] = None;
        }
    }
}

/// 恢复策略
pub trait RecoveryStrategy: Send + Sync {
    /// 执行恢复
    fn recover(&self, node_id: usize, data: &[u8]) -> Result<(), String>;

    /// 检查是否需要恢复
    fn needs_recovery(&self, node_id: usize) -> bool;

    /// 获取恢复进度
    fn get_recovery_progress(&self, node_id: usize) -> f64;
}

/// 检查点恢复策略
pub struct CheckpointRecovery {
    /// 检查点目录
    checkpoint_dir: String,
    /// 检查点间隔
    checkpoint_interval: Duration,
    /// 最后检查点时间
    last_checkpoint: RwLock<HashMap<usize, Instant>>,
}

impl CheckpointRecovery {
    /// 创建新的检查点恢复策略
    pub fn new(checkpoint_dir: impl Into<String>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            checkpoint_interval: Duration::from_secs(60),
            last_checkpoint: RwLock::new(HashMap::new()),
        }
    }

    /// 设置检查点间隔
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// 创建检查点
    pub fn create_checkpoint(&self, node_id: usize, _data: &[u8]) -> Result<String, String> {
        use std::time::SystemTime;

        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();

        let filename = format!(
            "{}/checkpoint_{}_{}.bin",
            self.checkpoint_dir, node_id, timestamp
        );

        // 在实际实现中，这里会写入文件
        // std::fs::write(&filename, data)?;

        let _ = *self
            .last_checkpoint
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .entry(node_id)
            .or_insert_with(Instant::now);

        Ok(filename)
    }

    /// 加载最新检查点
    pub fn load_latest_checkpoint(&self, _node_id: usize) -> Result<Vec<u8>, String> {
        // 在实际实现中，这里会读取文件
        // 这里返回空数据作为示例
        Ok(vec![])
    }
}

impl RecoveryStrategy for CheckpointRecovery {
    fn recover(&self, node_id: usize, _data: &[u8]) -> Result<(), String> {
        // 从检查点恢复
        let _checkpoint = self.load_latest_checkpoint(node_id)?;
        // 恢复逻辑
        Ok(())
    }

    fn needs_recovery(&self, node_id: usize) -> bool {
        let last_checkpoint = self
            .last_checkpoint
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(&last_time) = last_checkpoint.get(&node_id) {
            last_time.elapsed() > self.checkpoint_interval
        } else {
            true
        }
    }

    fn get_recovery_progress(&self, _node_id: usize) -> f64 {
        // 简化实现，返回 1.0 表示完成
        1.0
    }
}

/// 日志级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// 调试级别日志
    Debug,
    /// 信息级别日志
    Info,
    /// 警告级别日志
    Warn,
    /// 错误级别日志
    Error,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// 日志条目
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// 时间戳
    pub timestamp: Instant,
    /// 日志级别
    pub level: LogLevel,
    /// 目标模块
    pub target: String,
    /// 消息
    pub message: String,
    /// 节点 ID（可选）
    pub node_id: Option<usize>,
}

impl LogEntry {
    /// 创建新的日志条目
    pub fn new(level: LogLevel, target: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            timestamp: Instant::now(),
            level,
            target: target.into(),
            message: message.into(),
            node_id: None,
        }
    }

    /// 设置节点 ID
    pub fn with_node_id(mut self, node_id: usize) -> Self {
        self.node_id = Some(node_id);
        self
    }
}

/// 分布式日志系统
pub struct DistributedLogger {
    /// 日志级别
    min_level: AtomicUsize,
    /// 日志条目（使用 RwLock 提升读性能）
    entries: RwLock<Vec<LogEntry>>,
    /// 最大日志条目数
    max_entries: usize,
}

impl Default for DistributedLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributedLogger {
    /// 创建新的日志系统
    pub fn new() -> Self {
        Self {
            min_level: AtomicUsize::new(LogLevel::Info as usize),
            entries: RwLock::new(Vec::new()),
            max_entries: 10000,
        }
    }

    /// 设置最低日志级别
    pub fn with_min_level(self, level: LogLevel) -> Self {
        self.min_level.store(level as usize, Ordering::Relaxed);
        self
    }

    /// 设置最大日志条目数
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// 记录日志
    pub fn log(&self, entry: LogEntry) {
        if entry.level as usize >= self.min_level.load(Ordering::Relaxed) {
            let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
            entries.push(entry);

            // 限制日志数量
            if entries.len() > self.max_entries {
                let remove_count = entries.len() - self.max_entries;
                entries.drain(0..remove_count);
            }
        }
    }

    /// 记录调试日志
    pub fn debug(&self, target: impl Into<String>, message: impl Into<String>) {
        self.log(LogEntry::new(LogLevel::Debug, target, message));
    }

    /// 记录信息日志
    pub fn info(&self, target: impl Into<String>, message: impl Into<String>) {
        self.log(LogEntry::new(LogLevel::Info, target, message));
    }

    /// 记录警告日志
    pub fn warn(&self, target: impl Into<String>, message: impl Into<String>) {
        self.log(LogEntry::new(LogLevel::Warn, target, message));
    }

    /// 记录错误日志
    pub fn error(&self, target: impl Into<String>, message: impl Into<String>) {
        self.log(LogEntry::new(LogLevel::Error, target, message));
    }

    /// 获取日志条目（使用 read lock）
    pub fn get_entries(&self, level: Option<LogLevel>) -> Vec<LogEntry> {
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        if let Some(min_level) = level {
            entries
                .iter()
                .filter(|e| e.level >= min_level)
                .cloned()
                .collect()
        } else {
            entries.clone()
        }
    }

    /// 清空日志（使用 write lock）
    pub fn clear(&self) {
        self.entries
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

/// 带重试的执行函数
///
/// # Arguments
///
/// * `policy` - 重试策略
/// * `operation` - 要执行的操作
/// * `logger` - 日志记录器
///
/// # Returns
///
/// 返回操作结果或错误
pub fn execute_with_retry<T, F>(
    policy: &RetryPolicy,
    mut operation: F,
    logger: Option<&DistributedLogger>,
) -> Result<T, String>
where
    F: FnMut() -> Result<T, String>,
{
    let mut attempt = 0;
    let mut last_error = String::new();

    while attempt <= policy.max_retries {
        match operation() {
            Ok(result) => return Ok(result),
            Err(error) => {
                last_error = error.clone();

                if let Some(logger) = logger {
                    logger.warn(
                        "retry",
                        format!("Attempt {} failed: {}", attempt + 1, error),
                    );
                }

                // 检查是否可重试
                if !policy.is_retryable(&error) {
                    return Err(error);
                }

                if attempt < policy.max_retries {
                    let delay = policy.next_delay(attempt);
                    std::thread::sleep(delay);
                }

                attempt += 1;
            }
        }
    }

    Err(format!("Max retries exceeded. Last error: {}", last_error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_policy_builder() {
        use std::time::Duration;

        let policy = RetryPolicy::builder()
            .with_max_retries(5)
            .with_delay(Duration::from_millis(50))
            .with_max_delay(Duration::from_secs(5))
            .with_exponential_backoff(true)
            .with_jitter_factor(0.2)
            .with_retryable_error("timeout")
            .with_retryable_error("connection")
            .build();

        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.base_delay, Duration::from_millis(50));
        assert_eq!(policy.max_delay, Duration::from_secs(5));
        assert!(policy.exponential_backoff);
        assert!(policy.is_retryable("connection timeout"));
    }

    #[test]
    fn test_retry_policy_delay() {
        let policy = RetryPolicy::builder()
            .with_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_secs(1))
            .with_exponential_backoff(true)
            .with_jitter_factor(0.0)
            .build();

        assert_eq!(policy.next_delay(0).as_millis(), 100);
        assert_eq!(policy.next_delay(1).as_millis(), 200);
        assert_eq!(policy.next_delay(2).as_millis(), 400);
    }

    #[test]
    fn test_circuit_breaker() {
        let cb = CircuitBreaker::builder()
            .with_failure_threshold(3)
            .with_success_threshold(2)
            .with_timeout(Duration::from_millis(100))
            .build();

        // 初始状态为闭合
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_allowed());

        // 记录失败
        cb.record_failure("error1");
        cb.record_failure("error2");
        assert_eq!(cb.state(), CircuitState::Closed);

        // 达到阈值，熔断
        cb.record_failure("error3");
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.is_allowed());

        // 重置
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open() {
        use std::time::Duration;

        let cb = CircuitBreaker::builder()
            .with_failure_threshold(2)
            .with_success_threshold(2)
            .with_timeout(Duration::from_millis(50))
            .build();

        // 熔断
        cb.record_failure("error1");
        cb.record_failure("error2");
        assert_eq!(cb.state(), CircuitState::Open);

        // 等待超时
        std::thread::sleep(Duration::from_millis(60));

        // 应该切换到半开状态
        assert!(cb.is_allowed());
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // 记录成功
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_health_checker() {
        use std::time::Duration;

        let checker = HealthChecker::new()
            .with_interval(Duration::from_secs(5))
            .with_timeout(Duration::from_secs(2));

        checker.register_node(1);
        checker.register_node(2);

        assert!(checker.is_node_healthy(1));
        assert!(checker.is_node_healthy(2));

        // 记录失败
        checker.record_node_failure(1);
        checker.record_node_failure(1);
        checker.record_node_failure(1);

        assert!(!checker.is_node_healthy(1));
        assert!(checker.is_node_healthy(2));
    }

    #[test]
    fn test_failure_detector() {
        use std::time::Duration;

        let detector = FailureDetector::new()
            .with_suspicion_threshold(2)
            .with_failure_timeout(Duration::from_millis(50));

        detector.record_response(1);
        detector.record_response(2);

        assert!(!detector.is_failed(1));
        assert!(!detector.is_failed(2));

        // 怀疑节点 2
        assert!(!detector.suspect_node(2));
        assert!(detector.suspect_node(2));

        // 等待超时，节点 1 和 2 都会超时
        std::thread::sleep(Duration::from_millis(60));

        // 节点 1 和 2 都故障
        assert!(detector.is_failed(1));
        assert!(detector.is_failed(2));
    }

    #[test]
    fn test_distributed_logger() {
        let logger = DistributedLogger::new()
            .with_min_level(LogLevel::Info)
            .with_max_entries(100);

        logger.debug("test", "debug message");
        logger.info("test", "info message");
        logger.warn("test", "warn message");
        logger.error("test", "error message");

        let entries = logger.get_entries(None);
        assert_eq!(entries.len(), 3); // debug 被过滤

        let info_entries = logger.get_entries(Some(LogLevel::Info));
        assert_eq!(info_entries.len(), 3);

        let error_entries = logger.get_entries(Some(LogLevel::Error));
        assert_eq!(error_entries.len(), 1);
    }

    #[test]
    fn test_execute_with_retry_success() {
        let policy = RetryPolicy::builder()
            .with_max_retries(3)
            .with_delay(Duration::from_millis(10))
            .build();

        let logger = DistributedLogger::new();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let result = execute_with_retry(
            &policy,
            || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Ok::<_, String>("success".to_string())
            },
            Some(&logger),
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_execute_with_retry_failure() {
        let policy = RetryPolicy::builder()
            .with_max_retries(3)
            .with_delay(Duration::from_millis(10))
            .with_retryable_error("timeout")
            .build();

        let logger = DistributedLogger::new();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let result = execute_with_retry(
            &policy,
            || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Err::<String, _>("timeout".to_string())
            },
            Some(&logger),
        );

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 4); // 1 + 3 retries
    }

    #[test]
    fn test_fault_tolerance_stats() {
        let mut stats = FaultToleranceStats::new();
        stats.successes = 8;
        stats.failures = 2;

        assert!((stats.success_rate() - 0.8).abs() < 1e-10);
        assert!((stats.failure_rate() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_node_health() {
        let mut health = NodeHealth::new(1);

        assert!(health.healthy);
        assert_eq!(health.health_score, 100.0);
        assert_eq!(health.consecutive_failures, 0);

        health.record_failure();
        health.record_failure();
        health.record_failure();

        assert!(!health.healthy);
        assert_eq!(health.consecutive_failures, 3);
        assert!(health.health_score < 100.0);

        health.record_success();

        assert!(health.healthy);
        assert_eq!(health.consecutive_failures, 0);
    }
}
