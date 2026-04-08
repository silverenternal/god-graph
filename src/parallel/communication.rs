//! 分布式通信层
//!
//! 提供工作节点间的通信机制

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// 通信配置
#[derive(Debug, Clone)]
pub struct CommunicationConfig {
    /// 消息队列大小
    pub queue_size: usize,
    /// 超时时间
    pub timeout: Option<Duration>,
    /// 心跳间隔
    pub heartbeat_interval: Duration,
    /// 自定义配置
    pub properties: HashMap<String, String>,
}

use std::collections::HashMap;

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            queue_size: 1000,
            timeout: Some(Duration::from_secs(30)),
            heartbeat_interval: Duration::from_secs(5),
            properties: HashMap::new(),
        }
    }
}

impl CommunicationConfig {
    /// 创建新的通信配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置队列大小
    pub fn with_queue_size(mut self, size: usize) -> Self {
        self.queue_size = size;
        self
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// 设置心跳间隔
    pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
        self.heartbeat_interval = interval;
        self
    }
}

/// 消息类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageType {
    /// 请求
    Request,
    /// 响应
    Response,
    /// 广播
    Broadcast,
    /// 心跳
    Heartbeat,
    /// 同步屏障
    Barrier,
    /// 数据交换
    DataExchange,
}

/// 消息状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageStatus {
    /// 待发送
    Pending,
    /// 已发送
    Sent,
    /// 已接收
    Received,
    /// 已处理
    Processed,
    /// 失败
    Failed(String),
}

/// 消息 ID
pub type MessageId = u64;

/// 节点 ID
pub type NodeId = usize;

/// 消息
#[derive(Debug, Clone)]
pub struct Message {
    /// 消息 ID
    pub id: MessageId,
    /// 发送者 ID
    pub from: NodeId,
    /// 接收者 ID（None 表示广播）
    pub to: Option<NodeId>,
    /// 消息类型
    pub message_type: MessageType,
    /// 消息内容
    pub payload: MessagePayload,
    /// 时间戳
    pub timestamp: u64,
    /// 状态
    pub status: MessageStatus,
}

impl Message {
    /// 创建新的请求消息
    pub fn request(from: NodeId, to: NodeId, payload: MessagePayload) -> Self {
        Self {
            id: generate_message_id(),
            from,
            to: Some(to),
            message_type: MessageType::Request,
            payload,
            timestamp: current_timestamp_ms(),
            status: MessageStatus::Pending,
        }
    }

    /// 创建广播消息
    pub fn broadcast(from: NodeId, payload: MessagePayload) -> Self {
        Self {
            id: generate_message_id(),
            from,
            to: None,
            message_type: MessageType::Broadcast,
            payload,
            timestamp: current_timestamp_ms(),
            status: MessageStatus::Pending,
        }
    }

    /// 创建响应消息
    pub fn response(from: NodeId, to: NodeId, payload: MessagePayload) -> Self {
        Self {
            id: generate_message_id(),
            from,
            to: Some(to),
            message_type: MessageType::Response,
            payload,
            timestamp: current_timestamp_ms(),
            status: MessageStatus::Pending,
        }
    }

    /// 创建心跳消息
    pub fn heartbeat(from: NodeId) -> Self {
        Self {
            id: generate_message_id(),
            from,
            to: None,
            message_type: MessageType::Heartbeat,
            payload: MessagePayload::Heartbeat,
            timestamp: current_timestamp_ms(),
            status: MessageStatus::Pending,
        }
    }
}

/// 消息内容
#[derive(Debug, Clone)]
pub enum MessagePayload {
    /// 文本消息
    Text(String),
    /// 二进制数据
    Binary(Vec<u8>),
    /// JSON 数据
    Json(String),
    /// 节点值（用于 PageRank 等算法）
    NodeValues(Vec<(usize, f64)>),
    /// 边界值交换
    BoundaryValues(HashMap<usize, f64>),
    /// 屏障同步
    Barrier {
        /// 屏障 ID
        barrier_id: usize,
        /// 参与者数量
        participant_count: usize,
    },
    /// 心跳
    Heartbeat,
    /// 自定义
    Custom(String),
}

impl PartialEq for MessagePayload {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Json(a), Self::Json(b)) => a == b,
            (Self::Custom(a), Self::Custom(b)) => a == b,
            (Self::Heartbeat, Self::Heartbeat) => true,
            (Self::Barrier { barrier_id: a, .. }, Self::Barrier { barrier_id: b, .. }) => a == b,
            _ => false,
        }
    }
}

impl MessagePayload {
    /// 创建文本消息
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// 创建 JSON 消息
    pub fn json(json: impl Into<String>) -> Self {
        Self::Json(json.into())
    }

    /// 创建节点值消息
    pub fn node_values(values: Vec<(usize, f64)>) -> Self {
        Self::NodeValues(values)
    }
}

/// 消息通道 trait
pub trait Channel: Send + Sync {
    /// 发送消息
    fn send(&self, message: Message) -> Result<(), String>;

    /// 接收消息
    fn recv(&self, timeout: Option<Duration>) -> Option<Message>;

    /// 广播消息
    fn broadcast(&self, from: NodeId, payload: MessagePayload) -> Result<usize, String>;

    /// 获取队列长度
    fn len(&self) -> usize;

    /// 检查是否为空
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 内存通道（用于单机测试）
pub struct InMemoryChannel {
    queue: Arc<RwLock<VecDeque<Message>>>,
    _node_id: NodeId,
}

impl InMemoryChannel {
    /// 创建新的内存通道
    pub fn new(node_id: NodeId) -> Self {
        Self {
            queue: Arc::new(RwLock::new(VecDeque::new())),
            _node_id: node_id,
        }
    }

    /// 创建共享通道（用于多节点模拟）
    pub fn shared(node_id: NodeId, queue: Arc<RwLock<VecDeque<Message>>>) -> Self {
        Self {
            queue,
            _node_id: node_id,
        }
    }
}

impl Channel for InMemoryChannel {
    fn send(&self, message: Message) -> Result<(), String> {
        let mut queue = self.queue.write().map_err(|e| e.to_string())?;
        queue.push_back(message);
        Ok(())
    }

    fn recv(&self, _timeout: Option<Duration>) -> Option<Message> {
        let mut queue = self.queue.write().ok()?;
        queue.pop_front()
    }

    fn broadcast(&self, from: NodeId, payload: MessagePayload) -> Result<usize, String> {
        let message = Message::broadcast(from, payload);
        self.send(message)?;
        Ok(1)
    }

    fn len(&self) -> usize {
        self.queue.read().map(|q| q.len()).unwrap_or(0)
    }
}

/// 消息路由器
pub struct MessageRouter {
    channels: HashMap<NodeId, Arc<dyn Channel>>,
    broadcast_channel: Arc<RwLock<VecDeque<Message>>>,
}

impl MessageRouter {
    /// 创建新的消息路由器
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            broadcast_channel: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// 注册节点通道
    pub fn register_channel(&mut self, node_id: NodeId, channel: Arc<dyn Channel>) {
        self.channels.insert(node_id, channel);
    }

    /// 发送消息到指定节点
    pub fn send_to(&self, to: NodeId, message: Message) -> Result<(), String> {
        if let Some(channel) = self.channels.get(&to) {
            channel.send(message)
        } else {
            Err(format!("Node {} not found", to))
        }
    }

    /// 广播消息到所有节点
    pub fn broadcast(&self, from: NodeId, payload: MessagePayload) -> Result<usize, String> {
        let message = Message::broadcast(from, payload.clone());
        let mut count = 0;

        // 添加到广播队列
        self.broadcast_channel
            .write()
            .map_err(|e| e.to_string())?
            .push_back(message);

        // 发送到所有节点
        for (node_id, channel) in &self.channels {
            if *node_id != from {
                let msg = Message::request(from, *node_id, payload.clone());
                if channel.send(msg).is_ok() {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// 获取广播消息
    pub fn get_broadcast(&self) -> Option<Message> {
        self.broadcast_channel
            .write()
            .ok()
            .and_then(|mut q| q.pop_front())
    }
}

impl Default for MessageRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// 生成消息 ID
fn generate_message_id() -> MessageId {
    use std::sync::atomic::{AtomicU64, Ordering};
    static MESSAGE_COUNTER: AtomicU64 = AtomicU64::new(1);
    MESSAGE_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// 获取当前时间戳（毫秒）
fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn test_communication_config() {
        use std::time::Duration;

        let config = CommunicationConfig::new()
            .with_queue_size(2000)
            .with_timeout(Duration::from_secs(60))
            .with_heartbeat_interval(Duration::from_secs(10));

        assert_eq!(config.queue_size, 2000);
        assert_eq!(config.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_message_creation() {
        let msg = Message::request(0, 1, MessagePayload::text("hello"));
        assert_eq!(msg.from, 0);
        assert_eq!(msg.to, Some(1));
        assert_eq!(msg.message_type, MessageType::Request);
        assert_eq!(msg.status, MessageStatus::Pending);
    }

    #[test]
    fn test_broadcast_message() {
        let msg = Message::broadcast(0, MessagePayload::text("broadcast"));
        assert_eq!(msg.from, 0);
        assert_eq!(msg.to, None);
        assert_eq!(msg.message_type, MessageType::Broadcast);
    }

    #[test]
    fn test_heartbeat_message() {
        let msg = Message::heartbeat(0);
        assert_eq!(msg.from, 0);
        assert_eq!(msg.message_type, MessageType::Heartbeat);
        assert!(matches!(msg.payload, MessagePayload::Heartbeat));
    }

    #[test]
    fn test_in_memory_channel() {
        let channel = InMemoryChannel::new(0);
        assert!(channel.is_empty());

        let msg = Message::request(0, 1, MessagePayload::text("test"));
        assert!(channel.send(msg.clone()).is_ok());
        assert_eq!(channel.len(), 1);
        assert!(!channel.is_empty());

        let received = channel.recv(None);
        assert!(received.is_some());
        assert_eq!(received.unwrap().payload, msg.payload);
        assert!(channel.is_empty());
    }

    #[test]
    fn test_message_router() {
        let mut router = MessageRouter::new();

        let shared_queue = Arc::new(RwLock::new(VecDeque::new()));
        let channel1 = InMemoryChannel::shared(1, shared_queue.clone());
        let channel2 = InMemoryChannel::shared(2, shared_queue.clone());

        router.register_channel(1, Arc::new(channel1));
        router.register_channel(2, Arc::new(channel2));

        let payload = MessagePayload::text("broadcast test");
        let count = router.broadcast(0, payload.clone());
        assert!(count.is_ok());
        assert_eq!(count.unwrap(), 2); // 发送到 2 个节点
    }
}
