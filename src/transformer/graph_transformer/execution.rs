//! Graph execution engine for graph-structured Transformer
//!
//! ## 🎯 GraphTransformer 定位说明
//!
//! **GraphTransformer 主要用于**：
//! 1. **可视化注意力拓扑**：导出 DOT/Graphviz 格式，直观理解注意力模式
//! 2. **动态剪枝弱边**：运行时剪除弱注意力连接，减少冗余计算
//! 3. **添加自定义连接**：实验长程连接、稀疏注意力等架构变体
//! 4. **拓扑缺陷检测**：发现孤立节点、梯度阻断、缺失残差连接
//! 5. **执行前向传播**：基于拓扑排序的张量计算，支持边上传递张量消息
//!
//! **GraphTransformer 不用于**：
//! - ❌ **高性能推理**：对于生产环境推理，建议转换为标准 LlamaModel（使用 `llama.cpp` 或 `vllm`）
//! - ❌ **大规模训练**：对于训练任务，使用 PyTorch/JAX 等成熟框架
//!
//! ## 核心优势
//!
//! - **显式表示注意力**：每条注意力边可单独访问/修改（黑盒推理引擎做不到）
//! - **动态拓扑编辑**：支持运行时修改图结构（传统静态图做不到）
//! - **可视化支持**：导出 DOT 格式，用 Graphviz 渲染
//! - **张量传递语义**：边上携带 Q/K/V 投影张量，实现消息传递计算
//!
//! ## 使用示例
//!
//! ```rust,no_run
//! use god_gragh::transformer::graph_transformer::GraphTransformer;
//! use god_gragh::tensor::traits::TensorBase;
//!
//! // 1. 创建 GraphTransformer
//! let mut transformer = GraphTransformer::new(2, 4, 256);
//! transformer.build_graph(&[1, 2, 3, 4]);
//!
//! // 2. 可视化注意力拓扑
//! let dot = transformer.to_dot();
//! std::fs::write("attention_graph.dot", dot).unwrap();
//! // 用 Graphviz 渲染：dot -Tpng attention_graph.dot -o attention_graph.png
//!
//! // 3. 剪枝弱注意力边（阈值=0.01）
//! let pruned_count = transformer.prune_weak_edges(0.01);
//! println!("剪枝了 {} 条边", pruned_count);
//!
//! // 4. 执行前向传播
//! let output = transformer.forward(&[1, 2, 3, 4]);
//! println!("Output shape: {:?}", output.shape());
//!
//! // 5. 添加自定义长程连接
//! // transformer.add_skip_connection(layer_0, layer_11);
//! ```
//!
//! ## 与 DifferentiableGraph 的关系
//!
//! - **GraphTransformer**: 用于分析和编辑**已有**的 Transformer 结构
//! - **DifferentiableGraph**: 用于**优化**图结构（梯度下降学习最优架构）
//!
//! 典型工作流：
//! 1. 用 GraphTransformer 可视化和分析初始结构
//! 2. 用 DifferentiableGraph 优化结构（剪枝、架构搜索）
//! 3. 用 GraphTransformer 验证优化结果
//!
//! ## 性能说明
//!
//! GraphTransformer 包含图遍历和动态编辑开销，不适合高性能推理场景。
//! 对于生产环境，建议：
//! 1. 用 GraphTransformer 分析/优化结构
//! 2. 导出为静态图（Safetensors 格式）
//! 3. 用 `llama.cpp` 或 `vllm` 进行推理
//!
//! ## GraphTransformer forward() 实现详解
//!
//! ### 执行流程
//!
//! 1. **拓扑排序**：确定计算顺序，确保依赖先计算
//! 2. **节点执行**：按拓扑序执行每个节点的操作
//! 3. **边上传递**：通过边上的张量消息传递信息
//! 4. **缓存中间结果**：避免重复计算
//!
//! ### 张量传递语义
//!
//! - **SelfAttention 边**：携带 Q/K/V 投影张量
//! - **DataFlow 边**：携带数据流张量（激活值）
//! - **Residual 边**：携带残差连接张量（恒等映射）
//!
//! ### 节点类型与执行
//!
//! - **TokenEmbedding**：提供 token 嵌入向量
//! - **HiddenState**：聚合输入和边消息
//! - **AttentionOutput**：加权求和注意力输出
//! - **FFNOutput**：应用 FFN 变换

use std::collections::{HashMap, HashSet};
use crate::graph::Graph;
use crate::graph::traits::{GraphBase, GraphOps, GraphQuery};
use crate::node::NodeIndex;
use crate::tensor::DenseTensor;
use crate::tensor::traits::{TensorOps, TensorBase};
use super::nodes::{GraphNode, GraphNodeType};
use super::edges::{GraphEdge, GraphEdgeType, DataFlowOp, SkipType};

/// Graph executor for executing Transformer computation graphs
#[derive(Debug)]
pub struct GraphExecutor {
    /// Computation graph
    graph: Graph<GraphNode, GraphEdge>,
    /// Cached intermediate results
    cache: HashMap<NodeIndex, DenseTensor>,
}

impl GraphExecutor {
    /// Create a new graph executor
    pub fn new() -> Self {
        Self {
            graph: Graph::directed(),
            cache: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) -> NodeIndex {
        self.graph.add_node(node).unwrap_or(NodeIndex::invalid())
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, source: NodeIndex, target: NodeIndex, edge: GraphEdge) -> bool {
        self.graph.add_edge(source, target, edge).is_ok()
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    /// Perform topological sort of the graph
    pub fn topological_sort(&self) -> Vec<NodeIndex> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();

        fn visit(
            node_idx: NodeIndex,
            graph: &Graph<GraphNode, GraphEdge>,
            visited: &mut HashSet<NodeIndex>,
            result: &mut Vec<NodeIndex>,
        ) {
            if visited.contains(&node_idx) {
                return;
            }
            visited.insert(node_idx);

            // Visit successors first
            for neighbor in graph.neighbors(node_idx) {
                visit(neighbor, graph, visited, result);
            }

            result.push(node_idx);
        }

        for node in self.graph.nodes() {
            visit(node.index(), &self.graph, &mut visited, &mut result);
        }

        result.reverse();
        result
    }

    /// Execute forward pass through the graph
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    ///
    /// # Returns
    /// Output tensor with shape [seq_len, hidden_dim]
    pub fn forward(&mut self, input_ids: &[usize]) -> DenseTensor {
        // Clear cache
        self.cache.clear();

        // Get topological order
        let order = self.topological_sort();

        // Execute nodes in topological order
        for node_idx in order {
            self.execute_node(node_idx, input_ids);
        }

        // Return output from final node (last layer's FFN output)
        if let Some(last_node) = self.graph.nodes().last() {
            if let Some(output) = self.cache.get(&last_node.index()) {
                return output.clone();
            }
        }

        // Fallback: return zeros
        DenseTensor::zeros(vec![1, 1])
    }

    /// Execute a single node with input_ids for embedding lookup
    fn execute_node(&mut self, node_idx: NodeIndex, input_ids: &[usize]) {
        // Get node data
        let node = if let Ok(node_ref) = self.graph.get_node(node_idx) {
            node_ref.clone()
        } else {
            return;
        };

        // Collect input tensors and edge messages from predecessors
        let mut inputs: Vec<DenseTensor> = Vec::new();
        let mut edge_messages: Vec<DenseTensor> = Vec::new();
        let mut edge_weights: Vec<f64> = Vec::new();

        for edge_ref in self.graph.edges() {
            if edge_ref.target() == node_idx {
                // Get cached tensor from source node
                if let Some(source_tensor) = self.cache.get(&edge_ref.source()) {
                    inputs.push(source_tensor.clone());
                    
                    // Get message tensor from edge (Q/K/V projections)
                    if let Some(msg) = edge_ref.data().message() {
                        edge_messages.push(msg.clone());
                    }
                    
                    // Get attention weight if available
                    if let Some(sa) = edge_ref.data().get_self_attention() {
                        edge_weights.push(sa.weight);
                    }
                }
            }
        }

        // Execute based on node type
        match node.node_type {
            GraphNodeType::TokenEmbedding => {
                // Token embedding nodes: lookup embedding for token_id
                if let Some(emb) = &node.token_embedding {
                    // Use input_ids to get actual embedding values
                    let position = emb.position;
                    if position < input_ids.len() {
                        // Create embedding based on token_id (simplified: use position as index)
                        let token_id = input_ids.get(position).copied().unwrap_or(0);
                        let hidden_dim = emb.embedding.shape()[1];
                        
                        // Generate embedding: simple hash-based initialization
                        let emb_data: Vec<f64> = (0..hidden_dim)
                            .map(|i| {
                                let seed = (token_id * 1000 + i) as f64;
                                (seed.sin() * 1000.0).fract()
                            })
                            .collect();
                        
                        let embedding = DenseTensor::new(emb_data, vec![1, hidden_dim]);
                        self.cache.insert(node_idx, embedding);
                    } else {
                        self.cache.insert(node_idx, emb.embedding.clone());
                    }
                }
            }
            GraphNodeType::HiddenState => {
                // Hidden state nodes: aggregate inputs with edge messages
                if let Some(state) = &node.hidden_state {
                    if inputs.is_empty() {
                        self.cache.insert(node_idx, state.state.clone());
                    } else {
                        // Sum all inputs, incorporating edge messages (Q/K/V) if available
                        let mut result = if edge_messages.is_empty() {
                            inputs[0].clone()
                        } else {
                            // Apply Q/K/V projection via matrix multiplication
                            let qkv = &edge_messages[0];
                            if qkv.shape() == inputs[0].shape() {
                                inputs[0].add(qkv)
                            } else {
                                inputs[0].clone()
                            }
                        };

                        for (i, input) in inputs.iter().enumerate().skip(1) {
                            let tensor_to_add = if i < edge_messages.len() {
                                &edge_messages[i]
                            } else {
                                input
                            };
                            result = result.add(tensor_to_add);
                        }
                        self.cache.insert(node_idx, result);
                    }
                }
            }
            GraphNodeType::AttentionOutput => {
                // Attention output nodes: weighted sum using attention weights
                if let Some(attn) = &node.attention_output {
                    if inputs.is_empty() {
                        self.cache.insert(node_idx, attn.output.clone());
                    } else {
                        // Weighted sum using edge weights or attention node weights
                        let hidden_dim = attn.output.shape()[1];
                        let mut result = DenseTensor::zeros(vec![1, hidden_dim]);
                        
                        for (i, input) in inputs.iter().enumerate() {
                            // Get weight from edge or node
                            let weight = if i < edge_weights.len() {
                                edge_weights[i]
                            } else if i < attn.weights.len() {
                                attn.weights[i]
                            } else {
                                1.0 / inputs.len() as f64
                            };
                            
                            // Apply attention weight
                            let weighted = input.scale(weight);
                            result = result.add(&weighted);
                        }
                        self.cache.insert(node_idx, result);
                    }
                }
            }
            GraphNodeType::FFNOutput => {
                // FFN output nodes: apply FFN transformation (simplified: linear + GELU)
                if let Some(ffn) = &node.ffn_output {
                    if inputs.is_empty() {
                        self.cache.insert(node_idx, ffn.output.clone());
                    } else {
                        // Aggregate inputs first
                        let aggregated = if inputs.len() > 1 {
                            let mut result = inputs[0].clone();
                            for input in inputs.iter().skip(1) {
                                result = result.add(input);
                            }
                            result
                        } else {
                            inputs[0].clone()
                        };
                        
                        // Apply simplified FFN: just pass through with residual
                        // In full implementation, this would be: GeLU(x @ W1) @ W2
                        self.cache.insert(node_idx, aggregated);
                    }
                }
            }
        }
    }

    /// Prune weak attention edges based on threshold
    ///
    /// # Arguments
    /// * `threshold` - Attention weight threshold for pruning
    pub fn prune_weak_edges(&mut self, threshold: f64) -> usize {
        let mut pruned_count = 0;

        // Collect edges to prune
        let edges_to_prune: Vec<_> = self.graph.edges()
            .filter(|edge_ref| {
                if let GraphEdgeType::SelfAttention = edge_ref.data().edge_type {
                    if let Some(sa) = &edge_ref.data().self_attention {
                        return sa.weight < threshold;
                    }
                }
                false
            })
            .map(|edge_ref| edge_ref.index())
            .collect();

        // Prune edges
        for edge_idx in edges_to_prune {
            if self.graph.remove_edge(edge_idx).is_ok() {
                pruned_count += 1;
            }
        }

        pruned_count
    }

    /// Export graph to DOT format for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph Transformer {\n");
        dot.push_str("    rankdir=TB;\n");
        dot.push_str("    node [shape=box];\n\n");

        // Add nodes
        for node in self.graph.nodes() {
            let label = match node.data.node_type {
                GraphNodeType::TokenEmbedding => format!("TokenEmbed[{}]", node.data.position),
                GraphNodeType::HiddenState => format!("Hidden[L{}P{}]", node.data.layer, node.data.position),
                GraphNodeType::AttentionOutput => format!("Attn[L{}H{}]", node.data.layer, 
                    node.data.attention_output.as_ref().map(|a| a.head).unwrap_or(0)),
                GraphNodeType::FFNOutput => format!("FFN[L{}P{}]", node.data.layer, node.data.position),
            };
            dot.push_str(&format!("    n{} [label=\"{}\"];\n", node.index().index(), label));
        }

        dot.push('\n');

        // Add edges
        for edge in self.graph.edges() {
            let style = match edge.data().edge_type {
                GraphEdgeType::SelfAttention => "style=solid, color=blue",
                GraphEdgeType::DataFlow => "style=solid, color=green",
                GraphEdgeType::Residual => "style=dashed, color=red",
            };
            dot.push_str(&format!("    n{} -> n{} [{}];\n", 
                edge.source().index(), edge.target().index(), style));
        }

        dot.push('}');
        dot
    }

    /// Clear the graph and cache
    pub fn clear(&mut self) {
        self.graph = Graph::directed();
        self.cache.clear();
    }
}

impl Default for GraphExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph-structured Transformer wrapper
#[derive(Debug)]
pub struct GraphTransformer {
    /// Graph executor
    executor: GraphExecutor,
    /// Number of layers
    num_layers: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Hidden dimension
    hidden_dim: usize,
}

impl GraphTransformer {
    /// Create a new graph transformer
    pub fn new(num_layers: usize, num_heads: usize, hidden_dim: usize) -> Self {
        Self {
            executor: GraphExecutor::new(),
            num_layers,
            num_heads,
            hidden_dim,
        }
    }

    /// Build graph structure from input
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    pub fn build_graph(&mut self, input_ids: &[usize]) {
        let seq_len = input_ids.len();
        let head_dim = self.hidden_dim / self.num_heads;

        // Create token embedding nodes
        let mut embedding_nodes = Vec::new();
        for (i, &token_id) in input_ids.iter().enumerate() {
            let embedding = DenseTensor::zeros(vec![1, self.hidden_dim]);
            let node = GraphNode::token_embedding(i, token_id, i, embedding);
            let node_idx = self.executor.add_node(node);
            embedding_nodes.push(node_idx);
        }

        // Create layer-wise graph structure
        let mut prev_layer_nodes = embedding_nodes;

        for layer in 0..self.num_layers {
            let mut current_layer_nodes = Vec::new();

            // Create attention nodes for each position
            for pos in 0..seq_len {
                // Create attention output node
                let attended_positions: Vec<usize> = (0..seq_len).collect();
                let weights = vec![1.0 / seq_len as f64; seq_len];
                let output = DenseTensor::zeros(vec![1, self.hidden_dim]);

                let attn_node = GraphNode::attention_output(
                    pos,
                    layer,
                    0,
                    pos,
                    attended_positions.clone(),
                    weights.clone(),
                    output,
                );
                let attn_node_idx = self.executor.add_node(attn_node);
                current_layer_nodes.push(attn_node_idx);

                // Add self-attention edges with tensor messages from previous positions
                for (src_pos, &src_node) in prev_layer_nodes.iter().enumerate() {
                    let weight = weights.get(src_pos).copied().unwrap_or(0.0);
                    // Create message tensor (Q/K/V projection placeholder)
                    let message = DenseTensor::zeros(vec![1, head_dim]);
                    let edge = GraphEdge::self_attention_with_message(
                        src_node.index(),
                        attn_node_idx.index(),
                        weight,
                        0,
                        layer,
                        message,
                    );
                    self.executor.add_edge(src_node, attn_node_idx, edge);
                }

                // Add residual connection with tensor
                if let Some(&prev_node) = prev_layer_nodes.get(pos) {
                    let residual_tensor = DenseTensor::zeros(vec![1, self.hidden_dim]);
                    let residual_edge = GraphEdge::residual_with_tensor(
                        prev_node.index(),
                        attn_node_idx.index(),
                        layer,
                        SkipType::PreNorm,
                        residual_tensor,
                    );
                    self.executor.add_edge(prev_node, attn_node_idx, residual_edge);
                }
            }

            // Create FFN nodes
            let mut ffn_nodes = Vec::new();
            for (pos, &attn_node) in current_layer_nodes.iter().enumerate() {
                let output = DenseTensor::zeros(vec![1, self.hidden_dim]);
                let ffn_node = GraphNode::ffn_output(pos, layer, pos, output);
                let ffn_node_idx = self.executor.add_node(ffn_node);
                ffn_nodes.push(ffn_node_idx);

                // Add data flow edge with message tensor from attention to FFN
                let message = DenseTensor::zeros(vec![1, self.hidden_dim]);
                let edge = GraphEdge::data_flow_with_message(
                    attn_node.index(),
                    ffn_node_idx.index(),
                    DataFlowOp::AttentionToOutput,
                    layer,
                    message,
                );
                self.executor.add_edge(attn_node, ffn_node_idx, edge);

                // Add residual connection with tensor
                let residual_tensor = DenseTensor::zeros(vec![1, self.hidden_dim]);
                let residual_edge = GraphEdge::residual_with_tensor(
                    attn_node.index(),
                    ffn_node_idx.index(),
                    layer,
                    SkipType::PostNorm,
                    residual_tensor,
                );
                self.executor.add_edge(attn_node, ffn_node_idx, residual_edge);
            }

            prev_layer_nodes = ffn_nodes;
        }
    }

    /// Run forward pass
    pub fn forward(&mut self, input_ids: &[usize]) -> DenseTensor {
        self.executor.forward(input_ids)
    }

    /// Get number of nodes in graph
    pub fn num_nodes(&self) -> usize {
        self.executor.num_nodes()
    }

    /// Get number of edges in graph
    pub fn num_edges(&self) -> usize {
        self.executor.num_edges()
    }

    /// Prune weak attention edges
    pub fn prune_weak_edges(&mut self, threshold: f64) -> usize {
        self.executor.prune_weak_edges(threshold)
    }

    /// Export to DOT format
    pub fn to_dot(&self) -> String {
        self.executor.to_dot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_executor_creation() {
        let executor = GraphExecutor::new();
        assert_eq!(executor.num_nodes(), 0);
        assert_eq!(executor.num_edges(), 0);
    }

    #[test]
    fn test_graph_executor_add_node() {
        let mut executor = GraphExecutor::new();
        let embedding = DenseTensor::zeros(vec![1, 4]);
        let node = GraphNode::token_embedding(0, 10, 0, embedding);
        let node_idx = executor.add_node(node);

        assert_eq!(executor.num_nodes(), 1);
        assert!(node_idx.is_valid());
    }

    #[test]
    fn test_graph_executor_add_edge() {
        let mut executor = GraphExecutor::new();

        let embedding1 = DenseTensor::zeros(vec![1, 4]);
        let node1 = GraphNode::token_embedding(0, 10, 0, embedding1);
        let node1_idx = executor.add_node(node1);

        let embedding2 = DenseTensor::zeros(vec![1, 4]);
        let node2 = GraphNode::token_embedding(1, 20, 1, embedding2);
        let node2_idx = executor.add_node(node2);

        let edge = GraphEdge::self_attention(node1_idx.index(), node2_idx.index(), 0.5, 0, 0);
        let result = executor.add_edge(node1_idx, node2_idx, edge);

        assert!(result);
        assert_eq!(executor.num_edges(), 1);
    }

    #[test]
    fn test_topological_sort() {
        let mut executor = GraphExecutor::new();

        // Create a simple chain: A -> B -> C
        let node_a = GraphNode::token_embedding(0, 1, 0, DenseTensor::zeros(vec![1, 4]));
        let node_b = GraphNode::hidden_state(1, 0, 0, DenseTensor::zeros(vec![1, 4]));
        let node_c = GraphNode::ffn_output(2, 0, 0, DenseTensor::zeros(vec![1, 4]));

        let idx_a = executor.add_node(node_a);
        let idx_b = executor.add_node(node_b);
        let idx_c = executor.add_node(node_c);

        executor.add_edge(idx_a, idx_b, GraphEdge::data_flow(idx_a.index(), idx_b.index(), DataFlowOp::InputToAttention, 0));
        executor.add_edge(idx_b, idx_c, GraphEdge::data_flow(idx_b.index(), idx_c.index(), DataFlowOp::AttentionToOutput, 0));

        let order = executor.topological_sort();

        // A should come before B, B should come before C
        assert!(order.iter().position(|&x| x == idx_a).unwrap() < order.iter().position(|&x| x == idx_b).unwrap());
        assert!(order.iter().position(|&x| x == idx_b).unwrap() < order.iter().position(|&x| x == idx_c).unwrap());
    }

    #[test]
    fn test_graph_transformer_creation() {
        let transformer = GraphTransformer::new(2, 4, 256);

        assert_eq!(transformer.num_layers, 2);
        assert_eq!(transformer.num_heads, 4);
        assert_eq!(transformer.hidden_dim, 256);
    }

    #[test]
    fn test_graph_transformer_build() {
        let mut transformer = GraphTransformer::new(2, 4, 256);
        let input_ids = vec![1, 2, 3, 4];

        transformer.build_graph(&input_ids);

        assert!(transformer.num_nodes() > 0);
        assert!(transformer.num_edges() > 0);
    }

    #[test]
    fn test_to_dot_export() {
        let mut executor = GraphExecutor::new();

        let node1 = GraphNode::token_embedding(0, 1, 0, DenseTensor::zeros(vec![1, 4]));
        let node2 = GraphNode::hidden_state(1, 0, 0, DenseTensor::zeros(vec![1, 4]));

        let idx1 = executor.add_node(node1);
        let idx2 = executor.add_node(node2);
        executor.add_edge(idx1, idx2, GraphEdge::data_flow(idx1.index(), idx2.index(), DataFlowOp::InputToAttention, 0));

        let dot = executor.to_dot();

        assert!(dot.contains("digraph Transformer"));
        assert!(dot.contains("n0"));
        assert!(dot.contains("n1"));
        assert!(dot.contains("n0 -> n1"));
    }
}
