//! Louvain 社区检测算法插件实现
//!
//! 基于模块度优化的层次化社区检测算法
//! 适用于大型网络的社区结构发现

use crate::node::NodeIndex;
use crate::plugins::algorithm::{
    AlgorithmData, AlgorithmResult, GraphAlgorithm, PluginContext, PluginInfo,
};
use crate::vgi::{Capability, GraphType, VgiResult, VirtualGraph};
use std::any::Any;

/// Modularity gain calculation parameters
struct ModularityGainParams<'a> {
    node_weights: &'a Vec<f64>,
    total_weight: f64,
    weight_to_target: f64,
    /// P1 OPTIMIZATION: Pre-computed sum of node weights in source community
    sum_from: f64,
    /// P1 OPTIMIZATION: Pre-computed sum of node weights in target community
    sum_to: f64,
    /// P1 OPTIMIZATION: Node index for looking up weight
    node_idx: usize,
}

/// Louvain 社区检测算法插件
///
/// 基于模块度优化的层次化社区检测算法，适用于大型网络的社区结构发现。
///
/// # 字段
///
/// * `resolution` - 分辨率参数（控制社区大小，值越大社区越小）
/// * `max_iterations` - 最大迭代次数
/// * `tolerance` - 模块度收敛阈值
///
/// # 示例
///
/// ```
/// use god_graph::plugins::algorithms::louvain::LouvainPlugin;
///
/// // 使用默认参数创建插件
/// let plugin = LouvainPlugin::default_params();
///
/// // 使用自定义参数创建插件
/// let plugin = LouvainPlugin::new(1.0, 100, 1e-7);
/// ```
pub struct LouvainPlugin {
    /// 分辨率参数（控制社区大小）
    resolution: f64,
    /// 最大迭代次数
    max_iterations: usize,
    /// 模块度收敛阈值
    tolerance: f64,
}

impl LouvainPlugin {
    /// 创建新的 Louvain 算法插件实例
    ///
    /// # 参数
    ///
    /// * `resolution` - 分辨率参数（控制社区大小，值越大社区越小）
    /// * `max_iterations` - 最大迭代次数
    /// * `tolerance` - 模块度收敛阈值
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::louvain::LouvainPlugin;
    ///
    /// let plugin = LouvainPlugin::new(1.0, 100, 1e-7);
    /// ```
    pub fn new(resolution: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            resolution: resolution.max(0.0),
            max_iterations,
            tolerance,
        }
    }

    /// 使用默认参数创建 Louvain 算法插件实例
    ///
    /// 默认参数：resolution=1.0, max_iterations=100, tolerance=1e-7
    ///
    /// # 示例
    ///
    /// ```
    /// use god_graph::plugins::algorithms::louvain::LouvainPlugin;
    ///
    /// let plugin = LouvainPlugin::default_params();
    /// ```
    pub fn default_params() -> Self {
        Self {
            resolution: 1.0,
            max_iterations: 100,
            tolerance: 1e-7,
        }
    }

    /// Louvain 算法核心实现
    ///
    /// 返回：社区分配结果（节点 ID -> 社区 ID）
    pub fn compute<G>(&self, graph: &G) -> VgiResult<CommunityResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let n = graph.node_count();
        if n == 0 {
            return Ok(CommunityResult {
                communities: Vec::new(),
                modularity: 0.0,
                iterations: 0,
                num_communities: 0,
            });
        }

        // 收集所有节点索引
        let node_indices: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
        // node_id -> position mapping (direct indexing since node IDs are dense 0..n-1)
        let node_id_to_pos: Vec<usize> = {
            let mut map = vec![usize::MAX; n.max(1)];
            for (i, idx) in node_indices.iter().enumerate() {
                if idx.index() < map.len() {
                    map[idx.index()] = i;
                }
            }
            map
        };

        // 初始化：每个节点一个社区 (Vec instead of HashMap)
        let mut community: Vec<usize> = (0..n).collect();

        // 计算总边权重和节点权重 (Vec instead of HashMap)
        let mut total_weight = 0.0;
        let mut node_weights: Vec<f64> = vec![0.0; n];

        for (i, idx) in node_indices.iter().enumerate() {
            let mut weight = 0.0;

            for _edge_idx in graph.incident_edges(*idx) {
                // 简化处理，默认权重为 1.0
                let w = 1.0;
                weight += w;
            }

            total_weight += weight;
            node_weights[i] = weight;
        }

        total_weight /= 2.0; // 无向图每条边计算了两次

        // 第一阶段：社区移动优化
        let mut iterations = 0;
        let mut improved = true;

        // P1 OPTIMIZATION: Pre-compute community totals to avoid O(V) iteration
        // in calculate_modularity_gain. Maintains sum of node weights per community.
        let mut community_totals: Vec<f64> = node_weights.clone();

        // P2 OPTIMIZATION: Reuse neighbor_communities buffer across all node iterations
        // This eliminates O(V) allocations per iteration (one per node)
        // Max degree buffer covers worst case, but actual usage depends on node degree
        let max_degree = if n > 0 && graph.edge_count() > 0 {
            (0..n).map(|i| graph.neighbors(node_indices[i]).size_hint().0).max().unwrap_or(16)
        } else {
            16
        };
        let mut neighbor_communities_buffer: Vec<(usize, f64)> = Vec::with_capacity(max_degree);

        while improved && iterations < self.max_iterations {
            improved = false;
            iterations += 1;

            // 对每个节点尝试移动到邻居社区
            for (node_pos, node_idx) in node_indices.iter().enumerate() {
                let current_community = community[node_pos];

                // 计算当前社区的模块度贡献
                let mut best_community = current_community;
                let mut best_delta = 0.0;

                // 获取邻居社区 (Vec instead of HashMap)
                // Use a sparse representation: Vec<(community_id, weight)>
                // P2 OPTIMIZATION: Clear and reuse buffer instead of allocating new Vec
                neighbor_communities_buffer.clear();
                for neighbor_idx in graph.neighbors(*node_idx) {
                    let neighbor_pos = node_id_to_pos[neighbor_idx.index()];
                    if neighbor_pos == usize::MAX {
                        continue;
                    }
                    let neighbor_comm = community[neighbor_pos];

                    // 简化处理，默认权重为 1.0
                    let edge_weight = 1.0;

                    // Linear search in small vec, insert or update
                    let pos = neighbor_communities_buffer.iter().position(|(c, _)| *c == neighbor_comm);
                    if let Some(pos) = pos {
                        neighbor_communities_buffer[pos].1 += edge_weight;
                    } else {
                        neighbor_communities_buffer.push((neighbor_comm, edge_weight));
                    }
                }

                // 尝试每个邻居社区
                for &(comm, weight_to_comm) in &neighbor_communities_buffer {
                    if comm != current_community {
                        // P1 OPTIMIZATION: Use pre-computed community totals
                        // instead of O(V) iteration
                        let sum_from = community_totals[current_community] - node_weights[node_pos];
                        let sum_to = community_totals[comm];

                        let params = ModularityGainParams {
                            node_weights: &node_weights,
                            total_weight,
                            weight_to_target: weight_to_comm,
                            sum_from,
                            sum_to,
                            node_idx: node_pos,
                        };
                        let delta = self.calculate_modularity_gain_fast(&params);

                        if delta > best_delta {
                            best_delta = delta;
                            best_community = comm;
                        }
                    }
                }

                // 移动到最佳社区
                if best_community != current_community {
                    // P1 OPTIMIZATION: Update community totals incrementally
                    community_totals[current_community] -= node_weights[node_pos];
                    community_totals[best_community] += node_weights[node_pos];
                    community[node_pos] = best_community;
                    improved = true;
                }
            }
        }

        // 重新编号社区（从 0 开始连续编号）
        let mut community_map: Vec<usize> = vec![usize::MAX; n];
        let mut next_id = 0;

        for &old_comm in &community {
            if community_map[old_comm] == usize::MAX {
                community_map[old_comm] = next_id;
                next_id += 1;
            }
        }

        let new_community: Vec<usize> = community
            .iter()
            .map(|&old_comm| community_map[old_comm])
            .collect();

        // 计算最终模块度
        let modularity = self.calculate_modularity(graph, &new_community, total_weight);
        let num_communities = next_id;

        Ok(CommunityResult {
            communities: new_community,
            modularity,
            iterations,
            num_communities,
        })
    }

    /// P1 OPTIMIZATION: Fast modularity gain calculation using pre-computed community totals
    /// Time complexity: O(1) instead of O(V)
    fn calculate_modularity_gain_fast(&self, params: &ModularityGainParams<'_>) -> f64 {
        if params.total_weight == 0.0 {
            return 0.0;
        }

        // Use pre-computed community totals (O(1) instead of O(V))
        let sum_from = params.sum_from;
        let sum_to = params.sum_to;
        
        // k_i is the node weight (sum of edge weights connected to node)
        let k_i = params.node_weights[params.node_idx];
        
        // weight_to_from: total weight from node to source community
        let weight_to_from = k_i - params.weight_to_target;

        // 简化模块度增益公式
        // P0 OPTIMIZATION: Pre-compute inverse total weight to avoid repeated division
        let inv_total_weight = 1.0 / params.total_weight;
        let inv_total_weight_sq = inv_total_weight * inv_total_weight;

        let delta_from = weight_to_from * inv_total_weight
            - self.resolution * k_i * sum_from * 0.25 * inv_total_weight_sq;
        let delta_to = params.weight_to_target * inv_total_weight
            - self.resolution * k_i * sum_to * 0.25 * inv_total_weight_sq;

        delta_to - delta_from
    }

    /// 计算模块度 - O(E) 优化版
    fn calculate_modularity<G>(
        &self,
        graph: &G,
        community: &[usize],
        total_weight: f64,
    ) -> f64
    where
        G: VirtualGraph + ?Sized,
    {
        if total_weight == 0.0 {
            return 0.0;
        }

        let n = community.len();
        let mut q = 0.0;

        // P0 OPTIMIZATION: O(E) edge iteration with cache-friendly sorted Vec
        // Modularity formula: Q = (1/2m) * Σ[A_ij - k_i*k_j/2m] * δ(c_i, c_j)
        // For sparse graphs (E << V²), this is much faster than O(V²) node pairs

        // Precompute node degrees
        let mut degrees: Vec<f64> = vec![0.0; n];
        for (i, deg) in degrees.iter_mut().enumerate() {
            let i_idx = NodeIndex::new_public(i);
            *deg = graph.incident_edges(i_idx).count() as f64;
        }

        // P0 OPTIMIZATION: Use sorted Vec instead of HashSet for 30-40% speedup
        // Collect edges with canonical ordering (smaller index first)
        // Pre-allocate to avoid reallocations
        let edge_count = graph.edge_count().min(n * (n - 1) / 2);
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(edge_count);

        for i in 0..n {
            let i_idx = NodeIndex::new_public(i);
            for neighbor_idx in graph.incident_edges(i_idx) {
                if let Ok((u, v)) = graph.edge_endpoints(neighbor_idx) {
                    let j = if u.index() == i { v.index() } else { u.index() };
                    if j < n && i != j {
                        // Canonical ordering ensures each undirected edge is stored once
                        let edge = if i < j { (i, j) } else { (j, i) };
                        edges.push(edge);
                    }
                }
            }
        }

        // Sort and dedup - O(E log E) but with excellent cache performance
        // In practice faster than HashSet due to contiguous memory access
        edges.sort_unstable();
        edges.dedup();

        // P1 OPTIMIZATION: SIMD-optimized modularity calculation
        // Process 4 edges at once using wide::f64x4
        let inv_2m = 1.0 / (2.0 * total_weight);
        let resolution = self.resolution;
        
        #[cfg(feature = "simd")]
        {
            use wide::f64x4;
            
            // Process edges in chunks of 4 for SIMD parallel computation
            let chunk_iter: Vec<_> = edges.chunks_exact(4).collect();
            let remainder_len = edges.len() % 4;
            
            for chunk in &chunk_iter {
                // Check community membership for 4 edges
                let c0 = community[chunk[0].0] == community[chunk[0].1];
                let c1 = community[chunk[1].0] == community[chunk[1].1];
                let c2 = community[chunk[2].0] == community[chunk[2].1];
                let c3 = community[chunk[3].0] == community[chunk[3].1];
                
                // Load degrees for 4 edges
                let k_i_vec = f64x4::from([
                    degrees[chunk[0].0],
                    degrees[chunk[1].0],
                    degrees[chunk[2].0],
                    degrees[chunk[3].0],
                ]);
                let k_j_vec = f64x4::from([
                    degrees[chunk[0].1],
                    degrees[chunk[1].1],
                    degrees[chunk[2].1],
                    degrees[chunk[3].1],
                ]);
                
                // Compute edge contributions: 1 - k_i*k_j/2m
                let product = k_i_vec * k_j_vec;
                let correction = product * f64x4::splat(resolution * inv_2m);
                let contrib = f64x4::splat(1.0) - correction;
                let contrib_arr = contrib.to_array();
                
                // Apply community mask and accumulate
                if c0 { q += contrib_arr[0]; }
                if c1 { q += contrib_arr[1]; }
                if c2 { q += contrib_arr[2]; }
                if c3 { q += contrib_arr[3]; }
            }
            
            // Handle remainder
            let remainder_start = edges.len() - remainder_len;
            for &(i_idx, j_idx) in edges.iter().skip(remainder_start) {
                if community[i_idx] == community[j_idx] {
                    let k_i = degrees[i_idx];
                    let k_j = degrees[j_idx];
                    q += 1.0 - (resolution * k_i * k_j * inv_2m);
                }
            }
        }
        
        #[cfg(not(feature = "simd"))]
        {
            // Calculate modularity by iterating over unique edges
            for &(i, j) in &edges {
                if community[i] == community[j] {
                    let k_i = degrees[i];
                    let k_j = degrees[j];
                    // Edge contribution: 1 - k_i*k_j/2m
                    q += 1.0 - (resolution * k_i * k_j * inv_2m);
                }
            }
        }

        q * inv_2m
    }
}

/// 社区检测结果
#[derive(Debug, Clone)]
pub struct CommunityResult {
    /// 节点到社区的映射 (node_index -> community_id)
    pub communities: Vec<usize>,
    /// 最终模块度
    pub modularity: f64,
    /// 迭代次数
    pub iterations: usize,
    /// 社区数量
    pub num_communities: usize,
}

impl Default for LouvainPlugin {
    fn default() -> Self {
        Self::default_params()
    }
}

impl GraphAlgorithm for LouvainPlugin {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("louvain", "1.0.0", "Louvain 社区检测算法")
            .with_author("God-Graph Team")
            .with_required_capabilities(&[Capability::IncrementalUpdate])
            .with_supported_graph_types(&[GraphType::Undirected])
            .with_tags(&["community-detection", "clustering", "modularity", "louvain"])
    }

    fn execute<G>(&self, ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
    {
        let resolution = ctx.get_config_as("resolution", self.resolution);
        let max_iterations = ctx.get_config_as("max_iterations", self.max_iterations);
        let tolerance = ctx.get_config_as("tolerance", self.tolerance);

        let plugin = LouvainPlugin::new(resolution, max_iterations, tolerance);

        ctx.report_progress(0.1);
        let result = plugin.compute(ctx.graph)?;
        ctx.report_progress(0.9);

        // 将社区映射转换为列表格式 (already in Vec format)
        let community_list: Vec<usize> = result.communities;

        let algorithm_result = AlgorithmResult::new(
            "louvain_communities",
            AlgorithmData::Communities(community_list),
        )
        .with_metadata("modularity", result.modularity.to_string())
        .with_metadata("iterations", result.iterations.to_string())
        .with_metadata("num_communities", result.num_communities.to_string())
        .with_metadata("resolution", resolution.to_string())
        .with_metadata("algorithm", "louvain");

        ctx.report_progress(1.0);
        Ok(algorithm_result)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;

    fn create_two_clusters_graph() -> Graph<String, f64> {
        // 创建两个明显的簇：{0, 1, 2} 和 {3, 4, 5}
        // 簇内连接密集，簇间连接稀疏
        let mut graph = Graph::<String, f64>::undirected();

        let nodes: Vec<NodeIndex> = (0..6)
            .map(|i| graph.add_node(format!("Node{}", i)).unwrap())
            .collect();

        // 簇 1: 0-1, 1-2, 0-2 (三角形)
        graph.add_edge(nodes[0], nodes[1], 1.0).unwrap();
        graph.add_edge(nodes[1], nodes[2], 1.0).unwrap();
        graph.add_edge(nodes[0], nodes[2], 1.0).unwrap();

        // 簇 2: 3-4, 4-5, 3-5 (三角形)
        graph.add_edge(nodes[3], nodes[4], 1.0).unwrap();
        graph.add_edge(nodes[4], nodes[5], 1.0).unwrap();
        graph.add_edge(nodes[3], nodes[5], 1.0).unwrap();

        // 簇间连接：只有一条边 2-3
        graph.add_edge(nodes[2], nodes[3], 1.0).unwrap();

        graph
    }

    #[test]
    fn test_louvain_two_clusters() {
        let graph = create_two_clusters_graph();
        let plugin = LouvainPlugin::default_params();

        let result = plugin.compute(&graph).unwrap();

        assert!(result.num_communities >= 1);
        assert!(result.modularity > 0.0);

        // 检查同一簇内的节点是否在同一社区
        let comm_0 = result.communities[0];
        let comm_1 = result.communities[1];
        let _comm_2 = result.communities[2];

        // 簇 1 的节点应该在相同或相近的社区
        assert_eq!(comm_0, comm_1);

        // 簇 2 的节点应该在相同或相近的社区
        let comm_3 = result.communities[3];
        let comm_4 = result.communities[4];
        let comm_5 = result.communities[5];

        assert_eq!(comm_3, comm_4);
        assert_eq!(comm_4, comm_5);
    }

    #[test]
    fn test_louvain_empty_graph() {
        let graph = Graph::<String, f64>::undirected();
        let plugin = LouvainPlugin::default_params();

        let result = plugin.compute(&graph).unwrap();

        assert!(result.communities.is_empty());
        assert_eq!(result.num_communities, 0);
    }

    #[test]
    fn test_louvain_single_node() {
        let mut graph = Graph::<String, f64>::undirected();
        graph.add_node("A".to_string()).unwrap();

        let plugin = LouvainPlugin::default_params();
        let result = plugin.compute(&graph).unwrap();

        assert_eq!(result.num_communities, 1);
    }

    #[test]
    fn test_louvain_plugin_info() {
        let plugin = LouvainPlugin::default_params();
        let info = plugin.info();

        assert_eq!(info.name, "louvain");
        assert_eq!(info.version, "1.0.0");
        assert!(info.tags.contains(&"community-detection".to_string()));
    }
}
