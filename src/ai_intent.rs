//! AI 意图映射模块
//!
//! 提供自然语言意图到 API 的映射，帮助 AI Agent 理解用户请求并调用正确的 API
//!
//! # 使用示例
//!
//! ```rust
//! use god_graph::ai_intent::{intent_to_api, IntentCategory};
//!
//! // 意图识别
//! let user_input = "find the most important people in this network";
//! if let Some(api) = intent_to_api(user_input) {
//!     println!("Suggested API: {}", api);
//! }
//!
//! // 分类识别
//! let category = IntentCategory::from_phrase("find communities");
//! assert_eq!(category, Some(IntentCategory::CommunityDetection));
//! ```

/// 意图类别（AI 友好）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentCategory {
    /// 创建图
    GraphCreation,
    /// 添加节点/边
    GraphModification,
    /// 图遍历
    GraphTraversal,
    /// 最短路径
    ShortestPath,
    /// 中心性分析
    CentralityAnalysis,
    /// 社区发现
    CommunityDetection,
    /// 图属性检查
    GraphProperties,
    /// 导出/可视化
    ExportVisualization,
    /// Tensor 操作
    TensorOperation,
    /// LLM 模型加载
    LlmLoad,
    /// LLM 优化
    LlmOptimize,
}

impl IntentCategory {
    /// 从用户短语识别意图类别
    pub fn from_phrase(phrase: &str) -> Option<Self> {
        let phrase_lower = phrase.to_lowercase();
        
        // Graph Creation
        if phrase_lower.contains("create") && phrase_lower.contains("graph")
            || phrase_lower.contains("build") && phrase_lower.contains("network")
            || phrase_lower.contains("initialize") && phrase_lower.contains("graph")
        {
            return Some(IntentCategory::GraphCreation);
        }
        
        // Graph Modification
        if phrase_lower.contains("add") && (phrase_lower.contains("node") || phrase_lower.contains("vertex"))
            || phrase_lower.contains("add") && (phrase_lower.contains("edge") || phrase_lower.contains("link"))
            || phrase_lower.contains("connect") && phrase_lower.contains("node")
        {
            return Some(IntentCategory::GraphModification);
        }
        
        // Graph Traversal
        if phrase_lower.contains("traverse") || phrase_lower.contains("visit all")
            || phrase_lower.contains("walk through") || phrase_lower.contains("bfs")
            || phrase_lower.contains("dfs") || phrase_lower.contains("traversal")
        {
            return Some(IntentCategory::GraphTraversal);
        }
        
        // Shortest Path
        if phrase_lower.contains("shortest path") || phrase_lower.contains("minimum distance")
            || phrase_lower.contains("find path") || phrase_lower.contains("calculate distance")
        {
            return Some(IntentCategory::ShortestPath);
        }
        
        // Centrality Analysis
        if phrase_lower.contains("important") || phrase_lower.contains("key person")
            || phrase_lower.contains("influencer") || phrase_lower.contains("central")
            || phrase_lower.contains("rank") && phrase_lower.contains("node")
            || phrase_lower.contains("pagerank") || phrase_lower.contains("centrality")
        {
            return Some(IntentCategory::CentralityAnalysis);
        }
        
        // Community Detection
        if phrase_lower.contains("community") || phrase_lower.contains("cluster")
            || phrase_lower.contains("group") || phrase_lower.contains("segment")
            || phrase_lower.contains("connected component") || phrase_lower.contains("detect")
        {
            return Some(IntentCategory::CommunityDetection);
        }
        
        // Graph Properties
        if phrase_lower.contains("is connected") || phrase_lower.contains("has cycle")
            || phrase_lower.contains("density") || phrase_lower.contains("diameter")
            || phrase_lower.contains("is tree") || phrase_lower.contains("is dag")
            || phrase_lower.contains("property") || phrase_lower.contains("characteristic")
        {
            return Some(IntentCategory::GraphProperties);
        }
        
        // Export / Visualization
        if phrase_lower.contains("export") || phrase_lower.contains("visualize")
            || phrase_lower.contains("draw") || phrase_lower.contains("save") && phrase_lower.contains("graph")
            || phrase_lower.contains("dot") || phrase_lower.contains("graphviz")
        {
            return Some(IntentCategory::ExportVisualization);
        }
        
        // Tensor Operation
        if phrase_lower.contains("tensor") || phrase_lower.contains("matrix")
            || phrase_lower.contains("multiply") || phrase_lower.contains("transpose")
        {
            return Some(IntentCategory::TensorOperation);
        }
        
        // LLM Load
        if phrase_lower.contains("load") && phrase_lower.contains("model")
            || phrase_lower.contains("import") && phrase_lower.contains("safetensors")
        {
            return Some(IntentCategory::LlmLoad);
        }
        
        // LLM Optimize
        if phrase_lower.contains("optimize") && phrase_lower.contains("model")
            || phrase_lower.contains("prune") || phrase_lower.contains("compress")
            || phrase_lower.contains("defect") && phrase_lower.contains("model")
        {
            return Some(IntentCategory::LlmOptimize);
        }
        
        None
    }
    
    /// 获取该类别的推荐 API 列表
    pub fn recommended_apis(self) -> &'static [&'static str] {
        match self {
            IntentCategory::GraphCreation => &[
                "Graph::<T, E>::directed()",
                "Graph::<T, E>::undirected()",
            ],
            IntentCategory::GraphModification => &[
                "graph.add_node(data)",
                "graph.add_edge(from, to, data)",
            ],
            IntentCategory::GraphTraversal => &[
                "bfs(graph, start, visitor)",
                "dfs(graph, start, visitor)",
                "topological_sort(graph)",
            ],
            IntentCategory::ShortestPath => &[
                "dijkstra(graph, start)",
                "astar(graph, start, end, heuristic)",
                "bellman_ford(graph, start)",
            ],
            IntentCategory::CentralityAnalysis => &[
                "pagerank(graph, damping, iterations)",
                "betweenness_centrality(graph)",
                "closeness_centrality(graph)",
                "degree_centrality(graph)",
            ],
            IntentCategory::CommunityDetection => &[
                "connected_components(graph)",
                "louvain(graph)",
            ],
            IntentCategory::GraphProperties => &[
                "is_connected(graph)",
                "has_cycle(graph)",
                "density(graph)",
                "is_dag(graph)",
                "is_tree(graph)",
                "diameter(graph)",
            ],
            IntentCategory::ExportVisualization => &[
                "to_dot(graph)",
                "to_adjacency_list(graph)",
                "to_edge_list(graph)",
            ],
            IntentCategory::TensorOperation => &[
                "DenseTensor::from_vec(data, shape)",
                "tensor.matmul(&other)",
                "tensor.t()",
            ],
            IntentCategory::LlmLoad => &[
                "ModelSwitch::load_from_safetensors(path)",
                "ModelSwitch::validate_topology(graph)",
            ],
            IntentCategory::LlmOptimize => &[
                "CadStyleEditor::detect_defects()",
                "TensorRingCompressor::compress_graph(graph)",
                "LieGroupOptimizer::orthogonalize_weights(graph)",
            ],
        }
    }
}

/// 意图到 API 的映射表
///
/// 格式：`(触发关键词，API 名称，优先级)`
/// 优先级：1=最高，3=最低
pub const INTENT_MAP: &[(&str, &str, u8)] = &[
    // Graph Creation
    ("create graph", "Graph::directed", 1),
    ("build network", "Graph::directed", 1),
    ("initialize graph", "Graph::new", 1),
    
    // Centrality Analysis
    ("important node", "pagerank", 1),
    ("important nodes", "pagerank", 1),
    ("key person", "betweenness_centrality", 2),
    ("key people", "betweenness_centrality", 2),
    ("influencer", "pagerank", 1),
    ("central node", "closeness_centrality", 2),
    ("rank node", "pagerank", 1),
    ("pagerank", "pagerank", 1),
    ("centrality", "degree_centrality", 3),
    
    // Community Detection
    ("community", "connected_components", 1),
    ("detect community", "connected_components", 1),
    ("detect communities", "connected_components", 1),
    ("cluster", "louvain", 2),
    ("group", "connected_components", 1),
    ("segment", "connected_components", 2),
    ("connected component", "connected_components", 1),
    
    // Shortest Path
    ("shortest path", "dijkstra", 1),
    ("minimum distance", "dijkstra", 1),
    ("find path", "dijkstra", 1),
    ("calculate distance", "dijkstra", 1),
    ("a star", "astar", 2),
    ("bellman", "bellman_ford", 3),
    
    // Graph Traversal
    ("traverse", "bfs", 1),
    ("visit all", "bfs", 1),
    ("walk through", "dfs", 2),
    ("bfs", "bfs", 1),
    ("dfs", "dfs", 1),
    ("topological", "topological_sort", 1),
    
    // Graph Properties
    ("is connected", "is_connected", 1),
    ("has cycle", "has_cycle", 1),
    ("density", "density", 1),
    ("diameter", "diameter", 1),
    ("is tree", "is_tree", 1),
    ("is dag", "is_dag", 1),
    ("property", "density", 3),
    
    // Export
    ("export", "to_dot", 1),
    ("visualize", "to_dot", 1),
    ("draw", "to_dot", 2),
    ("dot", "to_dot", 1),
    ("graphviz", "to_dot", 1),
    
    // Tensor
    ("create tensor", "DenseTensor::from_vec", 1),
    ("matrix multiply", "tensor.matmul", 1),
    ("transpose", "tensor.t", 1),
    
    // LLM
    ("load model", "ModelSwitch::load_from_safetensors", 1),
    ("import safetensors", "ModelSwitch::load_from_safetensors", 1),
    ("validate model", "ModelSwitch::validate_topology", 1),
    ("optimize model", "CadStyleEditor::detect_defects", 1),
    ("compress", "TensorRingCompressor::compress_graph", 1),
    ("defect", "CadStyleEditor::detect_defects", 1),
];

/// 从用户输入映射到 API
///
/// 返回最匹配的 API 名称
///
/// # 示例
///
/// ```rust
/// use god_graph::ai_intent::intent_to_api;
///
/// assert_eq!(intent_to_api("find important nodes"), Some("pagerank"));
/// assert_eq!(intent_to_api("find communities"), Some("connected_components"));
/// ```
pub fn intent_to_api(input: &str) -> Option<&'static str> {
    let input_lower = input.to_lowercase();
    
    // 查找最佳匹配（优先级最高的）
    let mut best_match: Option<(&str, u8)> = None;
    
    for &(trigger, api, priority) in INTENT_MAP {
        if input_lower.contains(trigger) {
            match best_match {
                None => best_match = Some((api, priority)),
                Some((_, best_priority)) if priority < best_priority => {
                    best_match = Some((api, priority));
                }
                _ => {}
            }
        }
    }
    
    best_match.map(|(api, _)| api)
}

/// 获取 API 的详细描述
pub fn api_description(api: &str) -> &'static str {
    match api {
        // Graph Creation
        "Graph::directed" => "Create a directed graph",
        "Graph::undirected" => "Create an undirected graph",
        "Graph::new" => "Create a new graph with specified type",
        
        // Centrality
        "pagerank" => "PageRank algorithm for node importance ranking",
        "betweenness_centrality" => "Betweenness centrality for bridge/hub detection",
        "closeness_centrality" => "Closeness centrality for reachability analysis",
        "degree_centrality" => "Degree centrality for connection count",
        
        // Community
        "connected_components" => "Find connected components (basic clustering)",
        "louvain" => "Louvain algorithm for modularity-based community detection",
        
        // Shortest Path
        "dijkstra" => "Dijkstra's algorithm for shortest path (non-negative weights)",
        "astar" => "A* search algorithm with heuristic",
        "bellman_ford" => "Bellman-Ford algorithm (supports negative weights)",
        
        // Traversal
        "bfs" => "Breadth-first search traversal",
        "dfs" => "Depth-first search traversal",
        "topological_sort" => "Topological sorting for DAGs",
        
        // Properties
        "is_connected" => "Check if graph is connected",
        "has_cycle" => "Check if graph contains a cycle",
        "density" => "Calculate graph density",
        "diameter" => "Calculate graph diameter",
        "is_tree" => "Check if graph is a tree",
        "is_dag" => "Check if graph is a Directed Acyclic Graph",
        
        // Export
        "to_dot" => "Export to DOT format (Graphviz)",
        "to_adjacency_list" => "Export to adjacency list format",
        "to_edge_list" => "Export to edge list format",
        
        // Tensor
        "DenseTensor::from_vec" => "Create dense tensor from vector",
        "tensor.matmul" => "Matrix multiplication",
        "tensor.t" => "Matrix transpose",
        
        // LLM
        "ModelSwitch::load_from_safetensors" => "Load model from Safetensors file",
        "ModelSwitch::validate_topology" => "Validate model topology",
        "CadStyleEditor::detect_defects" => "Detect topology defects in model",
        "TensorRingCompressor::compress_graph" => "Compress model using tensor ring decomposition",
        
        _ => "Unknown API",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_intent_category_recognition() {
        assert_eq!(
            IntentCategory::from_phrase("find the most important nodes"),
            Some(IntentCategory::CentralityAnalysis)
        );
        assert_eq!(
            IntentCategory::from_phrase("detect communities"),
            Some(IntentCategory::CommunityDetection)
        );
        assert_eq!(
            IntentCategory::from_phrase("find shortest path"),
            Some(IntentCategory::ShortestPath)
        );
        assert_eq!(
            IntentCategory::from_phrase("create a graph"),
            Some(IntentCategory::GraphCreation)
        );
    }
    
    #[test]
    fn test_intent_to_api() {
        assert_eq!(intent_to_api("find important nodes"), Some("pagerank"));
        assert_eq!(intent_to_api("find key people"), Some("betweenness_centrality"));
        assert_eq!(intent_to_api("detect communities"), Some("connected_components"));
        assert_eq!(intent_to_api("find shortest path"), Some("dijkstra"));
        assert_eq!(intent_to_api("visualize the graph"), Some("to_dot"));
    }
    
    #[test]
    fn test_api_description() {
        assert!(api_description("pagerank").contains("PageRank"));
        assert!(api_description("dijkstra").contains("Dijkstra"));
        assert!(api_description("to_dot").contains("DOT"));
    }
    
    #[test]
    fn test_recommended_apis() {
        let apis = IntentCategory::CentralityAnalysis.recommended_apis();
        assert!(apis.contains(&"pagerank(graph, damping, iterations)"));
        
        let apis = IntentCategory::CommunityDetection.recommended_apis();
        assert!(apis.contains(&"connected_components(graph)"));
    }
}
