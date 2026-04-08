//! 补充测试：社区发现算法
//!
//! 目标：提升 community.rs 模块的测试覆盖率到 80%+

#[cfg(test)]
mod tests {
    use god_graph::algorithms::community::connected_components;
    use god_graph::graph::Graph;
    use god_graph::graph::traits::GraphOps;

    /// 测试连通分量 - 基本场景
    #[test]
    fn test_connected_components_basic() {
        let mut graph = Graph::<i32, f64>::undirected();

        // 创建两个独立的连通分量
        let comp1: Vec<_> = (0..3).map(|i| graph.add_node(i).unwrap()).collect();
        let comp2: Vec<_> = (3..6).map(|i| graph.add_node(i).unwrap()).collect();

        // 分量 1: 三角形
        graph.add_edge(comp1[0], comp1[1], 1.0).unwrap();
        graph.add_edge(comp1[1], comp1[2], 1.0).unwrap();
        graph.add_edge(comp1[2], comp1[0], 1.0).unwrap();

        // 分量 2: 链式
        graph.add_edge(comp2[0], comp2[1], 1.0).unwrap();
        graph.add_edge(comp2[1], comp2[2], 1.0).unwrap();

        let components = connected_components(&graph);
        assert_eq!(components.len(), 2);
    }

    /// 测试连通分量 - 空图
    #[test]
    fn test_connected_components_empty() {
        let graph = Graph::<i32, f64>::undirected();
        let components = connected_components(&graph);
        assert_eq!(components.len(), 0);
    }

    /// 测试连通分量 - 单节点
    #[test]
    fn test_connected_components_single() {
        let mut graph = Graph::<i32, f64>::undirected();
        let _ = graph.add_node(42).unwrap();

        let components = connected_components(&graph);
        assert_eq!(components.len(), 1);
    }

    /// 测试连通分量 - 完全连通
    #[test]
    fn test_connected_components_fully_connected() {
        let mut graph = Graph::<i32, f64>::undirected();

        let nodes: Vec<_> = (0..5).map(|i| graph.add_node(i).unwrap()).collect();

        // 完全图
        for i in 0..5 {
            for j in (i + 1)..5 {
                graph.add_edge(nodes[i], nodes[j], 1.0).unwrap();
            }
        }

        let components = connected_components(&graph);
        assert_eq!(components.len(), 1);
    }

    /// 测试连通分量 - 线性链
    #[test]
    fn test_connected_components_linear() {
        let mut graph = Graph::<i32, f64>::undirected();

        let nodes: Vec<_> = (0..10).map(|i| graph.add_node(i).unwrap()).collect();

        // 链式结构
        for i in 0..9 {
            graph.add_edge(nodes[i], nodes[i + 1], 1.0).unwrap();
        }

        let components = connected_components(&graph);
        assert_eq!(components.len(), 1);
    }
}
