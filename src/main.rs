//! God-Graph 示例程序

use god_graph::graph::traits::GraphQuery;
use god_graph::prelude::*;

fn main() {
    println!("🚀 God-Graph 高性能图库\n");

    // 示例 1：基本图操作
    println!("=== 示例 1：基本图操作 ===");
    basic_graph_example();

    // 示例 2：图遍历
    println!("\n=== 示例 2：图遍历 ===");
    traversal_example();

    // 示例 3：最短路径
    println!("\n=== 示例 3：最短路径 (Dijkstra) ===");
    shortest_path_example();

    // 示例 4：PageRank
    println!("\n=== 示例 4：PageRank ===");
    pagerank_example();

    // 示例 5：图生成器
    println!("\n=== 示例 5：网格图生成 ===");
    generator_example();

    // 示例 6：DOT 导出
    println!("\n=== 示例 6：DOT 格式导出 ===");
    export_example();
}

/// 基本图操作示例
fn basic_graph_example() {
    // 使用构建器创建图
    let graph = GraphBuilder::directed()
        .with_node("北京")
        .with_node("上海")
        .with_node("广州")
        .with_node("深圳")
        .with_edge(0, 1, 1200.0) // 北京 -> 上海
        .with_edge(0, 2, 2100.0) // 北京 -> 广州
        .with_edge(1, 3, 1400.0) // 上海 -> 深圳
        .with_edge(2, 3, 150.0) // 广州 -> 深圳
        .build()
        .expect("图构建失败");

    println!("节点数：{}", graph.node_count());
    println!("边数：{}", graph.edge_count());

    // 访问节点
    let beijing = graph.nodes().next().unwrap().index();
    println!("城市：{}", graph[beijing]);

    // 遍历邻居
    print!("北京的邻居：");
    for neighbor in graph.neighbors(beijing) {
        print!("{} ", graph[neighbor]);
    }
    println!();
}

/// 图遍历示例
fn traversal_example() {
    let graph = GraphBuilder::directed()
        .with_nodes(vec!["A", "B", "C", "D", "E"])
        .with_edges(vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
        ])
        .build()
        .unwrap();

    let start = graph.nodes().next().unwrap().index();

    // DFS
    print!("DFS 遍历：");
    dfs(&graph, start, |node| {
        print!("{} ", graph[node]);
        true
    });
    println!();

    // BFS
    print!("BFS 遍历：");
    bfs(&graph, start, |node, depth| {
        print!("{}(d={}) ", graph[node], depth);
        true
    });
    println!();

    // 拓扑排序
    match topological_sort(&graph) {
        Ok(order) => {
            print!("拓扑排序：");
            for node in order {
                print!("{} ", graph[node]);
            }
            println!();
        }
        Err(e) => println!("拓扑排序失败：{}", e),
    }
}

/// 最短路径示例
fn shortest_path_example() {
    let graph = GraphBuilder::directed()
        .with_nodes(vec!["A", "B", "C", "D"])
        .with_edges(vec![
            (0, 1, 1.0),
            (0, 2, 4.0),
            (1, 2, 2.0),
            (1, 3, 5.0),
            (2, 3, 1.0),
        ])
        .build()
        .unwrap();

    let start = graph.nodes().next().unwrap().index();

    match dijkstra(&graph, start, |_, _, w| *w) {
        Ok(distances) => {
            println!("从 A 到各节点的最短距离:");
            for (node, dist) in &distances {
                println!("  {}: {:.1}", graph[*node], dist);
            }
        }
        Err(e) => println!("Dijkstra 失败：{}", e),
    }
}

/// PageRank 示例
fn pagerank_example() {
    let graph = GraphBuilder::directed()
        .with_nodes(vec!["网页 A", "网页 B", "网页 C", "网页 D"])
        .with_edges(vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (2, 3, 1.0),
            (3, 2, 1.0),
        ])
        .build()
        .unwrap();

    let ranks = pagerank(&graph, 0.85, 20);

    println!("PageRank 分数:");
    let mut sorted: Vec<_> = ranks.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (node, rank) in sorted {
        println!("  {}: {:.4}", graph[*node], rank);
    }
}

/// 图生成器示例
fn generator_example() {
    // 生成 3x3 网格图
    let graph = grid_graph::<String>(3, 3, false);
    println!("3x3 网格图:");
    println!("  节点数：{}", graph.node_count());
    println!("  边数：{}", graph.edge_count());

    // 生成完全图 K5
    let graph = complete_graph::<String>(5);
    println!("\n完全图 K5:");
    println!("  节点数：{}", graph.node_count());
    println!("  边数：{}", graph.edge_count());
}

/// DOT 导出示例
fn export_example() {
    let graph = GraphBuilder::directed()
        .with_node("A")
        .with_node("B")
        .with_node("C")
        .with_edge(0, 1, 1.0)
        .with_edge(1, 2, 1.0)
        .with_edge(2, 0, 1.0)
        .build()
        .unwrap();

    let dot = to_dot(&graph);
    println!("DOT 格式:");
    println!("{}", dot);
}
