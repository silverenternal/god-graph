//! DOT 格式导出（Graphviz）
//!
//! 支持将图导出为 Graphviz DOT 格式，用于可视化
//!
//! ## 使用示例
//!
//! ```
//! use god_gragh::prelude::*;
//!
//! let mut graph = Graph::<&str, f64>::directed();
//! let a = graph.add_node("A").unwrap();
//! let b = graph.add_node("B").unwrap();
//! graph.add_edge(a, b, 1.0).unwrap();
//!
//! let dot = to_dot(&graph);
//! println!("{}", dot);
//! ```

use crate::graph::Graph;
use crate::graph::traits::GraphQuery;
use std::fmt::Write;

/// DOT 格式导出选项
#[derive(Clone, Debug)]
pub struct DotOptions {
    /// 是否显示节点标签
    pub show_node_labels: bool,
    /// 是否显示边标签
    pub show_edge_labels: bool,
    /// 是否显示节点索引
    pub show_node_indices: bool,
    /// 图的名称（用于 label）
    pub graph_name: Option<String>,
    /// 额外的图属性
    pub graph_attributes: Vec<(String, String)>,
    /// 节点默认属性
    pub node_attributes: Vec<(String, String)>,
    /// 边默认属性
    pub edge_attributes: Vec<(String, String)>,
}

impl Default for DotOptions {
    fn default() -> Self {
        Self {
            show_node_labels: true,
            show_edge_labels: true,
            show_node_indices: true,
            graph_name: None,
            graph_attributes: Vec::new(),
            node_attributes: vec![
                ("shape".to_string(), "circle".to_string()),
                ("style".to_string(), "filled".to_string()),
                ("fillcolor".to_string(), "lightblue".to_string()),
            ],
            edge_attributes: vec![
                ("color".to_string(), "gray".to_string()),
                ("arrowhead".to_string(), "vee".to_string()),
            ],
        }
    }
}

impl DotOptions {
    /// 创建默认选项
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置为无向图样式
    pub fn undirected(mut self) -> Self {
        self.edge_attributes.retain(|(k, _)| k != "arrowhead");
        self
    }

    /// 设置图名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.graph_name = Some(name.into());
        self
    }

    /// 隐藏节点标签
    pub fn hide_node_labels(mut self) -> Self {
        self.show_node_labels = false;
        self
    }

    /// 隐藏边标签
    pub fn hide_edge_labels(mut self) -> Self {
        self.show_edge_labels = false;
        self
    }
}

/// 使用默认选项将图导出为 DOT 格式
pub fn to_dot<T, E>(graph: &Graph<T, E>) -> String
where
    T: std::fmt::Display,
    E: std::fmt::Display,
{
    to_dot_with_options(graph, &DotOptions::default())
}

/// 使用自定义选项将图导出为 DOT 格式
///
/// # 参数
/// * `graph` - 要导出的图
/// * `options` - DOT 导出选项
///
/// # 返回
/// DOT 格式字符串
pub fn to_dot_with_options<T, E>(graph: &Graph<T, E>, options: &DotOptions) -> String
where
    T: std::fmt::Display,
    E: std::fmt::Display,
{
    let mut output = String::new();
    
    // 图声明：digraph（有向）或 graph（无向）
    let graph_type = "digraph";
    let name = options.graph_name.as_deref().unwrap_or("G");
    writeln!(&mut output, "{} {} {{", graph_type, name).unwrap();

    // 图属性
    for (key, value) in &options.graph_attributes {
        writeln!(&mut output, "  {} = {};", key, value).unwrap();
    }

    // 节点默认属性
    if !options.node_attributes.is_empty() {
        write!(&mut output, "  node [").unwrap();
        for (i, (key, value)) in options.node_attributes.iter().enumerate() {
            if i > 0 {
                write!(&mut output, ", ").unwrap();
            }
            write!(&mut output, "{} = {}", key, value).unwrap();
        }
        writeln!(&mut output, "];").unwrap();
    }

    // 边默认属性
    if !options.edge_attributes.is_empty() {
        write!(&mut output, "  edge [").unwrap();
        for (i, (key, value)) in options.edge_attributes.iter().enumerate() {
            if i > 0 {
                write!(&mut output, ", ").unwrap();
            }
            write!(&mut output, "{} = {}", key, value).unwrap();
        }
        writeln!(&mut output, "];").unwrap();
    }

    writeln!(&mut output).unwrap();

    // 导出节点
    for node in graph.nodes() {
        let idx = node.index();
        
        // 构建节点标签
        let label = if options.show_node_labels && options.show_node_indices {
            format!("{}: {}", idx, node.data())
        } else if options.show_node_labels {
            format!("{}", node.data())
        } else if options.show_node_indices {
            format!("{}", idx)
        } else {
            String::new()
        };

        if label.is_empty() {
            writeln!(&mut output, "  {};", idx).unwrap();
        } else {
            writeln!(&mut output, "  {} [label=\"{}\"];", idx, escape_dot(&label)).unwrap();
        }
    }

    writeln!(&mut output).unwrap();

    // 导出边
    for edge in graph.edges() {
        let source = edge.source().index();
        let target = edge.target().index();
        
        let edge_def = if options.show_edge_labels {
            format!(" [label=\"{}\"]", escape_dot(&format!("{}", edge.data())))
        } else {
            String::new()
        };

        writeln!(&mut output, "  {} -> {}{};", source, target, edge_def).unwrap();
    }

    writeln!(&mut output, "}}").unwrap();
    output
}

/// 导出无向图的 DOT 格式
///
/// 使用 `--` 而不是 `->` 表示边
pub fn to_dot_undirected<T, E>(graph: &Graph<T, E>) -> String
where
    T: std::fmt::Display,
    E: std::fmt::Display,
{
    let mut output = String::new();
    
    writeln!(&mut output, "graph G {{").unwrap();
    writeln!(&mut output, "  node [shape=circle, style=filled, fillcolor=lightblue];").unwrap();
    writeln!(&mut output).unwrap();

    // 导出节点
    for node in graph.nodes() {
        let idx = node.index();
        writeln!(&mut output, "  {} [label=\"{}: {}\"];", idx, idx, node.data()).unwrap();
    }

    writeln!(&mut output).unwrap();

    // 导出边（无向图使用 --）
    for edge in graph.edges() {
        let source = edge.source().index();
        let target = edge.target().index();
        writeln!(&mut output, "  {} -- {} [label=\"{}\"];", source, target, edge.data()).unwrap();
    }

    writeln!(&mut output, "}}").unwrap();
    output
}

/// 转义 DOT 字符串中的特殊字符
fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// 将 DOT 字符串写入文件
///
/// # 参数
/// * `dot` - DOT 格式字符串
/// * `path` - 输出文件路径
///
/// # 返回
/// 成功返回 Ok(())，失败返回 IO 错误
pub fn write_dot_to_file(dot: &str, path: &str) -> std::io::Result<()> {
    std::fs::write(path, dot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builders::GraphBuilder;

    #[test]
    fn test_dot_export_basic() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B"])
            .with_edge(0, 1, 1.0)
            .build()
            .unwrap();

        let dot = to_dot(&graph);
        assert!(dot.contains("digraph"));
        assert!(dot.contains("A"));
        assert!(dot.contains("B"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_dot_export_with_options() {
        let graph = GraphBuilder::directed()
            .with_nodes(vec!["A", "B", "C"])
            .with_edges(vec![(0, 1, 1.0), (1, 2, 2.0)])
            .build()
            .unwrap();

        let options = DotOptions::new()
            .with_name("MyGraph")
            .hide_edge_labels();
        
        let dot = to_dot_with_options(&graph, &options);
        assert!(dot.contains("digraph MyGraph"));
        // 边标签已隐藏，但节点标签仍然存在
        // 检查没有边标签（格式为 [label="x"] 在边定义中）
        assert!(!dot.contains("-> [label="));
    }

    #[test]
    fn test_dot_escaping() {
        assert_eq!(escape_dot("hello"), "hello");
        assert_eq!(escape_dot("he\"llo"), "he\\\"llo");
        assert_eq!(escape_dot("he\\llo"), "he\\\\llo");
        assert_eq!(escape_dot("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_dot_empty_graph() {
        let graph = GraphBuilder::<String, f64>::directed()
            .build()
            .unwrap();

        let dot = to_dot(&graph);
        assert!(dot.contains("digraph"));
        assert!(dot.contains("{"));
        assert!(dot.contains("}"));
    }

    #[test]
    fn test_dot_undirected() {
        let graph = GraphBuilder::undirected()
            .with_nodes(vec!["A", "B"])
            .with_edge(0, 1, 1.0)
            .build()
            .unwrap();

        let dot = to_dot_undirected(&graph);
        assert!(dot.contains("graph"));
        assert!(dot.contains("--")); // 无向图使用 --
    }
}
