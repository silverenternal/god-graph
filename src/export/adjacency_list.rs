//! 邻接表格式导出

use crate::graph::Graph;
use crate::graph::traits::GraphQuery;
use std::fmt::Write;

/// 将图导出为邻接表格式字符串
pub fn to_adjacency_list<T, E>(graph: &Graph<T, E>) -> String
where
    T: std::fmt::Display,
{
    let mut output = String::new();

    for node in graph.nodes() {
        write!(&mut output, "{}:", node.data()).unwrap();
        for neighbor in graph.neighbors(node.index()) {
            write!(&mut output, " {}", neighbor.index()).unwrap();
        }
        writeln!(&mut output).unwrap();
    }

    output
}
