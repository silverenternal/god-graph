//! 边列表格式导出

use crate::graph::traits::GraphQuery;
use crate::graph::Graph;
use std::fmt::Write;

/// 将图导出为边列表格式字符串
pub fn to_edge_list<T, E>(graph: &Graph<T, E>) -> String
where
    T: std::fmt::Display,
    E: std::fmt::Display,
{
    let mut output = String::new();

    for edge in graph.edges() {
        writeln!(
            &mut output,
            "{} {} {}",
            edge.source().index(),
            edge.target().index(),
            edge.data()
        )
        .unwrap();
    }

    output
}
