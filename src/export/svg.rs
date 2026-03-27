//! SVG 可视化导出
//!
//! 支持将图导出为 SVG 格式，用于 Web 可视化
//!
//! ## 示例
//!
//! ```rust,ignore
//! use god_gragh::graph::Graph;
//! use god_gragh::export::svg::to_svg;
//!
//! let graph: Graph<String, f64> = Graph::directed();
//! // ... 添加节点和边
//!
//! let svg = to_svg(&graph, 800, 600);
//! std::fs::write("graph.svg", svg).unwrap();
//! ```

use crate::graph::traits::GraphQuery;
use crate::graph::Graph;
use crate::node::NodeIndex;
use std::collections::HashMap;

/// SVG 可视化选项
#[derive(Debug, Clone)]
pub struct SvgOptions {
    /// SVG 宽度（像素）
    pub width: u32,
    /// SVG 高度（像素）
    pub height: u32,
    /// 节点半径（像素）
    pub node_radius: f64,
    /// 节点填充颜色
    pub node_fill: String,
    /// 节点描边颜色
    pub node_stroke: String,
    /// 节点描边宽度
    pub node_stroke_width: f64,
    /// 边颜色
    pub edge_color: String,
    /// 边宽度
    pub edge_width: f64,
    /// 字体大小
    pub font_size: f64,
    /// 字体颜色
    pub font_color: String,
    /// 是否显示节点标签
    pub show_labels: bool,
    /// 是否显示边权重
    pub show_weights: bool,
    /// 布局算法（目前仅支持 force-directed）
    pub layout: LayoutAlgorithm,
}

/// 布局算法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutAlgorithm {
    /// 力导向布局
    ForceDirected,
    /// 圆形布局
    Circular,
    /// 层次布局（仅适用于 DAG）
    Hierarchical,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            node_radius: 20.0,
            node_fill: "#4A90D9".to_string(),
            node_stroke: "#2C5282".to_string(),
            node_stroke_width: 2.0,
            edge_color: "#A0AEC0".to_string(),
            edge_width: 1.5,
            font_size: 12.0,
            font_color: "#2D3748".to_string(),
            show_labels: true,
            show_weights: false,
            layout: LayoutAlgorithm::ForceDirected,
        }
    }
}

impl SvgOptions {
    /// 创建默认 SVG 选项
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置 SVG 尺寸
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// 设置节点半径
    pub fn with_node_radius(mut self, radius: f64) -> Self {
        self.node_radius = radius;
        self
    }

    /// 设置是否显示标签
    pub fn with_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// 设置布局算法
    pub fn with_layout(mut self, layout: LayoutAlgorithm) -> Self {
        self.layout = layout;
        self
    }
}

/// 使用默认选项将图导出为 SVG 格式
pub fn to_svg<T: std::fmt::Display, E: std::fmt::Display + Clone>(graph: &Graph<T, E>) -> String {
    to_svg_with_options(graph, &SvgOptions::default())
}

/// 使用自定义选项将图导出为 SVG 格式
pub fn to_svg_with_options<T: std::fmt::Display, E: std::fmt::Display + Clone>(
    graph: &Graph<T, E>,
    options: &SvgOptions,
) -> String {
    let mut output = String::new();

    // SVG 头部
    output.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        options.width, options.height, options.width, options.height
    ));
    output.push('\n');

    // 背景
    output.push_str(r##"<rect width="100%" height="100%" fill="#FFFFFF"/>"##);
    output.push('\n');

    // 计算节点位置
    let positions = compute_layout(graph, options);

    // 绘制边
    for edge in graph.edges() {
        let src = edge.source();
        let tgt = edge.target();
        if let (Some(&src_pos), Some(&tgt_pos)) = (positions.get(&src), positions.get(&tgt)) {
            let (x1, y1) = src_pos;
            let (x2, y2) = tgt_pos;

            output.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" fill="none"/>"#,
                x1, y1, x2, y2, options.edge_color, options.edge_width
            ));
            output.push('\n');

            // 绘制箭头（有向图）
            draw_arrow(&mut output, x1, y1, x2, y2, options);
        }
    }

    // 绘制节点
    for node in graph.nodes() {
        let idx = node.index();
        if let Some(&(x, y)) = positions.get(&idx) {
            // 节点圆形
            output.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="{}" fill="{}" stroke="{}" stroke-width="{}"/>"#,
                x,
                y,
                options.node_radius,
                options.node_fill,
                options.node_stroke,
                options.node_stroke_width
            ));
            output.push('\n');

            // 节点标签
            if options.show_labels {
                let label = format!("{}", node.data());
                output.push_str(&format!(
                    r#"<text x="{}" y="{}" font-size="{}" fill="{}" text-anchor="middle" dominant-baseline="central">{}</text>"#,
                    x, y, options.font_size, options.font_color, escape_xml(&label)
                ));
                output.push('\n');
            }
        }
    }

    output.push_str("</svg>");
    output
}

/// 计算节点布局位置
fn compute_layout<T, E: Clone>(
    graph: &Graph<T, E>,
    options: &SvgOptions,
) -> HashMap<NodeIndex, (f64, f64)> {
    let nodes: Vec<NodeIndex> = graph.nodes().map(|n| n.index()).collect();
    let n = nodes.len();

    if n == 0 {
        return HashMap::new();
    }

    match options.layout {
        LayoutAlgorithm::Circular => compute_circular_layout(&nodes, options),
        LayoutAlgorithm::Hierarchical => compute_hierarchical_layout(graph, options),
        LayoutAlgorithm::ForceDirected => compute_force_directed_layout(graph, &nodes, options),
    }
}

/// 圆形布局
fn compute_circular_layout(
    nodes: &[NodeIndex],
    options: &SvgOptions,
) -> HashMap<NodeIndex, (f64, f64)> {
    let mut positions = HashMap::new();
    let n = nodes.len();
    let center_x = options.width as f64 / 2.0;
    let center_y = options.height as f64 / 2.0;
    let radius = (options.width.min(options.height) as f64 / 2.0) * 0.8;

    for (i, &node) in nodes.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();
        positions.insert(node, (x, y));
    }

    positions
}

/// 力导向布局（简化版）
fn compute_force_directed_layout<T, E>(
    graph: &Graph<T, E>,
    nodes: &[NodeIndex],
    options: &SvgOptions,
) -> HashMap<NodeIndex, (f64, f64)> {
    let mut positions = HashMap::new();
    let n = nodes.len();
    let center_x = options.width as f64 / 2.0;
    let center_y = options.height as f64 / 2.0;

    // 初始随机位置
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    for &node in nodes.iter() {
        let mut hasher = DefaultHasher::new();
        node.hash(&mut hasher);
        let seed = hasher.finish() as f64;
        let angle = seed * 0.001;
        let radius = 50.0 + ((seed as u64) % 200) as f64;
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();
        positions.insert(node, (x, y));
    }

    // 迭代优化（简化版力导向）
    let iterations = 50;
    let repulsion = 1000.0;
    let attraction = 0.01;
    let damping = 0.85;

    let mut velocities: HashMap<NodeIndex, (f64, f64)> =
        nodes.iter().map(|&n| (n, (0.0, 0.0))).collect();

    for _ in 0..iterations {
        let mut forces: HashMap<NodeIndex, (f64, f64)> =
            nodes.iter().map(|&n| (n, (0.0, 0.0))).collect();

        // 斥力（节点之间）
        for i in 0..n {
            for j in (i + 1)..n {
                let ni = nodes[i];
                let nj = nodes[j];
                let (xi, yi) = positions[&ni];
                let (xj, yj) = positions[&nj];

                let dx = xi - xj;
                let dy = yi - yj;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);

                let force = repulsion / (dist * dist);
                let fx = force * dx / dist;
                let fy = force * dy / dist;

                let (fix, fiy) = forces.get_mut(&ni).unwrap();
                *fix += fx;
                *fiy += fy;

                let (fjx, fjy) = forces.get_mut(&nj).unwrap();
                *fjx -= fx;
                *fjy -= fy;
            }
        }

        // 引力（边连接的节点）
        for edge in graph.edges() {
            let src = edge.source();
            let tgt = edge.target();
            if positions.contains_key(&src) && positions.contains_key(&tgt) {
                let (xs, ys) = positions[&src];
                let (xt, yt) = positions[&tgt];

                let dx = xt - xs;
                let dy = yt - ys;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);

                let force = attraction * dist;
                let fx = force * dx / dist;
                let fy = force * dy / dist;

                let (fsx, fsy) = forces.get_mut(&src).unwrap();
                *fsx += fx;
                *fsy += fy;

                let (ftx, fty) = forces.get_mut(&tgt).unwrap();
                *ftx -= fx;
                *fty -= fy;
            }
        }

        // 向中心引力
        for &node in nodes {
            let (x, y) = positions[&node];
            let dx = center_x - x;
            let dy = center_y - y;
            let (fx, fy) = forces.get_mut(&node).unwrap();
            *fx += dx * 0.001;
            *fy += dy * 0.001;
        }

        // 更新位置
        for &node in nodes {
            let (fx, fy) = forces[&node];
            let (vx, vy) = velocities.get_mut(&node).unwrap();
            *vx = (*vx + fx) * damping;
            *vy = (*vy + fy) * damping;

            let (x, y) = positions.get_mut(&node).unwrap();
            *x += *vx;
            *y += *vy;

            // 边界限制
            let margin = options.node_radius + 5.0;
            *x = (*x).max(margin).min(options.width as f64 - margin);
            *y = (*y).max(margin).min(options.height as f64 - margin);
        }
    }

    positions
}

/// 层次布局（简化版，按拓扑排序）
fn compute_hierarchical_layout<T, E: Clone>(
    graph: &Graph<T, E>,
    options: &SvgOptions,
) -> HashMap<NodeIndex, (f64, f64)> {
    use crate::algorithms::traversal::topological_sort;

    let mut positions = HashMap::new();
    let nodes_result = topological_sort(graph);

    // 如果有环或错误，回退到圆形布局
    let nodes = match nodes_result {
        Ok(n) => n,
        Err(_) => {
            return compute_circular_layout(
                &graph.nodes().map(|n| n.index()).collect::<Vec<_>>(),
                options,
            )
        }
    };

    if nodes.is_empty() {
        return compute_circular_layout(
            &graph.nodes().map(|n| n.index()).collect::<Vec<_>>(),
            options,
        );
    }

    let n = nodes.len();
    let levels: Vec<Vec<NodeIndex>> = vec![nodes]; // 简化：单层
    let num_levels = levels.len();

    let level_height = options.height as f64 / (num_levels as f64 + 1.0);
    let node_spacing = options.width as f64 / (n as f64 + 1.0);

    for (level_idx, level_nodes) in levels.iter().enumerate() {
        let y = level_height * (level_idx as f64 + 1.0);
        for (node_idx, &node) in level_nodes.iter().enumerate() {
            let x = node_spacing * (node_idx as f64 + 1.0);
            positions.insert(node, (x, y));
        }
    }

    positions
}

/// 绘制箭头
fn draw_arrow(output: &mut String, x1: f64, y1: f64, x2: f64, y2: f64, options: &SvgOptions) {
    let arrow_size = 8.0;
    let angle = (y2 - y1).atan2(x2 - x1);
    let arrow_angle = std::f64::consts::FRAC_PI_4;

    // 计算箭头尖端位置（在节点边缘）
    let dist = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    let stop_dist = dist - options.node_radius;

    if stop_dist < 0.0 {
        return; // 节点重叠，不绘制箭头
    }

    let x1_adj = x1 + (x2 - x1) * (stop_dist / dist);
    let y1_adj = y1 + (y2 - y1) * (stop_dist / dist);

    // 箭头左翼
    let left_angle = angle + arrow_angle;
    let x_left = x1_adj - arrow_size * left_angle.cos();
    let y_left = y1_adj - arrow_size * left_angle.sin();

    // 箭头右翼
    let right_angle = angle - arrow_angle;
    let x_right = x1_adj - arrow_size * right_angle.cos();
    let y_right = y1_adj - arrow_size * right_angle.sin();

    output.push_str(&format!(
        r#"<polygon points="{},{} {},{} {},{}" fill="{}" stroke="none"/>"#,
        x1_adj, y1_adj, x_left, y_left, x_right, y_right, options.edge_color
    ));
    output.push('\n');
}

/// 转义 XML 特殊字符
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// 将 SVG 字符串写入文件
pub fn write_svg_to_file(svg: &str, path: &str) -> std::io::Result<()> {
    std::fs::write(path, svg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::traits::GraphOps;
    use crate::graph::Graph;

    #[test]
    fn test_svg_export_basic() {
        let mut graph: Graph<String, f64> = Graph::directed();
        let a = graph.add_node("A".to_string()).unwrap();
        let b = graph.add_node("B".to_string()).unwrap();
        graph.add_edge(a, b, 1.0).unwrap();

        let svg = to_svg(&graph);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<line"));
    }

    #[test]
    fn test_svg_options() {
        let mut graph: Graph<String, f64> = Graph::directed();
        let a = graph.add_node("A".to_string()).unwrap();
        let b = graph.add_node("B".to_string()).unwrap();
        graph.add_edge(a, b, 1.0).unwrap();

        let options = SvgOptions::new()
            .with_size(400, 300)
            .with_node_radius(15.0)
            .with_labels(false);

        let svg = to_svg_with_options(&graph, &options);
        assert!(svg.contains(r#"width="400""#));
        assert!(svg.contains(r#"height="300""#));
    }

    #[test]
    fn test_svg_empty_graph() {
        let graph: Graph<String, f64> = Graph::directed();
        let svg = to_svg(&graph);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }
}
