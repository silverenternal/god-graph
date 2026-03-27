//! God-Graph 进程服务器
//!
//! 通过 stdin/stdout 提供 JSON 协议的图操作 API，
//! 使 god-graph 可从任何语言调用，不再局限于 Rust 生态。
//!
//! ## 协议
//!
//! 每个请求是一行 JSON 对象：
//! ```json
//! {"id": 1, "method": "create_graph", "params": {"directed": true}}
//! ```
//!
//! 每个响应是一行 JSON 对象（成功或错误）：
//! ```json
//! {"id": 1, "result": {"graph_id": 0}}
//! {"id": 2, "error": "图不存在：graph_id=99"}
//! ```
//!
//! ## 支持的方法
//!
//! | 方法 | 参数 | 返回 |
//! |------|------|------|
//! | `create_graph` | `directed: bool` | `graph_id: u64` |
//! | `drop_graph` | `graph_id: u64` | `ok: true` |
//! | `list_graphs` | — | `graphs: [u64]` |
//! | `add_node` | `graph_id, data: string` | `node_id: usize` |
//! | `remove_node` | `graph_id, node_id: usize` | `data: string` |
//! | `get_node` | `graph_id, node_id: usize` | `data: string` |
//! | `node_count` | `graph_id` | `count: usize` |
//! | `add_edge` | `graph_id, from, to: usize, weight: f64` | `edge_id: usize` |
//! | `remove_edge` | `graph_id, edge_id: usize` | `weight: f64` |
//! | `edge_count` | `graph_id` | `count: usize` |
//! | `neighbors` | `graph_id, node_id: usize` | `neighbors: [usize]` |
//! | `bfs` | `graph_id, start: usize` | `order: [usize]` |
//! | `dfs` | `graph_id, start: usize` | `order: [usize]` |
//! | `dijkstra` | `graph_id, start: usize` | `distances: {usize: f64}` |
//! | `pagerank` | `graph_id, damping?: f64, iterations?: u64` | `scores: {usize: f64}` |
//! | `connected_components` | `graph_id` | `components: [[usize]]` |
//! | `is_dag` | `graph_id` | `is_dag: bool` |
//! | `topological_sort` | `graph_id` | `order: [usize]` |
//! | `to_dot` | `graph_id` | `dot: string` |

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use god_gragh::algorithms::centrality::pagerank;
use god_gragh::algorithms::community::connected_components;
use god_gragh::algorithms::properties::is_dag;
use god_gragh::algorithms::shortest_path::dijkstra;
use god_gragh::algorithms::traversal::{bfs, dfs, topological_sort};
use god_gragh::export::to_dot;
use god_gragh::graph::traits::{GraphBase, GraphOps, GraphQuery};
use god_gragh::graph::Graph;
use god_gragh::{EdgeIndex, NodeIndex};
use serde_json::{json, Value};

/// 图存储：graph_id → Graph
type GraphStore = HashMap<u64, Graph<String, f64>>;

/// 按原始槽位索引查找有效的 NodeIndex
fn find_node(graph: &Graph<String, f64>, raw: usize) -> Option<NodeIndex> {
    graph
        .nodes()
        .find(|nr| nr.index().index() == raw)
        .map(|nr| nr.index())
}

/// 按原始槽位索引查找有效的 EdgeIndex
fn find_edge(graph: &Graph<String, f64>, raw: usize) -> Option<EdgeIndex> {
    graph
        .edges()
        .find(|er| er.index.index() == raw)
        .map(|er| er.index)
}

/// 处理单个请求，返回结果 Value 或错误字符串
fn handle(store: &mut GraphStore, next_id: &mut u64, method: &str, params: &Value) -> Result<Value, String> {
    /// 从 params 提取 graph_id
    macro_rules! graph_id {
        () => {{
            params
                .get("graph_id")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| "缺少参数 graph_id".to_string())?
        }};
    }

    /// 从 params 提取指定键的 usize
    macro_rules! usize_param {
        ($key:expr) => {{
            params
                .get($key)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("缺少参数 {}", $key))? as usize
        }};
    }

    /// 获取图的不可变引用
    macro_rules! get_graph {
        ($gid:expr) => {
            store
                .get(&$gid)
                .ok_or_else(|| format!("图不存在：graph_id={}", $gid))?
        };
    }

    /// 获取图的可变引用
    macro_rules! get_graph_mut {
        ($gid:expr) => {
            store
                .get_mut(&$gid)
                .ok_or_else(|| format!("图不存在：graph_id={}", $gid))?
        };
    }

    match method {
        // ── 图管理 ─────────────────────────────────────────────────────────────
        "create_graph" => {
            let directed = params
                .get("directed")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let graph = if directed {
                Graph::<String, f64>::directed()
            } else {
                Graph::<String, f64>::undirected()
            };
            let id = *next_id;
            *next_id += 1;
            store.insert(id, graph);
            Ok(json!({ "graph_id": id }))
        }

        "drop_graph" => {
            let gid = graph_id!();
            store
                .remove(&gid)
                .ok_or_else(|| format!("图不存在：graph_id={}", gid))?;
            Ok(json!({ "ok": true }))
        }

        "list_graphs" => {
            let mut ids: Vec<u64> = store.keys().copied().collect();
            ids.sort_unstable();
            Ok(json!({ "graphs": ids }))
        }

        // ── 节点操作 ───────────────────────────────────────────────────────────
        "add_node" => {
            let gid = graph_id!();
            let data = params
                .get("data")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let graph = get_graph_mut!(gid);
            let ni = graph.add_node(data).map_err(|e| e.to_string())?;
            Ok(json!({ "node_id": ni.index() }))
        }

        "remove_node" => {
            let gid = graph_id!();
            let node_id = usize_param!("node_id");
            let graph = get_graph_mut!(gid);
            let ni = find_node(graph, node_id)
                .ok_or_else(|| format!("节点不存在：node_id={}", node_id))?;
            let data = graph.remove_node(ni).map_err(|e| e.to_string())?;
            Ok(json!({ "data": data }))
        }

        "get_node" => {
            let gid = graph_id!();
            let node_id = usize_param!("node_id");
            let graph = get_graph!(gid);
            let ni = find_node(graph, node_id)
                .ok_or_else(|| format!("节点不存在：node_id={}", node_id))?;
            let data = graph.get_node(ni).map_err(|e| e.to_string())?;
            Ok(json!({ "data": data }))
        }

        "node_count" => {
            let gid = graph_id!();
            let graph = get_graph!(gid);
            Ok(json!({ "count": graph.node_count() }))
        }

        // ── 边操作 ─────────────────────────────────────────────────────────────
        "add_edge" => {
            let gid = graph_id!();
            let from = usize_param!("from");
            let to = usize_param!("to");
            let weight = params
                .get("weight")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            let graph = get_graph_mut!(gid);
            let from_ni = find_node(graph, from)
                .ok_or_else(|| format!("节点不存在：node_id={}", from))?;
            let to_ni = find_node(graph, to)
                .ok_or_else(|| format!("节点不存在：node_id={}", to))?;
            let ei = graph
                .add_edge(from_ni, to_ni, weight)
                .map_err(|e| e.to_string())?;
            Ok(json!({ "edge_id": ei.index() }))
        }

        "remove_edge" => {
            let gid = graph_id!();
            let edge_id = usize_param!("edge_id");
            let graph = get_graph_mut!(gid);
            let ei = find_edge(graph, edge_id)
                .ok_or_else(|| format!("边不存在：edge_id={}", edge_id))?;
            let weight = graph.remove_edge(ei).map_err(|e| e.to_string())?;
            Ok(json!({ "weight": weight }))
        }

        "edge_count" => {
            let gid = graph_id!();
            let graph = get_graph!(gid);
            Ok(json!({ "count": graph.edge_count() }))
        }

        // ── 图查询 ─────────────────────────────────────────────────────────────
        "neighbors" => {
            let gid = graph_id!();
            let node_id = usize_param!("node_id");
            let graph = get_graph!(gid);
            let ni = find_node(graph, node_id)
                .ok_or_else(|| format!("节点不存在：node_id={}", node_id))?;
            let neighbors: Vec<usize> = graph.neighbors(ni).map(|n| n.index()).collect();
            Ok(json!({ "neighbors": neighbors }))
        }

        // ── 遍历算法 ───────────────────────────────────────────────────────────
        "bfs" => {
            let gid = graph_id!();
            let start = usize_param!("start");
            let graph = get_graph!(gid);
            let start_ni = find_node(graph, start)
                .ok_or_else(|| format!("节点不存在：node_id={}", start))?;
            let mut order: Vec<usize> = Vec::new();
            bfs(graph, start_ni, |node, _depth| {
                order.push(node.index());
                true
            });
            Ok(json!({ "order": order }))
        }

        "dfs" => {
            let gid = graph_id!();
            let start = usize_param!("start");
            let graph = get_graph!(gid);
            let start_ni = find_node(graph, start)
                .ok_or_else(|| format!("节点不存在：node_id={}", start))?;
            let mut order: Vec<usize> = Vec::new();
            dfs(graph, start_ni, |node| {
                order.push(node.index());
                true
            });
            Ok(json!({ "order": order }))
        }

        "topological_sort" => {
            let gid = graph_id!();
            let graph = get_graph!(gid);
            let sorted = topological_sort(graph).map_err(|e| e.to_string())?;
            let order: Vec<usize> = sorted.iter().map(|ni| ni.index()).collect();
            Ok(json!({ "order": order }))
        }

        // ── 最短路径 ───────────────────────────────────────────────────────────
        "dijkstra" => {
            let gid = graph_id!();
            let start = usize_param!("start");
            let graph = get_graph!(gid);
            let start_ni = find_node(graph, start)
                .ok_or_else(|| format!("节点不存在：node_id={}", start))?;
            let distances =
                dijkstra(graph, start_ni, |_, _, w| *w).map_err(|e| e.to_string())?;
            // 键用字符串（JSON 对象键只能是字符串）
            let map: HashMap<String, f64> = distances
                .into_iter()
                .map(|(ni, dist)| (ni.index().to_string(), dist))
                .collect();
            Ok(json!({ "distances": map }))
        }

        // ── 中心性 ─────────────────────────────────────────────────────────────
        "pagerank" => {
            let gid = graph_id!();
            let damping = params
                .get("damping")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.85);
            let iterations = params
                .get("iterations")
                .and_then(|v| v.as_u64())
                .unwrap_or(20) as usize;
            let graph = get_graph!(gid);
            let scores = pagerank(graph, damping, iterations);
            let map: HashMap<String, f64> = scores
                .into_iter()
                .map(|(ni, score)| (ni.index().to_string(), score))
                .collect();
            Ok(json!({ "scores": map }))
        }

        // ── 连通分量 ───────────────────────────────────────────────────────────
        "connected_components" => {
            let gid = graph_id!();
            let graph = get_graph!(gid);
            let components = connected_components(graph);
            let result: Vec<Vec<usize>> = components
                .into_iter()
                .map(|comp| comp.into_iter().map(|ni| ni.index()).collect())
                .collect();
            Ok(json!({ "components": result }))
        }

        // ── 图属性 ─────────────────────────────────────────────────────────────
        "is_dag" => {
            let gid = graph_id!();
            let graph = get_graph!(gid);
            Ok(json!({ "is_dag": is_dag(graph) }))
        }

        // ── 导出 ───────────────────────────────────────────────────────────────
        "to_dot" => {
            let gid = graph_id!();
            let graph = get_graph!(gid);
            Ok(json!({ "dot": to_dot(graph) }))
        }

        _ => Err(format!("未知方法：{}", method)),
    }
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    let mut store: GraphStore = HashMap::new();
    let mut next_id: u64 = 0;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // 解析 JSON 请求
        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(out, "{}", json!({ "error": format!("JSON 解析失败：{}", e) }));
                let _ = out.flush();
                continue;
            }
        };

        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Object(Default::default()));

        let response = match handle(&mut store, &mut next_id, method, &params) {
            Ok(result) => json!({ "id": id, "result": result }),
            Err(msg) => json!({ "id": id, "error": msg }),
        };

        let _ = writeln!(out, "{}", response);
        let _ = out.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(store: &mut GraphStore, next_id: &mut u64, method: &str, params: Value) -> Value {
        match handle(store, next_id, method, &params) {
            Ok(v) => v,
            Err(e) => json!({ "__error": e }),
        }
    }

    #[test]
    fn test_create_and_drop_graph() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        let r = run(&mut store, &mut next_id, "create_graph", json!({ "directed": true }));
        assert_eq!(r["graph_id"], 0);

        let r2 = run(&mut store, &mut next_id, "create_graph", json!({ "directed": false }));
        assert_eq!(r2["graph_id"], 1);

        let list = run(&mut store, &mut next_id, "list_graphs", json!({}));
        let graphs: Vec<u64> = serde_json::from_value(list["graphs"].clone()).unwrap();
        assert_eq!(graphs, vec![0, 1]);

        let drop = run(&mut store, &mut next_id, "drop_graph", json!({ "graph_id": 0 }));
        assert_eq!(drop["ok"], true);

        let list2 = run(&mut store, &mut next_id, "list_graphs", json!({}));
        let graphs2: Vec<u64> = serde_json::from_value(list2["graphs"].clone()).unwrap();
        assert_eq!(graphs2, vec![1]);
    }

    #[test]
    fn test_node_operations() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        run(&mut store, &mut next_id, "create_graph", json!({ "directed": true }));

        let r = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "A" }));
        let a_id = r["node_id"].as_u64().unwrap() as usize;

        let r2 = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "B" }));
        let b_id = r2["node_id"].as_u64().unwrap() as usize;

        let count = run(&mut store, &mut next_id, "node_count", json!({ "graph_id": 0 }));
        assert_eq!(count["count"], 2);

        let get = run(&mut store, &mut next_id, "get_node", json!({ "graph_id": 0, "node_id": a_id }));
        assert_eq!(get["data"], "A");

        let del = run(&mut store, &mut next_id, "remove_node", json!({ "graph_id": 0, "node_id": b_id }));
        assert_eq!(del["data"], "B");

        let count2 = run(&mut store, &mut next_id, "node_count", json!({ "graph_id": 0 }));
        assert_eq!(count2["count"], 1);
    }

    #[test]
    fn test_edge_operations() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        run(&mut store, &mut next_id, "create_graph", json!({ "directed": true }));
        let a = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "A" }))["node_id"].as_u64().unwrap() as usize;
        let b = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "B" }))["node_id"].as_u64().unwrap() as usize;

        let e = run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": a, "to": b, "weight": 2.5 }));
        let edge_id = e["edge_id"].as_u64().unwrap() as usize;

        let ec = run(&mut store, &mut next_id, "edge_count", json!({ "graph_id": 0 }));
        assert_eq!(ec["count"], 1);

        let nb = run(&mut store, &mut next_id, "neighbors", json!({ "graph_id": 0, "node_id": a }));
        let neighbors: Vec<usize> = serde_json::from_value(nb["neighbors"].clone()).unwrap();
        assert!(neighbors.contains(&b));

        let del = run(&mut store, &mut next_id, "remove_edge", json!({ "graph_id": 0, "edge_id": edge_id }));
        assert!((del["weight"].as_f64().unwrap() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_bfs_dfs() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        run(&mut store, &mut next_id, "create_graph", json!({ "directed": true }));
        let a = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "A" }))["node_id"].as_u64().unwrap() as usize;
        let b = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "B" }))["node_id"].as_u64().unwrap() as usize;
        let c = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "C" }))["node_id"].as_u64().unwrap() as usize;
        run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": a, "to": b, "weight": 1.0 }));
        run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": b, "to": c, "weight": 1.0 }));

        let bfs_r = run(&mut store, &mut next_id, "bfs", json!({ "graph_id": 0, "start": a }));
        let bfs_order: Vec<usize> = serde_json::from_value(bfs_r["order"].clone()).unwrap();
        assert_eq!(bfs_order.len(), 3);

        let dfs_r = run(&mut store, &mut next_id, "dfs", json!({ "graph_id": 0, "start": a }));
        let dfs_order: Vec<usize> = serde_json::from_value(dfs_r["order"].clone()).unwrap();
        assert_eq!(dfs_order.len(), 3);
    }

    #[test]
    fn test_dijkstra() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        run(&mut store, &mut next_id, "create_graph", json!({ "directed": true }));
        let a = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "A" }))["node_id"].as_u64().unwrap() as usize;
        let b = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "B" }))["node_id"].as_u64().unwrap() as usize;
        let c = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "C" }))["node_id"].as_u64().unwrap() as usize;
        run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": a, "to": b, "weight": 1.0 }));
        run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": b, "to": c, "weight": 2.0 }));
        run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": a, "to": c, "weight": 5.0 }));

        let r = run(&mut store, &mut next_id, "dijkstra", json!({ "graph_id": 0, "start": a }));
        let distances: HashMap<String, f64> = serde_json::from_value(r["distances"].clone()).unwrap();
        assert!((distances[&c.to_string()] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_pagerank() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        run(&mut store, &mut next_id, "create_graph", json!({ "directed": true }));
        let a = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "A" }))["node_id"].as_u64().unwrap() as usize;
        let b = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "B" }))["node_id"].as_u64().unwrap() as usize;
        run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": a, "to": b, "weight": 1.0 }));
        run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": b, "to": a, "weight": 1.0 }));

        let r = run(&mut store, &mut next_id, "pagerank", json!({ "graph_id": 0 }));
        let scores: HashMap<String, f64> = serde_json::from_value(r["scores"].clone()).unwrap();
        assert!(scores.contains_key(&a.to_string()));
        assert!(scores.contains_key(&b.to_string()));
    }

    #[test]
    fn test_connected_components() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        run(&mut store, &mut next_id, "create_graph", json!({ "directed": false }));
        let a = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "A" }))["node_id"].as_u64().unwrap() as usize;
        let b = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "B" }))["node_id"].as_u64().unwrap() as usize;
        let c = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "C" }))["node_id"].as_u64().unwrap() as usize;
        run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": a, "to": b, "weight": 1.0 }));
        // c is isolated

        let r = run(&mut store, &mut next_id, "connected_components", json!({ "graph_id": 0 }));
        let components: Vec<Vec<usize>> = serde_json::from_value(r["components"].clone()).unwrap();
        assert_eq!(components.len(), 2);
        let _ = c; // used in graph
    }

    #[test]
    fn test_is_dag_and_topological_sort() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        run(&mut store, &mut next_id, "create_graph", json!({ "directed": true }));
        let a = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "A" }))["node_id"].as_u64().unwrap() as usize;
        let b = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "B" }))["node_id"].as_u64().unwrap() as usize;
        let c = run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "C" }))["node_id"].as_u64().unwrap() as usize;
        run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": a, "to": b, "weight": 1.0 }));
        run(&mut store, &mut next_id, "add_edge", json!({ "graph_id": 0, "from": b, "to": c, "weight": 1.0 }));

        let dag = run(&mut store, &mut next_id, "is_dag", json!({ "graph_id": 0 }));
        assert_eq!(dag["is_dag"], true);

        let topo = run(&mut store, &mut next_id, "topological_sort", json!({ "graph_id": 0 }));
        let order: Vec<usize> = serde_json::from_value(topo["order"].clone()).unwrap();
        assert_eq!(order.len(), 3);
        // a must come before b, b before c
        let pos_a = order.iter().position(|&x| x == a).unwrap();
        let pos_b = order.iter().position(|&x| x == b).unwrap();
        let pos_c = order.iter().position(|&x| x == c).unwrap();
        assert!(pos_a < pos_b && pos_b < pos_c);
    }

    #[test]
    fn test_to_dot() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        run(&mut store, &mut next_id, "create_graph", json!({ "directed": true }));
        run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "A" }));
        run(&mut store, &mut next_id, "add_node", json!({ "graph_id": 0, "data": "B" }));

        let r = run(&mut store, &mut next_id, "to_dot", json!({ "graph_id": 0 }));
        let dot = r["dot"].as_str().unwrap();
        assert!(dot.contains("digraph") || dot.contains("graph"));
    }

    #[test]
    fn test_error_unknown_graph() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        let r = run(&mut store, &mut next_id, "node_count", json!({ "graph_id": 99 }));
        assert!(r.get("__error").is_some());
    }

    #[test]
    fn test_error_unknown_method() {
        let mut store = GraphStore::new();
        let mut next_id = 0u64;

        let r = run(&mut store, &mut next_id, "no_such_method", json!({}));
        assert!(r.get("__error").is_some());
    }
}
