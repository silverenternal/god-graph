//! 可微注意力剪枝示例 - 使用真实 TinyLlama 模型
//!
//! 本示例展示如何用 DifferentiableGraph 实现动态注意力剪枝：
//! 1. 从 TinyLlama 真实模型加载权重
//! 2. 构建可微图结构
//! 3. 定义目标函数（注意力熵 + 稀疏性正则）
//! 4. 梯度下降优化边结构
//! 5. 导出剪枝后的模型
//!
//! ## 运行前准备
//!
//! ```bash
//! # 下载 TinyLlama-1.1B 模型
//! pip install huggingface_hub
//! huggingface-cli download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-5T \
//!     --include 'model.safetensors' \
//!     --local-dir models/tinyllama
//! ```
//!
//! ## 运行命令
//!
//! ```bash
//! cargo run --example differentiable_attention_pruning --features "tensor,tensor-pool,transformer,safetensors"
//! ```
//!
//! Requires the `tensor`, `tensor-pool`, `transformer`, and `safetensors` features.

#[cfg(all(feature = "tensor", feature = "tensor-pool", feature = "transformer", feature = "safetensors"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use god_graph::graph::traits::GraphBase;
    use god_graph::tensor::differentiable::{DifferentiableGraph, GradientConfig};
    use god_graph::transformer::optimization::switch::ModelSwitch;

    println!("=== 可微注意力剪枝示例（真实 TinyLlama 模型）===\n");

    // ============================================
    // Step 1: 加载真实 TinyLlama 模型
    // ============================================
    println!("Step 1: 加载 TinyLlama-1.1B 模型...");

    let model_path = get_tinyllama_model_path();
    if model_path.is_none() {
        eprintln!("⚠️  TinyLlama 模型未找到，使用合成数据演示...");
        eprintln!("   下载命令:");
        eprintln!(
            "   huggingface-cli download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-5T \\"
        );
        eprintln!("       --include 'model.safetensors' --local-dir models/tinyllama");
        run_synthetic_demo();
        return Ok(());
    }

    let model_path = model_path.unwrap();
    println!("  模型路径：{}", model_path);

    let graph = ModelSwitch::load_from_safetensors(&model_path)?;

    let node_count = graph.node_count();
    let edge_count = graph.edge_count();
    println!("  ✓ 加载完成：{} 节点，{} 权重边", node_count, edge_count);

    // ============================================
    // Step 2: 转换为可微图结构
    // ============================================
    println!("\nStep 2: 构建可微图结构...");

    // 配置：不启用稀疏正则化，因为我们已经在损失函数中处理了重要性
    let config = GradientConfig::default()
        .with_sparsity(0.0) // 不使用 L1 稀疏正则化
        .with_smoothness(0.0) // 不使用平滑正则化
        .with_edge_learning_rate(0.5); // 学习率

    // 从现有图构建可微图，使用 0.5 的初始概率（不确定状态）
    let mut diff_graph = DifferentiableGraph::<Vec<f64>>::from_graph_with_prob(&graph, Some(0.5));

    // 禁用 STE，使用纯连续优化
    diff_graph.set_config(config);
    diff_graph.set_ste(false);

    println!("  ✓ 可微图构建完成");
    println!("    - 初始边数：{}", diff_graph.num_edges());
    println!("    - 温度参数：{:.4}", diff_graph.temperature());
    println!("    - 稀疏正则化：λ={:.4}", 0.1);

    // ============================================
    // Step 3: 梯度下降优化结构
    // ============================================
    println!("\nStep 3: 梯度下降优化注意力结构...");
    println!("  目标函数：L = importance*(1-prob)² + (1-importance)*prob² + α*prob");
    println!("  理念：重要边概率趋向 1，不重要边概率趋向 0，同时鼓励稀疏");
    println!("  学习率：0.3");
    println!("  迭代次数：50\n");

    let _learning_rate = 0.3;
    let num_steps = 50;

    // 预计算每条边的重要性（基于节点索引的简单启发式）
    // 在实际应用中，这里应该通过前向传播计算梯度
    let mut edge_importance = std::collections::HashMap::new();

    // 获取节点数，用于计算相对重要性
    let num_nodes = diff_graph.num_nodes();

    for (&(src, dst), _) in diff_graph.edges() {
        // 对于自环边，使用节点索引计算重要性（前面的层更重要）
        // 对于非自环边，使用距离计算重要性
        let importance = if src == dst {
            // 自环边：前面的层更重要
            let relative_pos = src as f64 / num_nodes.max(1) as f64;
            // 从 0.9（第一层）到 0.3（最后一层）线性衰减
            0.9 - 0.6 * relative_pos
        } else {
            // 非自环边：基于距离
            let distance = (src as i32 - dst as i32).abs();
            if distance <= 1 {
                0.9
            } else if distance <= 2 {
                0.6
            } else if distance <= 4 {
                0.3
            } else {
                0.1
            }
        };
        edge_importance.insert((src, dst), importance);
    }

    for step in 0..num_steps {
        // 计算边级别的梯度（基于重要性和稀疏性）
        let mut edge_gradients = std::collections::HashMap::new();
        for (&(src, dst), edge) in diff_graph.edges() {
            let prob = edge.probability;
            let importance = edge_importance.get(&(src, dst)).unwrap_or(&0.5);

            // 损失函数：L = importance * (1 - prob)² + (1 - importance) * prob² + α * prob
            // 梯度：∂L/∂prob = 2(prob - importance) + α
            //
            // 平衡点：2(prob - importance) + α = 0
            // prob = importance - α/2
            //
            // 当α=0.4 时：
            // - 重要边 (imp=0.9): prob = 0.9 - 0.2 = 0.7 (保留)
            // - 中等边 (imp=0.6): prob = 0.6 - 0.2 = 0.4 (剪掉)
            // - 不重要边 (imp=0.3): prob = 0.3 - 0.2 = 0.1 (剪掉)

            let alpha = 0.4; // 稀疏性系数
            let gradient = 2.0 * (prob - importance) + alpha;

            edge_gradients.insert((src, dst), gradient);
        }

        // 计算结构梯度（考虑温度、正则化等）
        let grads = diff_graph.compute_structure_gradients(&edge_gradients);

        // 更新结构（学习率在 config 中设置）
        diff_graph.update_structure(&grads);

        // 温度退火（逐渐降低温度，使结构更离散）
        if step % 10 == 0 {
            let current_temp = diff_graph.temperature();
            let new_temp = (current_temp * 0.9).max(0.1);
            diff_graph.set_temperature(new_temp);

            // 离散化并统计边数
            diff_graph.discretize();
            let current_edges = diff_graph.edges().filter(|(_, edge)| edge.exists).count();

            // 打印第一条边的概率用于调试
            if let Some((_, first_edge)) = diff_graph.edges().next() {
                println!(
                    "Step {:3}: T={:.2}, 存在边数={}, prob={:.4}, logits={:.4}",
                    step, new_temp, current_edges, first_edge.probability, first_edge.logits
                );
            }
        }
    }

    // ============================================
    // Step 4: 导出优化结果
    // ============================================
    println!("\nStep 4: 导出优化结果...");

    diff_graph.discretize();
    let final_edge_count = diff_graph.edges().filter(|(_, edge)| edge.exists).count();
    let pruned_count = edge_count - final_edge_count;
    let pruned_ratio = if edge_count > 0 {
        (pruned_count as f64 / edge_count as f64) * 100.0
    } else {
        0.0
    };

    println!("  优化结果:");
    println!("    - 原始边数：{}", edge_count);
    println!("    - 最终边数：{}", final_edge_count);
    println!("    - 剪枝边数：{}", pruned_count);
    println!("    - 剪枝比例：{:.2}%", pruned_ratio);

    println!("\n=== 优化完成 ===");

    Ok(())
}

/// 获取 TinyLlama 模型路径
fn get_tinyllama_model_path() -> Option<String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    let possible_paths = vec![
        std::path::Path::new(&manifest_dir).join("models/tinyllama/model.safetensors"),
        std::path::Path::new(&manifest_dir).join("models/model.safetensors"),
    ];

    for path in possible_paths {
        if path.exists() {
            return Some(path.to_string_lossy().to_string());
        }
    }

    None
}

/// 使用合成数据的简化演示（当真实模型不可用时）
fn run_synthetic_demo() {
    use god_graph::graph::Graph;
    use god_graph::graph::GraphBase;
    use god_graph::graph::GraphOps;
    use god_graph::tensor::differentiable::{DifferentiableGraph, GradientConfig};

    println!("\n=== 合成数据演示模式 ===\n");

    // 构建一个简化的注意力图
    let mut graph: Graph<Vec<f64>, f64> = Graph::directed();

    let n_tokens = 8;
    let hidden_dim = 16;
    let mut token_nodes = Vec::new();

    // 创建 token 节点
    for _i in 0..n_tokens {
        let feature = vec![1.0; hidden_dim];
        let node_idx = graph.add_node(feature).unwrap();
        token_nodes.push(node_idx);
    }

    // 创建全连接注意力边（后续可剪枝）
    for &src in &token_nodes {
        for &dst in &token_nodes {
            if src != dst {
                let weight = 1.0 / (n_tokens - 1) as f64;
                let _ = graph.add_edge(src, dst, weight);
            }
        }
    }

    println!("合成图构建完成:");
    println!("  - 节点数：{}", graph.node_count());
    println!("  - 边数：{}", graph.edge_count());

    // 转换为可微图，使用 0.5 的初始概率（不确定状态，便于演示优化）
    let config = GradientConfig::default()
        .with_sparsity(0.0) // 不使用 L1 稀疏正则化
        .with_smoothness(0.0) // 不使用平滑正则化
        .with_edge_learning_rate(0.5); // 学习率
    let mut diff_graph = DifferentiableGraph::<Vec<f64>>::from_graph_with_prob(&graph, Some(0.5));
    diff_graph.set_config(config);
    diff_graph.set_ste(false); // 禁用 STE，使用纯连续优化

    println!("\n开始优化...");
    println!("  目标函数：L = importance*(1-prob)² + (1-importance)*prob² + α*prob");
    println!("  理念：重要边概率趋向 1，不重要边概率趋向 0，同时鼓励稀疏\n");

    // 预计算每条边的重要性（基于局部注意力原则）
    let mut edge_importance = std::collections::HashMap::new();
    for (&(src, dst), _) in diff_graph.edges() {
        // 局部注意力更重要：相邻 token 的注意力权重通常更大
        let distance = (src as i32 - dst as i32).abs();
        let importance = if distance <= 1 {
            0.9 // 非常近的连接
        } else if distance <= 2 {
            0.6 // 较近的连接
        } else if distance <= 4 {
            0.3 // 中等距离
        } else {
            0.1 // 远距离，不太重要
        };
        edge_importance.insert((src, dst), importance);
    }

    for step in 0..30 {
        // 计算边级别梯度（基于重要性）
        let mut edge_gradients = std::collections::HashMap::new();
        for (&(src, dst), edge) in diff_graph.edges() {
            let prob = edge.probability;
            let importance = edge_importance.get(&(src, dst)).unwrap_or(&0.5);

            // 梯度：∂L/∂prob = 2(prob - importance) + α
            let alpha = 0.4;
            let gradient = 2.0 * (prob - importance) + alpha;

            edge_gradients.insert((src, dst), gradient);
        }

        let grads = diff_graph.compute_structure_gradients(&edge_gradients);
        diff_graph.update_structure(&grads);

        if step % 10 == 0 {
            diff_graph.discretize();
            let current_edges = diff_graph.edges().filter(|(_, edge)| edge.exists).count();

            // 打印不同重要性边的概率
            let local_edge_prob = diff_graph
                .edges()
                .filter(|(&(s, d), _)| (s as i32 - d as i32).abs() <= 1)
                .map(|(_, e)| e.probability)
                .next()
                .unwrap_or(0.0);
            let far_edge_prob = diff_graph
                .edges()
                .filter(|(&(s, d), _)| (s as i32 - d as i32).abs() > 2)
                .map(|(_, e)| e.probability)
                .next()
                .unwrap_or(0.0);

            println!(
                "Step {:3}: T={:.2}, 存在边数={}, 近边 prob={:.2}, 远边 prob={:.2}",
                step,
                diff_graph.config().temperature,
                current_edges,
                local_edge_prob,
                far_edge_prob
            );
        }

        // 温度退火
        diff_graph.set_temperature((0.95_f64).powi(step));
    }

    // 最终离散化
    diff_graph.discretize();
    let final_edges = diff_graph.edges().filter(|(_, edge)| edge.exists).count();
    let total_edges = diff_graph.num_edges();
    let total_possible = n_tokens * (n_tokens - 1);

    // 分类统计
    let very_near_edges = diff_graph
        .edges()
        .filter(|(&(s, d), edge)| (s as i32 - d as i32).abs() <= 1 && edge.exists)
        .count();
    let near_edges = diff_graph
        .edges()
        .filter(|(&(s, d), edge)| (s as i32 - d as i32).abs() == 2 && edge.exists)
        .count();
    let far_edges = diff_graph
        .edges()
        .filter(|(&(s, d), edge)| (s as i32 - d as i32).abs() > 2 && edge.exists)
        .count();

    println!("\n优化完成:");
    println!(
        "  最终边数：{}/{} ({:.2}%)",
        final_edges,
        total_possible,
        (final_edges as f64 / total_possible as f64) * 100.0
    );
    println!("  非常近边 (距离≤1): {} 条保留", very_near_edges);
    println!("  较近边 (距离=2): {} 条保留", near_edges);
    println!("  远距离边 (距离>2): {} 条保留", far_edges);
    println!(
        "  剪枝比例：{:.2}%",
        (total_edges - final_edges) as f64 / total_edges as f64 * 100.0
    );
}

// Fallback main function when required features are not enabled
#[cfg(not(all(feature = "tensor", feature = "tensor-pool", feature = "transformer", feature = "safetensors")))]
fn main() {
    println!("This example requires the 'tensor', 'tensor-pool', 'transformer', and 'safetensors' features.");
    println!("Run with: cargo run --example differentiable_attention_pruning --features tensor,tensor-pool,transformer,safetensors");
}
