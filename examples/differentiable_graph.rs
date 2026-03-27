//! 可微图结构变换示例
//!
//! 本示例演示如何使用 God-Graph 的可微图结构模块进行：
//! 1. 图结构的梯度计算
//! 2. 基于梯度的结构优化
//! 3. Gumbel-Softmax 采样
//! 4. Straight-Through Estimator (STE)

#[cfg(feature = "tensor")]
mod differentiable_graph_example {
    use god_gragh::tensor::differentiable::{
        DifferentiableGraph, GradientConfig, GraphTransformer, GumbelSoftmaxSampler,
        ThresholdEditPolicy,
    };
    use std::collections::HashMap;

    /// 示例 1: 基本的可微图结构
    pub fn basic_differentiable_graph() {
        println!("=== 示例 1: 基本的可微图结构 ===\n");

        // 创建可微图
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(4);

        // 添加可学习边（初始概率表示边的存在可能性）
        graph.add_learnable_edge(0, 1, 0.5);
        graph.add_learnable_edge(0, 2, 0.3);
        graph.add_learnable_edge(1, 2, 0.8);
        graph.add_learnable_edge(2, 3, 0.6);

        println!("初始图结构:");
        println!("  节点数：{}", graph.num_nodes());
        println!("  边数：{}", graph.num_edges());
        println!("  当前温度：{:.4}", graph.temperature());

        // 获取概率矩阵
        let prob_matrix = graph.get_probability_matrix();
        println!("\n边概率矩阵:");
        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.01 {
                    println!("  P({}→{}) = {:.4}", i, j, prob);
                }
            }
        }

        // 离散化（使用 STE）
        graph.discretize();
        println!("\n离散化后的邻接矩阵:");
        let adj_matrix = graph.get_adjacency_matrix();
        for (i, row) in adj_matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val > 0.5 {
                    println!("  {}→{}: 存在", i, j);
                }
            }
        }
    }

    /// 示例 2: 结构梯度计算
    pub fn structure_gradient_computation() {
        println!("\n=== 示例 2: 结构梯度计算 ===\n");

        let mut graph = DifferentiableGraph::<Vec<f64>>::new(4);
        graph.add_learnable_edge(0, 1, 0.5);
        graph.add_learnable_edge(0, 2, 0.3);
        graph.add_learnable_edge(1, 2, 0.8);
        graph.add_learnable_edge(2, 3, 0.6);

        graph.discretize();

        // 模拟从下游任务（如 GNN 分类）传来的梯度
        // 这些梯度表示：如果某条边存在，损失会增加/减少多少
        let mut loss_gradients = HashMap::new();

        // 正梯度：损失随边存在而增加 → 应该删除
        loss_gradients.insert((0, 1), 0.5);

        // 负梯度：损失随边存在而减少 → 应该保留/加强
        loss_gradients.insert((1, 2), -0.8);

        // 小的负梯度
        loss_gradients.insert((0, 2), -0.2);
        loss_gradients.insert((2, 3), -0.3);

        println!("输入的损失梯度:");
        for ((src, dst), &grad) in &loss_gradients {
            let direction = if grad > 0.0 { "删除" } else { "保留" };
            println!("  ∂L/∂A_{}{} = {:.4} → 建议：{}", src, dst, grad, direction);
        }

        // 计算结构梯度（考虑温度、正则化等）
        let structure_gradients = graph.compute_structure_gradients(&loss_gradients);

        println!("\n计算得到的结构梯度 (∂L/∂logits):");
        for ((src, dst), &grad) in &structure_gradients {
            let edge = graph.get_edge_probability(*src, *dst).unwrap();
            println!(
                "  ∂L/∂logits_{}{} = {:.4}, 当前 P({}→{}) = {:.4}",
                src, dst, grad, src, dst, edge
            );
        }

        // 基于梯度更新结构
        graph.update_structure(&structure_gradients);

        println!("\n更新后的边概率:");
        for (src, dst) in [(0, 1), (0, 2), (1, 2), (2, 3)] {
            if let Some(prob) = graph.get_edge_probability(src, dst) {
                println!("  P({}→{}) = {:.4}", src, dst, prob);
            }
        }
    }

    /// 示例 3: 完整的优化循环
    pub fn optimization_loop() {
        println!("\n=== 示例 3: 完整的优化循环 ===\n");

        // 配置优化器
        let config = GradientConfig::new(
            1.0,  // 初始温度
            true, // 使用 STE
            0.05, // 边学习率
            0.01, // 节点学习率
        )
        .with_sparsity(0.001) // L1 稀疏正则化
        .with_smoothness(0.0001); // 平滑正则化

        let mut graph = DifferentiableGraph::<Vec<f64>>::with_config(5, config);

        // 初始化边
        graph.add_learnable_edge(0, 1, 0.5);
        graph.add_learnable_edge(0, 2, 0.5);
        graph.add_learnable_edge(1, 2, 0.5);
        graph.add_learnable_edge(1, 3, 0.5);
        graph.add_learnable_edge(2, 3, 0.5);
        graph.add_learnable_edge(3, 4, 0.5);

        println!("优化前:");
        print_graph_state(&graph);

        // 模拟优化循环
        for step in 0..10 {
            // 模拟损失梯度（实际应用中来自反向传播）
            let mut loss_gradients = HashMap::new();

            // 模拟一个任务：希望形成 0→1→2→3→4 的链式结构
            // 对不在链上的边给正梯度（鼓励删除）
            // 对在链上的边给负梯度（鼓励保留）
            loss_gradients.insert((0, 1), -0.5); // 链上
            loss_gradients.insert((1, 2), -0.5); // 链上
            loss_gradients.insert((2, 3), -0.5); // 链上
            loss_gradients.insert((3, 4), -0.5); // 链上
            loss_gradients.insert((0, 2), 0.3); // 不在链上
            loss_gradients.insert((1, 3), 0.3); // 不在链上

            // 一步优化
            let _gradients = graph.optimization_step(loss_gradients);

            if step % 3 == 0 {
                println!("Step {}: T={:.4}", step, graph.temperature());
                print_graph_state(&graph);
            }
        }

        println!("\n优化后:");
        print_graph_state(&graph);
    }

    /// 示例 4: Gumbel-Softmax 采样
    pub fn gumbel_softmax_sampling() {
        println!("\n=== 示例 4: Gumbel-Softmax 采样 ===\n");

        let mut sampler = GumbelSoftmaxSampler::new(1.0);

        // 边的 logits（可学习参数）
        let edge_logits = vec![0.5, 1.0, -0.5, 2.0];

        println!("输入 logits: {:?}", edge_logits);

        // 软采样（可微，用于训练）
        let soft_sample = sampler.sample_soft(&edge_logits);
        println!(
            "软采样结果: {:?}",
            soft_sample
                .iter()
                .map(|x| format!("{:.4}", x))
                .collect::<Vec<_>>()
        );
        println!("  和：{:.6}", soft_sample.iter().sum::<f64>());

        // 硬采样（不可微，用于推理）
        let hard_sample = sampler.sample_hard(&edge_logits);
        println!(
            "硬采样结果: {:?}",
            hard_sample
                .iter()
                .map(|x| format!("{:.4}", x))
                .collect::<Vec<_>>()
        );

        // STE 采样（前向硬，反向软）
        let (hard_ste, soft_ste) = sampler.sample_ste(&edge_logits);
        println!("STE 采样:");
        println!(
            "  前向（硬）: {:?}",
            hard_ste
                .iter()
                .map(|x| format!("{:.4}", x))
                .collect::<Vec<_>>()
        );
        println!(
            "  反向（软）: {:?}",
            soft_ste
                .iter()
                .map(|x| format!("{:.4}", x))
                .collect::<Vec<_>>()
        );

        // 温度退火演示
        println!("\n温度退火演示:");
        for &temp in &[1.0, 0.5, 0.2, 0.1] {
            sampler.set_temperature(temp);
            let sample = sampler.sample_soft(&edge_logits);
            println!(
                "  T={:.1}: {:?}",
                temp,
                sample
                    .iter()
                    .map(|x| format!("{:.3}", x))
                    .collect::<Vec<_>>()
            );
        }
    }

    /// 示例 5: 使用编辑策略进行结构变换
    pub fn structure_transformation_with_policy() {
        println!("\n=== 示例 5: 使用编辑策略进行结构变换 ===\n");

        // 创建图和策略
        let mut graph = DifferentiableGraph::<Vec<f64>>::new(4);
        graph.add_learnable_edge(0, 1, 0.5);
        graph.add_learnable_edge(0, 2, 0.7);
        graph.add_learnable_edge(1, 2, 0.3);
        graph.add_learnable_edge(2, 3, 0.9);

        // 使用阈值编辑策略
        let policy = Box::new(ThresholdEditPolicy {
            add_threshold: 0.1,
            remove_threshold: -0.1,
            min_prob: 0.01,
            max_prob: 0.99,
        });

        let mut transformer = GraphTransformer::new(policy);

        println!("变换前的图结构:");
        print_graph_state(&graph);

        // 模拟多轮变换
        for round in 0..5 {
            // 模拟梯度（实际来自反向传播）
            let mut gradients = HashMap::new();
            gradients.insert((0, 1), -0.2); // 鼓励
            gradients.insert((0, 2), 0.15); // 抑制
            gradients.insert((1, 2), -0.3); // 鼓励
            gradients.insert((2, 3), -0.1); // 轻微鼓励

            // 记录梯度
            transformer.record_gradients(&gradients);

            // 执行变换
            let edits = transformer.transform(&mut graph);

            println!("\nRound {} 的编辑操作:", round + 1);
            for edit in &edits {
                match &edit.operation {
                    god_gragh::tensor::differentiable::EditOperation::EdgeEdit(src, dst, op) => {
                        let op_str = match op {
                            god_gragh::tensor::differentiable::EdgeEditOp::Add => "添加",
                            god_gragh::tensor::differentiable::EdgeEditOp::Remove => "删除",
                            god_gragh::tensor::differentiable::EdgeEditOp::Modify => "修改",
                        };
                        println!(
                            "  {}→{}: {} (P: {:.3}→{:.3}, ∇: {:.4})",
                            src, dst, op_str, edit.before, edit.after, edit.gradient
                        );
                    }
                    _ => {}
                }
            }
        }

        println!("\n变换后的图结构:");
        print_graph_state(&graph);
    }

    /// 示例 6: 稀疏正则化的效果
    pub fn sparsity_regularization_effect() {
        println!("\n=== 示例 6: 稀疏正则化的效果 ===\n");

        // 无稀疏正则化
        let mut graph_no_sparse = DifferentiableGraph::<Vec<f64>>::with_config(
            4,
            GradientConfig::new(1.0, true, 0.05, 0.01),
        );

        // 有稀疏正则化
        let mut graph_with_sparse = DifferentiableGraph::<Vec<f64>>::with_config(
            4,
            GradientConfig::new(1.0, true, 0.05, 0.01).with_sparsity(0.1), // 较强的稀疏权重
        );

        // 初始化相同的边
        for g in [&mut graph_no_sparse, &mut graph_with_sparse] {
            g.add_learnable_edge(0, 1, 0.5);
            g.add_learnable_edge(0, 2, 0.5);
            g.add_learnable_edge(1, 2, 0.5);
        }

        // 模拟相同的梯度（小的负梯度，鼓励所有边）
        let mut loss_gradients = HashMap::new();
        loss_gradients.insert((0, 1), -0.1);
        loss_gradients.insert((0, 2), -0.1);
        loss_gradients.insert((1, 2), -0.1);

        // 优化 20 步
        for _ in 0..20 {
            graph_no_sparse.optimization_step(loss_gradients.clone());
            graph_with_sparse.optimization_step(loss_gradients.clone());
        }

        println!("无稀疏正则化:");
        for (src, dst) in [(0, 1), (0, 2), (1, 2)] {
            let prob = graph_no_sparse.get_edge_probability(src, dst).unwrap();
            println!("  P({}→{}) = {:.4}", src, dst, prob);
        }
        println!(
            "  平均概率：{:.4}",
            [(0, 1), (0, 2), (1, 2)]
                .iter()
                .map(|(s, d)| graph_no_sparse.get_edge_probability(*s, *d).unwrap())
                .sum::<f64>()
                / 3.0
        );

        println!("\n有稀疏正则化 (λ=0.1):");
        for (src, dst) in [(0, 1), (0, 2), (1, 2)] {
            let prob = graph_with_sparse.get_edge_probability(src, dst).unwrap();
            println!("  P({}→{}) = {:.4}", src, dst, prob);
        }
        println!(
            "  平均概率：{:.4}",
            [(0, 1), (0, 2), (1, 2)]
                .iter()
                .map(|(s, d)| graph_with_sparse.get_edge_probability(*s, *d).unwrap())
                .sum::<f64>()
                / 3.0
        );
    }

    fn print_graph_state<T: Clone + Default>(graph: &DifferentiableGraph<T>) {
        println!(
            "  节点数：{}, 边数：{}, 温度：{:.4}",
            graph.num_nodes(),
            graph.num_edges(),
            graph.temperature()
        );

        let prob_matrix = graph.get_probability_matrix();
        for (i, row) in prob_matrix.iter().enumerate() {
            for (j, &prob) in row.iter().enumerate() {
                if prob > 0.01 {
                    let exists = if prob > 0.5 { "✓" } else { " " };
                    println!("    [{}] {}→{}: P={:.4}", exists, i, j, prob);
                }
            }
        }
    }

    pub fn run_all_examples() {
        basic_differentiable_graph();
        structure_gradient_computation();
        optimization_loop();
        gumbel_softmax_sampling();
        structure_transformation_with_policy();
        sparsity_regularization_effect();
    }
}

fn main() {
    #[cfg(feature = "tensor")]
    differentiable_graph_example::run_all_examples();

    #[cfg(not(feature = "tensor"))]
    println!(
        "请启用 tensor 特性运行此示例：cargo run --example differentiable_graph --features tensor"
    );
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "tensor")]
    #[test]
    fn test_examples_compile() {
        // 确保示例代码可以编译
        use god_gragh::tensor::differentiable::{
            DifferentiableGraph, GradientConfig, GumbelSoftmaxSampler,
        };

        let _graph = DifferentiableGraph::<Vec<f64>>::new(4);
        let _sampler = GumbelSoftmaxSampler::new(1.0);
        let _config = GradientConfig::default();
    }
}
