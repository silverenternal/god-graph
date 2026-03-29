//! Edge IndexMut tests
//!
//! This module tests the Edge IndexMut implementation,
//! including basic modification, generation checking, and concurrent modification.

#[cfg(all(test, feature = "tensor"))]
mod tests {
    use god_gragh::graph::traits::{GraphOps, GraphQuery};
    use god_gragh::graph::Graph;
    use god_gragh::transformer::optimization::switch::{OperatorType, WeightTensor};

    /// Test basic edge modification using IndexMut
    #[test]
    fn test_edge_index_mut_basic() {
        let mut graph = Graph::<i32, f64>::directed();

        // Add nodes and edge
        let a = graph.add_node(1).unwrap();
        let b = graph.add_node(2).unwrap();
        let edge = graph.add_edge(a, b, 42.0).unwrap();

        // Use IndexMut to modify edge data
        graph[edge] = 100.0;

        // Verify the modification
        assert_eq!(graph[edge], 100.0);
        assert_eq!(*graph.get_edge(edge).unwrap(), 100.0);
    }

    /// Test edge modification with generation check (should panic on deleted edge)
    #[test]
    #[should_panic(expected = "边索引已失效")]
    fn test_edge_index_mut_generation_check() {
        let mut graph = Graph::<i32, f64>::directed();

        // Add nodes and edge
        let a = graph.add_node(1).unwrap();
        let b = graph.add_node(2).unwrap();
        let edge = graph.add_edge(a, b, 42.0).unwrap();

        // Remove the edge (increments generation)
        graph.remove_edge(edge).unwrap();

        // Try to add a new edge (reuses the slot with new generation)
        let new_edge = graph.add_edge(a, b, 100.0).unwrap();

        // The old edge index should have a different generation
        assert_ne!(edge.generation(), new_edge.generation());

        // This should panic because the old edge index has an invalid generation
        graph[edge] = 200.0;
    }

    /// Test edge modification after edge removal (should panic on wrong generation)
    #[test]
    #[should_panic(expected = "边索引无效")]
    fn test_edge_index_mut_after_removal() {
        let mut graph = Graph::<i32, f64>::directed();

        // Add nodes and edge
        let a = graph.add_node(1).unwrap();
        let b = graph.add_node(2).unwrap();
        let edge = graph.add_edge(a, b, 42.0).unwrap();

        // Store the edge index
        let stored_edge = edge;

        // Remove the edge (increments generation)
        graph.remove_edge(edge).unwrap();

        // This should panic because the stored edge has an old generation
        graph[stored_edge] = 100.0;
    }

    /// Test concurrent edge modification (multiple edges)
    #[test]
    fn test_edge_index_mut_concurrent() {
        let mut graph = Graph::<i32, f64>::directed();

        // Add nodes
        let nodes: Vec<_> = (0..5).map(|i| graph.add_node(i).unwrap()).collect();

        // Add multiple edges
        let mut edges = Vec::new();
        for i in 0..4 {
            let edge = graph
                .add_edge(nodes[i], nodes[i + 1], (i as f64) * 10.0)
                .unwrap();
            edges.push(edge);
        }

        // Modify all edges using IndexMut
        for (i, &edge) in edges.iter().enumerate() {
            graph[edge] = (i as f64) * 100.0;
        }

        // Verify all modifications
        for (i, &edge) in edges.iter().enumerate() {
            assert_eq!(graph[edge], (i as f64) * 100.0);
        }
    }

    /// Test edge modification with WeightTensor data
    #[test]
    fn test_edge_index_mut_weight_tensor() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut graph = Graph::<OperatorType, WeightTensor>::directed();

        // Add nodes
        let a = graph
            .add_node(OperatorType::Linear {
                in_features: 4,
                out_features: 4,
            })
            .unwrap();
        let b = graph
            .add_node(OperatorType::Linear {
                in_features: 4,
                out_features: 4,
            })
            .unwrap();

        // Add edge with WeightTensor
        let weight_data: Vec<f64> = (0..16).map(|_| rng.gen::<f64>()).collect();
        let original_data = weight_data.clone();
        let weight = WeightTensor::new("test_weight".to_string(), weight_data, vec![4, 4]);
        let edge = graph.add_edge(a, b, weight).unwrap();

        // Modify the weight data using IndexMut
        let weight_mut = &mut graph[edge];
        for i in 0..weight_mut.data.len() {
            weight_mut.data[i] *= 2.0;
        }

        // Verify the modification
        let modified_weight = &graph[edge];
        for i in 0..original_data.len() {
            assert!((modified_weight.data[i] - original_data[i] * 2.0).abs() < 1e-10);
        }
    }

    /// Test edge modification preserves generation
    #[test]
    fn test_edge_index_mut_preserves_generation() {
        let mut graph = Graph::<i32, f64>::directed();

        // Add nodes and edge
        let a = graph.add_node(1).unwrap();
        let b = graph.add_node(2).unwrap();
        let edge = graph.add_edge(a, b, 42.0).unwrap();

        // Store the original generation
        let original_generation = edge.generation();

        // Modify the edge data
        graph[edge] = 100.0;

        // Generation should remain the same
        assert_eq!(edge.generation(), original_generation);
        assert_eq!(graph[edge], 100.0);
    }

    /// Test edge modification with remove and reuse
    #[test]
    fn test_edge_index_mut_after_reuse() {
        let mut graph = Graph::<i32, f64>::directed();

        // Add nodes
        let a = graph.add_node(1).unwrap();
        let b = graph.add_node(2).unwrap();

        // Add and remove edge multiple times
        let edge1 = graph.add_edge(a, b, 10.0).unwrap();
        graph.remove_edge(edge1).unwrap();

        let edge2 = graph.add_edge(a, b, 20.0).unwrap();
        graph.remove_edge(edge2).unwrap();

        let edge3 = graph.add_edge(a, b, 30.0).unwrap();

        // edge3 should have a higher generation than edge1
        assert!(edge3.generation() > edge1.generation());

        // Modifying edge3 should work
        graph[edge3] = 40.0;
        assert_eq!(graph[edge3], 40.0);

        // Trying to access with edge1 should panic (wrong generation)
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut g = graph;
            g[edge1] = 50.0;
        }));
        assert!(result.is_err());
    }
}
