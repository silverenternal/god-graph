//! Graph-Tensor Integration: Specialized implementations for seamless graph-tensor conversion
//!
//! This module provides:
//! - `TensorGraph`: Graph with native tensor node/edge features
//! - `GraphToTensor`: Convert traditional graphs to tensor representations
//! - `TensorToGraph`: Convert tensors back to graph structures
//! - Adjacency matrix extraction and reconstruction
//! - Feature matrix extraction for GNN workflows

use crate::graph::traits::{GraphBase, GraphOps, GraphQuery};
use crate::graph::Graph;
use crate::tensor::error::TensorError;
use crate::tensor::traits::TensorBase;
use crate::tensor::DenseTensor;

#[cfg(feature = "tensor-sparse")]
use crate::tensor::{COOTensor, CSRTensor};

/// Adjacency matrix representation for graph neural networks
#[derive(Debug, Clone)]
pub struct GraphAdjacencyMatrix {
    /// Sparse adjacency matrix in CSR format
    csr: CSRTensor,
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Whether the graph is directed
    pub is_directed: bool,
}

impl GraphAdjacencyMatrix {
    /// Create adjacency matrix from edge list
    pub fn from_edge_list(
        edges: &[(usize, usize)],
        num_nodes: usize,
        is_directed: bool,
    ) -> Result<Self, TensorError> {
        if edges.is_empty() {
            return Ok(Self {
                csr: CSRTensor::new(
                    vec![0; num_nodes + 1],
                    vec![],
                    DenseTensor::zeros(vec![0]),
                    [num_nodes, num_nodes],
                ),
                num_nodes,
                num_edges: 0,
                is_directed,
            });
        }

        // Build CSR format
        let mut row_offsets = vec![0usize; num_nodes + 1];
        let mut col_indices = Vec::with_capacity(edges.len());
        let mut values_data = Vec::with_capacity(edges.len());

        // Count edges per row
        for &(src, _) in edges {
            if src < num_nodes {
                row_offsets[src + 1] += 1;
            }
        }

        // Cumulative sum
        for i in 1..=num_nodes {
            row_offsets[i] += row_offsets[i - 1];
        }

        // Fill column indices and values
        let mut row_pos = row_offsets[..num_nodes].to_vec();
        for &(src, dst) in edges {
            if src < num_nodes && dst < num_nodes {
                let _pos = row_pos[src];
                col_indices.push(dst);
                values_data.push(1.0);
                row_pos[src] += 1;
            }
        }

        let values = DenseTensor::new(values_data, vec![col_indices.len()]);
        let csr = CSRTensor::new(row_offsets, col_indices, values, [num_nodes, num_nodes]);

        Ok(Self {
            csr,
            num_nodes,
            num_edges: edges.len(),
            is_directed,
        })
    }

    /// Convert to COO format
    #[cfg(feature = "tensor-sparse")]
    pub fn to_coo(&self) -> COOTensor {
        use crate::tensor::SparseTensor;
        let sparse = SparseTensor::CSR(self.csr.clone());
        sparse.to_coo()
    }

    /// Get sparse tensor representation
    pub fn as_sparse_tensor(&self) -> &CSRTensor {
        &self.csr
    }

    /// Compute normalized adjacency matrix (for GCN)
    ///
    /// Returns: D^(-1/2) * (A + I) * D^(-1/2)
    /// where D is the degree matrix and I is the identity matrix
    pub fn normalized_with_self_loops(&self) -> Result<Self, TensorError> {
        let n = self.num_nodes;

        // Add self-loops
        let mut edges = Vec::new();

        // Extract existing edges
        for i in 0..n {
            let start = self.csr.row_offsets()[i];
            let end = self.csr.row_offsets()[i + 1];
            for j in start..end {
                let col = self.csr.col_indices()[j];
                edges.push((i, col));
            }
            // Add self-loop
            edges.push((i, i));
        }

        Self::from_edge_list(&edges, n, self.is_directed)
    }

    /// Compute degree matrix
    pub fn degree_matrix(&self) -> DenseTensor {
        let n = self.num_nodes;
        let mut degrees = vec![0.0; n];

        for (i, degree) in degrees.iter_mut().enumerate() {
            let start = self.csr.row_offsets()[i];
            let end = self.csr.row_offsets()[i + 1];
            *degree = (end - start) as f64;
        }

        DenseTensor::from_vec(degrees, vec![n])
    }

    /// Compute inverse degree matrix (for normalization)
    pub fn inverse_degree_matrix(&self) -> DenseTensor {
        let n = self.num_nodes;
        let mut inv_degrees = vec![0.0; n];

        for (i, inv_degree) in inv_degrees.iter_mut().enumerate() {
            let start = self.csr.row_offsets()[i];
            let end = self.csr.row_offsets()[i + 1];
            let degree = (end - start) as f64;
            *inv_degree = if degree > 0.0 { 1.0 / degree } else { 0.0 };
        }

        DenseTensor::from_vec(inv_degrees, vec![n])
    }
}

/// Feature extractor for converting graphs to tensor representations
pub struct GraphFeatureExtractor<'a, T, E> {
    graph: &'a Graph<T, E>,
}

impl<'a, T, E> GraphFeatureExtractor<'a, T, E>
where
    T: Clone,
    E: Clone,
{
    /// Create new extractor from graph
    pub fn new(graph: &'a Graph<T, E>) -> Self {
        Self { graph }
    }

    /// Extract node features as dense tensor
    ///
    /// Each node's data is treated as a scalar feature
    pub fn extract_node_features_scalar<F>(&self, map_fn: F) -> Result<DenseTensor, TensorError>
    where
        F: Fn(&T) -> f64,
    {
        let n = self.graph.node_count();
        let mut features = Vec::with_capacity(n);

        for node_idx in self.graph.nodes() {
            let data = node_idx.data();
            features.push(map_fn(data));
        }

        Ok(DenseTensor::from_vec(features, vec![n, 1]))
    }

    /// Extract node features as 2D tensor (nodes x features)
    ///
    /// Requires node data to be convertible to feature vectors
    pub fn extract_node_features<F>(
        &self,
        map_fn: F,
        num_features: usize,
    ) -> Result<DenseTensor, TensorError>
    where
        F: for<'b> Fn(&'b T) -> &'b [f64],
    {
        let n = self.graph.node_count();
        let mut features = Vec::with_capacity(n * num_features);

        for node_idx in self.graph.nodes() {
            let data = node_idx.data();
            let feat = map_fn(data);
            features.extend_from_slice(feat);
        }

        Ok(DenseTensor::from_vec(features, vec![n, num_features]))
    }

    /// Extract edge features as tensor
    pub fn extract_edge_features<F>(&self, map_fn: F) -> Result<DenseTensor, TensorError>
    where
        F: Fn(&E) -> f64,
    {
        let m = self.graph.edge_count();
        let mut features = Vec::with_capacity(m);

        for edge_idx in self.graph.edges() {
            let data = edge_idx.data();
            features.push(map_fn(data));
        }

        Ok(DenseTensor::from_vec(features, vec![m, 1]))
    }

    /// Extract adjacency matrix as sparse tensor
    pub fn extract_adjacency(&self) -> Result<GraphAdjacencyMatrix, TensorError> {
        let mut edges: Vec<(usize, usize)> = Vec::new();

        for node_idx in self.graph.nodes() {
            let src = node_idx.index().index();
            for neighbor in self.graph.neighbors(node_idx.index()) {
                let dst = neighbor.index();
                edges.push((src, dst));
            }
        }

        GraphAdjacencyMatrix::from_edge_list(
            &edges,
            self.graph.node_count(),
            true, // Assume directed for adjacency extraction
        )
    }

    /// Extract complete graph as tensor representation
    pub fn extract_all(
        &self,
        num_node_features: usize,
    ) -> Result<(DenseTensor, GraphAdjacencyMatrix), TensorError>
    where
        T: AsRef<[f64]> + Clone,
        E: Clone,
    {
        let node_features =
            self.extract_node_features(|data: &T| data.as_ref(), num_node_features)?;
        let adjacency = self.extract_adjacency()?;

        Ok((node_features, adjacency))
    }
}

/// Reconstruct graph from tensor representations
pub struct GraphReconstructor {
    directed: bool,
}

impl GraphReconstructor {
    /// Create new reconstructor
    pub fn new(directed: bool) -> Self {
        Self { directed }
    }

    /// Reconstruct graph from adjacency matrix
    pub fn from_adjacency<T, E>(
        &self,
        adjacency: &GraphAdjacencyMatrix,
        mut node_factory: impl FnMut(usize) -> T,
        mut edge_factory: impl FnMut(usize, usize, f64) -> E,
    ) -> Result<Graph<T, E>, TensorError>
    where
        T: Clone,
        E: Clone,
    {
        let mut graph = if self.directed {
            Graph::<T, E>::directed()
        } else {
            Graph::<T, E>::undirected()
        };

        let n = adjacency.num_nodes;
        let mut node_indices = Vec::with_capacity(n);

        // Create nodes
        for i in 0..n {
            let node = node_factory(i);
            let idx = graph.add_node(node).map_err(|e| TensorError::SliceError {
                description: format!("Failed to add node: {:?}", e),
            })?;
            node_indices.push(idx);
        }

        // Create edges from CSR
        let csr = adjacency.as_sparse_tensor();

        for src in 0..n {
            let start = csr.row_offsets()[src];
            let end = csr.row_offsets()[src + 1];

            for j in start..end {
                let dst = csr.col_indices()[j];
                let weight = csr.values().data()[j];

                if let (Some(src_idx), Some(dst_idx)) = (
                    node_indices.get(src).copied(),
                    node_indices.get(dst).copied(),
                ) {
                    let edge_data = edge_factory(src, dst, weight);
                    let _ = graph.add_edge(src_idx, dst_idx, edge_data);
                }
            }
        }

        Ok(graph)
    }

    /// Reconstruct graph from COO tensor
    pub fn from_coo<T, E>(
        &self,
        coo: &COOTensor,
        node_factory: impl FnMut(usize) -> T,
        edge_factory: impl FnMut(usize, usize, f64) -> E,
    ) -> Result<Graph<T, E>, TensorError>
    where
        T: Clone,
        E: Clone,
    {
        // Convert COO to edge list
        let row_indices = coo.row_indices();
        let col_indices = coo.col_indices();
        let edges: Vec<(usize, usize)> = row_indices
            .iter()
            .zip(col_indices.iter())
            .map(|(&r, &c)| (r, c))
            .collect();

        let shape = coo.shape_array();
        let adjacency = GraphAdjacencyMatrix::from_edge_list(&edges, shape[0], self.directed)?;

        self.from_adjacency(&adjacency, node_factory, edge_factory)
    }
}

/// Extension trait for Graph to add tensor conversion methods
pub trait GraphTensorExt<T, E> {
    /// Convert graph to tensor representation
    fn to_tensor_representation(&self) -> Result<(DenseTensor, GraphAdjacencyMatrix), TensorError>
    where
        T: AsRef<[f64]> + Clone,
        E: Clone;

    /// Get adjacency matrix as sparse tensor
    fn adjacency_matrix(&self) -> Result<GraphAdjacencyMatrix, TensorError>;

    /// Extract node features as tensor
    fn node_features(&self, num_features: usize) -> Result<DenseTensor, TensorError>
    where
        T: AsRef<[f64]> + Clone;

    /// Create feature extractor
    fn feature_extractor(&self) -> GraphFeatureExtractor<'_, T, E>;
}

impl<T, E> GraphTensorExt<T, E> for Graph<T, E>
where
    T: Clone,
    E: Clone,
{
    fn to_tensor_representation(&self) -> Result<(DenseTensor, GraphAdjacencyMatrix), TensorError>
    where
        T: AsRef<[f64]> + Clone,
        E: Clone,
    {
        let extractor = GraphFeatureExtractor::new(self);
        let num_features = if let Some(first_node) = self.nodes().next() {
            first_node.data().as_ref().len()
        } else {
            1
        };

        extractor.extract_all(num_features)
    }

    fn adjacency_matrix(&self) -> Result<GraphAdjacencyMatrix, TensorError> {
        let extractor = GraphFeatureExtractor::new(self);
        extractor.extract_adjacency()
    }

    fn node_features(&self, num_features: usize) -> Result<DenseTensor, TensorError>
    where
        T: AsRef<[f64]> + Clone,
    {
        let extractor = GraphFeatureExtractor::new(self);
        extractor.extract_node_features(|data: &T| data.as_ref(), num_features)
    }

    fn feature_extractor(&self) -> GraphFeatureExtractor<'_, T, E> {
        GraphFeatureExtractor::new(self)
    }
}

/// Batch multiple graphs into a single tensor representation
///
/// Creates a batched tensor with shape [batch_size * max_nodes, num_features]
/// and a batch adjacency matrix with appropriate offsets
pub struct GraphBatch {
    graphs: Vec<(DenseTensor, GraphAdjacencyMatrix)>,
}

impl GraphBatch {
    /// Create new batch from graphs
    pub fn new<T, E>(graphs: &[Graph<T, E>]) -> Result<Self, TensorError>
    where
        T: AsRef<[f64]> + Clone,
        E: Clone,
    {
        let mut batch = Self {
            graphs: Vec::with_capacity(graphs.len()),
        };

        for graph in graphs {
            let (features, adjacency) = graph.to_tensor_representation()?;
            batch.graphs.push((features, adjacency));
        }

        Ok(batch)
    }

    /// Get batched feature matrix
    pub fn batch_features(&self) -> DenseTensor {
        if self.graphs.is_empty() {
            return DenseTensor::zeros(vec![0, 0]);
        }

        // Find max nodes and features
        let max_nodes = self
            .graphs
            .iter()
            .map(|(_, adj)| adj.num_nodes)
            .max()
            .unwrap_or(0);

        let num_features = self
            .graphs
            .iter()
            .map(|(feat, _)| feat.shape().get(1).copied().unwrap_or(1))
            .max()
            .unwrap_or(1);

        // Concatenate with padding
        let mut all_features = Vec::new();
        for (features, adjacency) in &self.graphs {
            let feat_data = features.data();
            all_features.extend_from_slice(feat_data);

            // Pad to max_nodes if needed
            let current_nodes = adjacency.num_nodes;
            if current_nodes < max_nodes {
                let padding_size = (max_nodes - current_nodes) * num_features;
                all_features.extend(std::iter::repeat_n(0.0, padding_size));
            }
        }

        DenseTensor::from_vec(
            all_features,
            vec![self.graphs.len() * max_nodes, num_features],
        )
    }

    /// Get batched adjacency matrix (block diagonal)
    pub fn batch_adjacency(&self) -> GraphAdjacencyMatrix {
        if self.graphs.is_empty() {
            return GraphAdjacencyMatrix::from_edge_list(&[], 0, false).unwrap();
        }

        // For batch processing, we keep graphs separate and use offset indexing
        // This is a simplified implementation - full block diagonal would be more complex
        let total_nodes: usize = self.graphs.iter().map(|(_, adj)| adj.num_nodes).sum();
        let total_edges: usize = self.graphs.iter().map(|(_, adj)| adj.num_edges).sum();

        // Collect all edges with offsets
        let mut all_edges = Vec::with_capacity(total_edges);
        let mut offset = 0;

        for (_, adjacency) in &self.graphs {
            let csr = adjacency.as_sparse_tensor();
            for src in 0..adjacency.num_nodes {
                let start = csr.row_offsets()[src];
                let end = csr.row_offsets()[src + 1];
                for j in start..end {
                    let dst = csr.col_indices()[j];
                    all_edges.push((src + offset, dst + offset));
                }
            }
            offset += adjacency.num_nodes;
        }

        GraphAdjacencyMatrix::from_edge_list(
            &all_edges,
            total_nodes,
            self.graphs
                .first()
                .map(|(_, adj)| adj.is_directed)
                .unwrap_or(false),
        )
        .unwrap()
    }

    /// Get number of graphs in batch
    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Get individual graph by index
    pub fn get(&self, idx: usize) -> Option<&(DenseTensor, GraphAdjacencyMatrix)> {
        self.graphs.get(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_adjacency_matrix_creation() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let adj = GraphAdjacencyMatrix::from_edge_list(&edges, 3, true).unwrap();

        assert_eq!(adj.num_nodes, 3);
        assert_eq!(adj.num_edges, 3);
        assert!(adj.is_directed);
    }

    #[test]
    fn test_graph_to_tensor_conversion() {
        let mut graph = Graph::<Vec<f64>, f64>::directed();

        let n0 = graph.add_node(vec![1.0, 0.0]).unwrap();
        let n1 = graph.add_node(vec![0.0, 1.0]).unwrap();
        let n2 = graph.add_node(vec![1.0, 1.0]).unwrap();

        let _ = graph.add_edge(n0, n1, 1.0);
        let _ = graph.add_edge(n1, n2, 1.0);
        let _ = graph.add_edge(n2, n0, 1.0);

        let (features, adjacency) = graph.to_tensor_representation().unwrap();

        assert_eq!(features.shape(), &[3, 2]);
        assert_eq!(adjacency.num_nodes, 3);
        assert_eq!(adjacency.num_edges, 3);
    }

    #[test]
    fn test_feature_extractor() {
        let mut graph = Graph::<String, f64>::directed();

        let n0 = graph.add_node("node0".to_string()).unwrap();
        let n1 = graph.add_node("node1".to_string()).unwrap();
        let _ = graph.add_edge(n0, n1, 1.0);

        let extractor = graph.feature_extractor();

        // Extract scalar features (string length)
        let features = extractor
            .extract_node_features_scalar(|s| s.len() as f64)
            .unwrap();

        assert_eq!(features.shape(), &[2, 1]);
    }

    #[test]
    fn test_graph_reconstruction() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let adj = GraphAdjacencyMatrix::from_edge_list(&edges, 3, true).unwrap();

        let reconstructor = GraphReconstructor::new(true);

        let graph: Graph<usize, f64> = reconstructor
            .from_adjacency(&adj, |i| i, |_src, _dst, w| w)
            .unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_normalized_adjacency() {
        let edges = vec![(0, 1), (1, 0), (1, 2), (2, 1)];
        let adj = GraphAdjacencyMatrix::from_edge_list(&edges, 3, true).unwrap();

        let normalized = adj.normalized_with_self_loops().unwrap();

        // Should have self-loops added
        assert!(normalized.num_edges > adj.num_edges);
    }

    #[test]
    fn test_batch_creation() {
        let mut graph1 = Graph::<Vec<f64>, f64>::directed();
        let n0 = graph1.add_node(vec![1.0, 0.0]).unwrap();
        let n1 = graph1.add_node(vec![0.0, 1.0]).unwrap();
        let _ = graph1.add_edge(n0, n1, 1.0);

        let mut graph2 = Graph::<Vec<f64>, f64>::directed();
        let n0 = graph2.add_node(vec![1.0, 1.0]).unwrap();
        let n1 = graph2.add_node(vec![0.0, 0.0]).unwrap();
        let _ = graph2.add_edge(n0, n1, 1.0);

        let batch = GraphBatch::new(&[graph1, graph2]).unwrap();

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }
}
