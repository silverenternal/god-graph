//! Batch processing utilities for graph algorithms
//!
//! Provides batch processing capabilities for improved throughput
//! when processing multiple queries or operations.
//!
//! # Examples
//!
//! ```rust,no_run
//! use god_graph::graph::Graph;
//! use god_graph::utils::batch::BatchProcessor;
//!
//! let graph: Graph<i32, f64> = Graph::undirected();
//! // ... add nodes and edges ...
//!
//! // Batch process multiple BFS queries
//! let queries = vec![/* start nodes */];
//! let results: Vec<_> = BatchProcessor::new(queries)
//!     .with_batch_size(64)
//!     .process(|start_node| {
//!         // BFS from each start node
//!         god_graph::algorithms::traversal::bfs(&graph, start_node, |_, _| true);
//!     });
//! ```

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Batch processor for efficient bulk operations
///
/// Processes multiple operations in batches to improve:
/// - Cache locality (process related data together)
/// - Parallel throughput (distribute batches across cores)
/// - Memory efficiency (reuse buffers across batch)
pub struct BatchProcessor<T, B = Vec<T>> {
    /// Items to process
    items: B,
    /// Batch size for chunking
    batch_size: usize,
    /// Pre-allocated buffer for reuse
    buffer: Vec<T>,
}

impl<T> BatchProcessor<T, Vec<T>> {
    /// Create a new batch processor
    pub fn new(items: Vec<T>) -> Self {
        Self {
            items,
            batch_size: 64,
            buffer: Vec::with_capacity(64),
        }
    }

    /// Create a new batch processor with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            batch_size: 64,
            buffer: Vec::with_capacity(64),
        }
    }
}

impl<T, B: AsRef<[T]>> BatchProcessor<T, B> {
    /// Set the batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Get the number of items to process
    pub fn len(&self) -> usize {
        self.items.as_ref().len()
    }

    /// Check if there are no items to process
    pub fn is_empty(&self) -> bool {
        self.items.as_ref().is_empty()
    }
}

impl<T: Clone + Send + 'static> BatchProcessor<T, Vec<T>> {
    /// Process items in parallel batches
    ///
    /// Divides items into batches and processes each batch in parallel.
    /// Reuses buffers across batch operations to reduce allocations.
    ///
    /// # Arguments
    /// * `f` - Processing function called for each item
    ///
    /// # Returns
    /// Vector of results from processing each item
    ///
    /// # Performance
    /// - Uses rayon for parallel batch processing
    /// - Reuses buffers to reduce allocation overhead
    /// - Best for batches with > 100 items
    #[cfg(feature = "parallel")]
    pub fn process_par<F, R>(self, f: F) -> Vec<R>
    where
        F: Fn(T) -> R + Send + Sync + Copy,
        R: Send + 'static,
    {
        let Self {
            items,
            batch_size,
            buffer: _,
        } = self;

        // Chunk items into batches and process in parallel
        items
            .into_par_iter()
            .chunks(batch_size)
            .flat_map(move |chunk| {
                // Process each chunk sequentially
                chunk.into_iter().map(f).collect::<Vec<R>>()
            })
            .collect()
    }

    /// Process items sequentially with buffer reuse
    ///
    /// Processes items one by one, but reuses the internal buffer
    /// to reduce allocations in the processing function.
    ///
    /// # Arguments
    /// * `f` - Processing function that receives item and buffer
    ///
    /// # Returns
    /// Vector of results from processing each item
    pub fn process_sequential<F, R>(self, mut f: F) -> Vec<R>
    where
        F: FnMut(&T, &mut Vec<T>) -> R,
    {
        let Self {
            items,
            batch_size: _,
            buffer,
        } = self;

        let mut results = Vec::with_capacity(items.len());
        let mut buffer = buffer;

        for item in items {
            // Clear buffer for reuse
            buffer.clear();
            let result = f(&item, &mut buffer);
            results.push(result);
        }

        results
    }
}

/// Batch query processor for multiple graph queries
///
/// Optimized for scenarios where you need to run the same
/// algorithm from multiple start nodes (e.g., multi-source BFS,
/// all-pairs shortest paths, centrality computations).
pub struct GraphBatchQuery<G, T> {
    /// Graph reference
    graph: G,
    /// Query parameters (e.g., start nodes)
    queries: Vec<T>,
    /// Batch size
    batch_size: usize,
}

impl<G, T> GraphBatchQuery<G, T> {
    /// Create a new batch query processor
    pub fn new(graph: G, queries: Vec<T>) -> Self {
        Self {
            graph,
            queries,
            batch_size: 64,
        }
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Get the number of queries
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Check if there are no queries
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

#[cfg(feature = "parallel")]
impl<G, T> GraphBatchQuery<G, T>
where
    G: Send + Sync,
    T: Send + Sync,
{
    /// Execute batch queries in parallel
    ///
    /// # Arguments
    /// * `query_fn` - Function to execute for each query
    ///
    /// # Returns
    /// Vector of query results
    ///
    /// # Example
    /// ```rust,no_run
    /// use god_graph::graph::Graph;
    /// use god_graph::utils::batch::GraphBatchQuery;
    /// use god_graph::algorithms::traversal::bfs;
    ///
    /// let graph: Graph<i32, f64> = Graph::undirected();
    /// // ... add nodes and edges ...
    ///
    /// let start_nodes = vec![/* multiple start nodes */];
    /// let results = GraphBatchQuery::new(&graph, start_nodes)
    ///     .with_batch_size(32)
    ///     .execute_par(|&graph, &start| {
    ///         let mut count = 0;
    ///         bfs(&graph, start, |_| { count += 1; true });
    ///         count
    ///     });
    /// ```
    pub fn execute_par<F, R>(self, query_fn: F) -> Vec<R>
    where
        F: Fn(&G, &T) -> R + Send + Sync,
        R: Send + 'static,
    {
        let Self {
            graph,
            queries,
            batch_size,
        } = self;

        queries
            .into_par_iter()
            .chunks(batch_size)
            .flat_map(|chunk| {
                chunk
                    .into_iter()
                    .map(|query| query_fn(&graph, &query))
                    .collect::<Vec<R>>()
            })
            .collect()
    }
}

impl<G, T> GraphBatchQuery<G, T>
where
    G: Clone,
{
    /// Execute batch queries sequentially
    ///
    /// # Arguments
    /// * `query_fn` - Function to execute for each query
    ///
    /// # Returns
    /// Vector of query results
    pub fn execute_sequential<F, R>(self, mut query_fn: F) -> Vec<R>
    where
        F: FnMut(&G, &T) -> R,
    {
        let Self {
            graph,
            queries,
            batch_size: _,
        } = self;

        queries
            .into_iter()
            .map(|query| query_fn(&graph, &query))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::new(vec![1, 2, 3, 4, 5]);
        assert_eq!(processor.len(), 5);
        assert!(!processor.is_empty());
    }

    #[test]
    fn test_batch_processor_empty() {
        let processor: BatchProcessor<i32> = BatchProcessor::new(vec![]);
        assert_eq!(processor.len(), 0);
        assert!(processor.is_empty());
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_batch_processor_parallel() {
        let processor = BatchProcessor::new(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let results: Vec<_> = processor.process_par(|x| x * 2);
        assert_eq!(results, vec![2, 4, 6, 8, 10, 12, 14, 16]);
    }

    #[test]
    fn test_batch_processor_sequential() {
        let processor = BatchProcessor::new(vec![1, 2, 3, 4, 5]);
        let results: Vec<_> = processor.process_sequential(|&x, buffer| {
            buffer.push(x);
            x * 2
        });
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }
}
