# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-03-29

### Added
- **ModelSwitch Export Function** (`src/transformer/optimization/switch.rs`)
  - `save_to_safetensors()` - Export GodGraph to HuggingFace Safetensors format
  - Bidirectional conversion: Safetensors ↔ GodGraph
  - Supports F32, F64, F16 data types
  - Round-trip precision loss < 1e-5
  - Uses `TensorView` and `BTreeMap` for lifetime management
- **ModelSwitch Integration Tests**
  - `test_save_to_safetensors` - Full save/load/verify workflow
  - `test_save_load_round_trip` - Precision verification
  - Tests weight preservation across conversion
- **ModelSwitch Example** (`examples/cad_llm_switch.rs`)
  - Complete workflow demonstration
  - Creates demo GodGraph with Embedding, Attention, MLP, Norm nodes
  - Topology validation
  - Weight verification
  - Export to Safetensors
- **WeightTensor Debug Trait**
  - Added `#[derive(Debug)]` for better error reporting
- **Topology Validation**
  - `validate_topology()` - Checks connectivity, cycles, isolated nodes
  - `verify_weights()` - L2 norm comparison between graphs
- **Operator Type Inference**
  - Automatic operator type inference from weight names
  - Supports Attention, MLP, Norm, Embedding, Linear operators

### Changed
- **Documentation Updates**
  - README.md - Added ModelSwitch export section
  - CAD_LLM_1B_VALIDATION_REPORT.md - Updated with export validation
  - TODO_IMPLEMENTATION_STATUS.md - Added Phase 5 completion status
- **Test Count**
  - Total tests: 344 → 346 (added 2 ModelSwitch export tests)
  - All tests passing (100% pass rate)

### Removed
- Beta status - First stable release with LLM optimization features

### Fixed
- **cad_editor.rs Warnings**
  - Fixed unused variable warnings in pattern matching
- **Example Compilation**
  - Fixed `cad_llm_switch.rs` imports for GraphBase and GraphOps traits

### Technical Details
- **ModelSwitch Implementation**
  - Two-phase serialization: collect data first, then create TensorViews
  - Proper lifetime management for borrow checker
  - F64 → F32 conversion for storage efficiency
  - Weight name preservation for matching

### Performance
- **Export Performance**
  - 1.78 MB Safetensors file (demo graph with 4 nodes, 7 edges)
  - Round-trip time: < 100ms for typical models

### Documentation
- **Updated Files**:
  - `README.md` - ModelSwitch usage examples
  - `CAD_LLM_1B_VALIDATION_REPORT.md` - Export validation results
  - `TODO_IMPLEMENTATION_STATUS.md` - Phase 5 completion
  - `CHANGELOG.md` - This changelog

### Testing
- **Test Coverage**: 346 tests all passing
  - 298 unit tests
  - 20 integration tests
  - 15 property tests
  - 13 doc tests

---

## [0.4.3-beta] - 2026-03-28

### Added
- **DifferentiableGraph** - Differentiable graph structure for neural architecture search
- **Lie Group Orthogonalization** - Weight matrix orthogonalization using Lie group theory
- **Tensor Ring Compression** - Tensor ring decomposition for model compression
- **Topology Constraint Solving** - CAD-style topology validation and repair
- **Memory Pool** - Tensor memory pool with 98-99.9% allocation reduction
- **UnifiedGraph** - Unified graph structure for multi-modal models

### Changed
- **Version Number**: 0.3.0-beta → 0.4.3-beta (reflects CAD-LLM integration maturity)
- **Test Count**: 122 → 344 tests (182% increase)

### Fixed
- **QR Decomposition** - Numerical stability fix with reorthogonalization
- **Example Files** - Fixed imports and type errors in CAD-LLM examples

### Added (v0.3.1-beta)
- **SIMD Vectorization Support** (Experimental)
  - `par_pagerank_simd` - Batch PageRank computation using `wide::f64x4`
  - `par_degree_centrality_simd` - SIMD-optimized degree centrality
  - Uses `wide` crate, supports stable Rust, enable via `simd` feature
  - Achieves 2-4x speedup on CPUs with AVX2/AVX-512 support
- **SVG Visualization Export**
  - `to_svg` and `to_svg_with_options` functions
  - Supports 3 layout algorithms: Force-Directed, Circular, Hierarchical
  - Customizable node colors, sizes, labels
  - Interactive web viewer (`examples/graph_viewer.html`)
- **Test Coverage Integration**
  - Integrated `cargo-tarpaulin` for coverage statistics
  - Current coverage: 66.64% (target 80%+)
  - Added `coverage.sh` script
- **Documentation Enhancements**
  - Added SVG export examples to README
  - Added test coverage section

### Fixed (v0.3.1 Candidate)
- Flow algorithm memory optimization clarification: Current implementation uses O(V+E) adjacency list, but initialization has O(V) fixed overhead
  - Plan: Use `HashMap<usize, Vec<(usize, f64)>>` instead of `Vec<Vec<(usize, f64)>>`
  - Reduces memory waste for ultra-large-scale sparse graphs (V >> E)

---

## [0.3.0-beta] - 2026-03-26

### Added
- **Comprehensive Performance Benchmark Report** (`docs/performance.md`)
  - PageRank 1000 nodes: Serial 53.9ms → Parallel 668µs = **80.7x speedup**
  - DFS 50K nodes: Serial 9.7ms → Parallel 1.3ms = **7.5x speedup**
  - Connected Components 2000 nodes: 357.8µs
  - Degree Centrality 5000 nodes: 146µs
- **petgraph Migration Guide** (`docs/migration-from-petgraph.md`)
  - 412-line comprehensive migration document
  - Core differences comparison table
  - API reference table
  - Complete code examples
- **Parallel Algorithm Suite** (based on rayon)
  - `par_dfs` - Subtree parallel DFS
  - `par_bfs` - Layered parallel BFS
  - `par_dijkstra` - Parallel relaxation Dijkstra
  - `par_pagerank` - Iteration-parallel PageRank
  - `par_connected_components` - Parallel union-find
- **Benchmark Configuration**
  - `speedup.rs` - Complete speedup benchmarks
  - `parallel.rs` - Parallel algorithm performance tests
  - `centrality.rs` - Centrality algorithm benchmarks

### Changed
- **Version Number**: 0.1.0 → 0.3.0-beta (reflects project maturity)
- **README Performance Data Updates**
  - Updated performance claims to match performance.md
  - Added detailed performance data tables
  - Updated version numbers in example code
- **Architecture Documentation Clarification**
  - `ROADMAP.json` clarifies "arena-style slot management" (not independent Arena type)
  - `ROADMAP.json` clarifies "bucket-based CSR variant" (not traditional CSR)

### Fixed
- **centrality.rs Compilation Error**
  - Fixed `rng.gen::<f64>()` → `rng.gen_range(0.0..1.0)` (Rust 2024 reserved keyword changes)
- **par_connected_components Deadlock**
  - Root cause: Parallel union-find path compression causes race conditions
  - Fix: Removed path compression, uses atomic CAS operations
- **Doctest Ignore Issues**
  - Fixed `src/lib.rs` line 39 doctest (added `par_pagerank` import)
  - Fixed `src/utils/arena.rs` doctest (`gen` → `generation` keyword)
  - Achieved 100% pass rate: 23 passed, 0 ignored
- **Clippy Warnings**
  - Fixed `needless_range_loop` warning
  - Main code maintains 0 warnings

### Performance
- **PageRank**: 80.7x speedup (1000 nodes, 53.9ms → 668µs)
- **DFS**: 7.5x speedup (50K nodes, 9.7ms → 1.3ms)
- **Connected Components**: 357.8µs (2000 nodes)
- **Degree Centrality**: 146µs (5000 nodes)

### Documentation
- **New Files**:
  - `docs/performance.md` - Performance benchmark report
  - `docs/migration-from-petgraph.md` - petgraph migration guide
- **Updated Files**:
  - `README.md` - Performance data, version numbers
  - `ROADMAP.json` - Architecture implementation details clarification

### Testing
- **Test Coverage**: 122 tests all passing
  - 88 unit tests
  - 19 integration tests
  - 15 property tests
  - 23 doctests (100% pass rate)
- **Clippy**: 0 warnings (main code)

### Technical Details
- **Bucket-based CSR Implementation**
  - `AdjBucket` struct: `neighbors + edge_indices + deleted_mask(bitmap) + deleted_count`
  - `#[repr(align(64))]` cache line alignment
  - O(1) incremental insertion, lazy deletion, on-demand compaction
- **Arena-style Slot Management**
  - `NodeSlot`: `data: Option<T> + generation: u32`
  - `EdgeStorage`: `source + target + data + generation: u32`
  - `free_nodes + free_edges` free lists
- **Generation Validation**
  - `contains_node`: Checks `slot.generation == index.generation()`
  - `get_node`: Validates generation and returns `NodeDeleted` error
  - `remove_node`: Validates generation
  - `neighbors`: Automatically gets target node's latest generation

---

## [0.2.0-alpha] - 2026-03-26

### Added
- **Bucket-based CSR Memory Layout**
  - Supports O(1) incremental updates
  - 64-byte alignment optimization
  - Software prefetching support (conditional compilation nightly)
- **Generation Validation Mechanism**
  - Fully integrated into `add_node`/`get_node`/`update_node`/`remove_node`/`contains_node`
  - Prevents ABA problems
- **Complete Algorithm Suite**
  - Traversal: DFS, BFS, Topological Sort, Tarjan SCC
  - Shortest Paths: Dijkstra, Bellman-Ford, Floyd-Warshall, A*
  - Minimum Spanning Tree: Kruskal, Prim
  - Centrality: Degree, Betweenness, Closeness, PageRank
  - Community Detection: Connected Components, Label Propagation
- **Random Graph Generators**
  - Erdős-Rényi Model
  - Barabási-Albert Model
  - Watts-Strogatz Model
  - Complete Graph, Grid Graph, Tree
- **Graph Export Functions**
  - DOT/Graphviz Format
  - Adjacency List, Edge List

### Changed
- **Flow Algorithm Memory Optimization**
  - Residual graph changed from `Vec<Vec<f64>>` to O(V+E) adjacency list
- **API Improvements**
  - Generic trait design
  - Iterator API follows Rust conventions

### Fixed
- **Parallel Algorithm Crash**
  - Debugged and fixed `par_connected_components` deadlock
- **Benchmark Memory Optimization**
  - Reduced benchmark scale to avoid memory explosion

---

## [0.1.0-alpha] - Initial Release

### Added
- **Core Graph Structure**
  - `Graph<T, E>` generic graph type
  - `NodeIndex` / `EdgeIndex` newtype wrappers
  - Adjacency list representation
- **Basic CRUD Operations**
  - `add_node` / `get_node` / `update_node` / `remove_node`
  - `add_edge` / `get_edge` / `update_edge` / `remove_edge`
  - `neighbors` / `incident_edges` / `has_edge` / `degree`
- **Basic Algorithms**
  - DFS (recursive/iterative)
  - BFS (standard/layered)
  - Topological Sort
  - Dijkstra's Algorithm
- **Graph Builder**
  - `GraphBuilder` chain API
- **Test Suite**
  - Unit Tests
  - Integration Tests
  - Property Tests (proptest)

---

## Migration Guide

### From 0.4.3-beta to 0.5.0-alpha

1. **Update Cargo.toml**:
   ```toml
   [dependencies]
   god-gragh = "0.5.0-alpha"
   ```

2. **Enable Safetensors Feature** (optional):
   ```toml
   god-gragh = { version = "0.5.0-alpha", features = ["safetensors"] }
   ```

3. **New API Usage**:
   ```rust
   use god_gragh::transformer::optimization::ModelSwitch;
   
   // Export GodGraph to Safetensors
   ModelSwitch::save_to_safetensors(&graph, "optimized.safetensors")?;
   
   // Load from Safetensors
   let graph = ModelSwitch::load_from_safetensors("model.safetensors")?;
   ```

4. **API Changes**: No breaking changes, additive only

### From 0.1.0 to 0.5.0-alpha

1. **Update Cargo.toml**:
   ```toml
   [dependencies]
   god-gragh = "0.5.0-alpha"
   ```

2. **Enable Parallel Feature** (optional):
   ```toml
   god-gragh = { version = "0.5.0-alpha", features = ["parallel"] }
   ```

3. **API Changes**: No breaking changes

4. **Performance Improvement**: Automatically enjoy 80x PageRank speedup

### From petgraph to god-gragh

For detailed migration guide, see [`docs/migration-from-petgraph.md`](docs/migration-from-petgraph.md).

Core differences:
- God-Graph uses **bucket-based CSR variant** instead of traditional adjacency list
- God-Graph uses **arena-style slot management** instead of independent Arena type
- God-Graph includes **built-in parallel algorithm suite**
- God-Graph provides **generation-indexed stability**

---

## Version History Summary

| Version | Release Date | Status | Key Features |
|---------|--------------|--------|--------------|
| 0.1.0-alpha | 2026-03-26 | Released | Core graph structure, basic CRUD, DFS/BFS |
| 0.2.0-alpha | 2026-03-26 | Released | Bucket-based CSR, complete algorithm suite, random graph generation |
| 0.3.0-beta | 2026-03-26 | Released | Performance reports, migration guide, parallel algorithm verification |
| 0.4.3-beta | 2026-03-28 | Released | DifferentiableGraph, Lie Group, Tensor Ring, Memory Pool |
| **0.5.0-alpha** | **2026-03-29** | **Current** | **ModelSwitch Export**, Safetensors bidirectional conversion |
| 0.6.0-beta | Planned | - | Memory pool benchmarks, GraphTransformer execution engine |
| 1.0.0-rc | Planned | - | API stabilization, production-ready |

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code passes `cargo clippy` and `cargo fmt`
- Add appropriate tests
- Update documentation

## License

This project is licensed under either of:
- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

## Contact

- Issues: [GitHub Issues](https://github.com/silverenternal/god-gragh/issues)
- Discussions: [GitHub Discussions](https://github.com/silverenternal/god-gragh/discussions)
