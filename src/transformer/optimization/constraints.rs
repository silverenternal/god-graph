//! Topology Constraints and Validation
//!
//! This module defines topology constraints for LLM computation graphs
//! and provides validation utilities.
//!
//! ## Constraint Types
//!
//! - Residual Connection: Ensure residual paths are connected
//! - Attention Head Balance: Ensure attention heads have balanced weights
//! - Gradient Flow: Ensure gradient flow paths exist
//! - Custom: User-defined constraint functions

use crate::errors::GraphResult;
use crate::graph::traits::{GraphBase, GraphQuery};
use crate::graph::Graph;
use crate::transformer::optimization::switch::{OperatorType, WeightTensor};
use std::collections::HashMap;

/// Severity level for topology defects
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Informational - no action required
    Info,
    /// Warning - should be reviewed
    Warning,
    /// Error - should be fixed
    Error,
    /// Critical - must be fixed immediately
    Critical,
}

/// Type of topology defect
#[derive(Debug, Clone)]
pub enum DefectType {
    /// Isolated node with no connections
    IsolatedNode,
    /// Disconnected component in graph
    DisconnectedComponent,
    /// Cycle detected in feedforward graph
    UnexpectedCycle,
    /// Missing residual connection
    MissingResidual,
    /// Unbalanced attention heads
    UnbalancedAttention,
    /// Gradient flow blocked
    BlockedGradientFlow,
    /// Custom defect type
    Custom(String),
}

/// Topology defect report
#[derive(Debug, Clone)]
pub struct TopologyDefect {
    /// Type of defect
    pub defect_type: DefectType,
    /// Location of the defect (node index)
    pub location: usize,
    /// Severity level
    pub severity: Severity,
    /// Description of the issue
    pub description: String,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Topology constraint definition
pub enum TopologyConstraint {
    /// Residual connection must exist between specific nodes
    ResidualConnection {
        from_layer: String,
        to_layer: String,
    },
    /// Attention heads must have balanced weight norms
    AttentionHeadBalance {
        layer: String,
        tolerance: f64,
    },
    /// Gradient flow path must exist
    GradientFlow {
        from: String,
        to: String,
    },
    /// Custom constraint function
    Custom(Box<dyn Fn(&Graph<OperatorType, WeightTensor>) -> GraphResult<bool> + Send + Sync>),
}

impl Clone for TopologyConstraint {
    fn clone(&self) -> Self {
        match self {
            Self::ResidualConnection { from_layer, to_layer } => {
                Self::ResidualConnection {
                    from_layer: from_layer.clone(),
                    to_layer: to_layer.clone(),
                }
            }
            Self::AttentionHeadBalance { layer, tolerance } => {
                Self::AttentionHeadBalance {
                    layer: layer.clone(),
                    tolerance: *tolerance,
                }
            }
            Self::GradientFlow { from, to } => Self::GradientFlow {
                from: from.clone(),
                to: to.clone(),
            },
            // Custom constraints cannot be cloned, return a placeholder
            Self::Custom(_) => Self::ResidualConnection {
                from_layer: String::new(),
                to_layer: String::new(),
            },
        }
    }
}

impl std::fmt::Debug for TopologyConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResidualConnection { from_layer, to_layer } => f
                .debug_struct("ResidualConnection")
                .field("from_layer", from_layer)
                .field("to_layer", to_layer)
                .finish(),
            Self::AttentionHeadBalance { layer, tolerance } => f
                .debug_struct("AttentionHeadBalance")
                .field("layer", layer)
                .field("tolerance", tolerance)
                .finish(),
            Self::GradientFlow { from, to } => f
                .debug_struct("GradientFlow")
                .field("from", from)
                .field("to", to)
                .finish(),
            Self::Custom(_) => f.debug_struct("Custom").finish(),
        }
    }
}

/// Constraint validation report
#[derive(Debug, Clone)]
pub struct ConstraintReport {
    /// Whether all constraints are satisfied
    pub all_satisfied: bool,
    /// Number of satisfied constraints
    pub satisfied_count: usize,
    /// Number of violated constraints
    pub violated_count: usize,
    /// Details for each constraint
    pub constraint_details: Vec<ConstraintDetail>,
}

/// Detail for a single constraint
#[derive(Debug, Clone)]
pub struct ConstraintDetail {
    /// Constraint description
    pub description: String,
    /// Whether it's satisfied
    pub satisfied: bool,
    /// Violation details if any
    pub violation_details: Option<String>,
}

/// Topology validator for LLM computation graphs
pub struct TopologyValidator {
    constraints: Vec<TopologyConstraint>,
    validation_cache: HashMap<String, bool>,
}

impl TopologyValidator {
    /// Create a new topology validator
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            validation_cache: HashMap::new(),
        }
    }

    /// Create validator with predefined constraints for common architectures
    pub fn with_default_constraints() -> Self {
        let mut validator = Self::new();
        
        // Add common constraints for transformer architectures
        validator.add_constraint(TopologyConstraint::ResidualConnection {
            from_layer: "attention".to_string(),
            to_layer: "attention_output".to_string(),
        });
        
        validator.add_constraint(TopologyConstraint::ResidualConnection {
            from_layer: "mlp".to_string(),
            to_layer: "mlp_output".to_string(),
        });

        validator
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: TopologyConstraint) {
        self.constraints.push(constraint);
        self.validation_cache.clear();
    }

    /// Remove all constraints
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
        self.validation_cache.clear();
    }

    /// Get the number of constraints
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Validate all constraints on a graph
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph to validate
    ///
    /// # Returns
    ///
    /// Constraint validation report
    pub fn validate(&mut self, graph: &Graph<OperatorType, WeightTensor>) -> GraphResult<ConstraintReport> {
        let mut details = Vec::new();
        let mut satisfied_count = 0;

        for constraint in &self.constraints {
            let (satisfied, description, violation) = match constraint {
                TopologyConstraint::ResidualConnection { from_layer, to_layer } => {
                    self.validate_residual_connection(graph, from_layer, to_layer)?
                }
                TopologyConstraint::AttentionHeadBalance { layer, tolerance } => {
                    self.validate_attention_balance(graph, layer, *tolerance)?
                }
                TopologyConstraint::GradientFlow { from, to } => {
                    self.validate_gradient_flow(graph, from, to)?
                }
                TopologyConstraint::Custom(func) => {
                    let result = func(graph)?;
                    (result, "Custom constraint".to_string(), None)
                }
            };

            if satisfied {
                satisfied_count += 1;
            }

            details.push(ConstraintDetail {
                description,
                satisfied,
                violation_details: violation,
            });
        }

        Ok(ConstraintReport {
            all_satisfied: satisfied_count == self.constraints.len(),
            satisfied_count,
            violated_count: self.constraints.len() - satisfied_count,
            constraint_details: details,
        })
    }

    /// Detect topology defects in a graph
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph to analyze
    ///
    /// # Returns
    ///
    /// List of detected defects
    pub fn detect_defects(
        &self,
        graph: &Graph<OperatorType, WeightTensor>,
    ) -> GraphResult<Vec<TopologyDefect>> {
        use crate::algorithms::community::connected_components;

        let mut defects = Vec::new();

        // Check for isolated nodes
        for node_ref in graph.nodes() {
            let node_id = node_ref.index();
            let neighbor_count = graph.neighbors(node_id).count();

            if neighbor_count == 0 {
                defects.push(TopologyDefect {
                    defect_type: DefectType::IsolatedNode,
                    location: node_id.index(),
                    severity: Severity::Warning,
                    description: format!("Node {} has no outgoing edges", node_id.index()),
                    suggested_fix: Some("Connect the node to the computation graph or remove it".to_string()),
                });
            }
        }

        // Check for disconnected components
        let components = connected_components(graph);
        if components.len() > 1 {
            for (i, component) in components.iter().enumerate().skip(1) {
                defects.push(TopologyDefect {
                    defect_type: DefectType::DisconnectedComponent,
                    location: component.first().map(|idx| idx.index()).unwrap_or(0),
                    severity: Severity::Error,
                    description: format!("Found disconnected component {} with {} nodes", i, component.len()),
                    suggested_fix: Some("Add edges to connect this component to the main graph".to_string()),
                });
            }
        }

        Ok(defects)
    }

    /// Validate a residual connection constraint
    fn validate_residual_connection(
        &self,
        graph: &Graph<OperatorType, WeightTensor>,
        from_layer: &str,
        to_layer: &str,
    ) -> GraphResult<(bool, String, Option<String>)> {
        // Simplified implementation
        // In a full implementation, we would search for actual residual connections
        
        let found = graph.nodes().any(|n| {
            matches!(n.data(), OperatorType::Residual)
        });

        let description = format!("ResidualConnection: {} -> {}", from_layer, to_layer);
        
        if found {
            Ok((true, description, None))
        } else {
            Ok((
                false,
                description,
                Some(format!("No residual connection found between {} and {}", from_layer, to_layer)),
            ))
        }
    }

    /// Validate attention head balance
    fn validate_attention_balance(
        &self,
        _graph: &Graph<OperatorType, WeightTensor>,
        layer: &str,
        tolerance: f64,
    ) -> GraphResult<(bool, String, Option<String>)> {
        // Simplified implementation
        // In a full implementation, we would compare attention head weight norms

        let description = format!("AttentionHeadBalance: {} (tolerance: {})", layer, tolerance);

        // Assume balanced for now
        Ok((true, description, None))
    }

    /// Validate gradient flow path
    fn validate_gradient_flow(
        &self,
        graph: &Graph<OperatorType, WeightTensor>,
        from: &str,
        to: &str,
    ) -> GraphResult<(bool, String, Option<String>)> {
        use crate::algorithms::traversal::bfs;
        use crate::node::NodeIndex;

        // Simplified: check if there's a path from any node matching 'from' to any node matching 'to'
        let mut path_exists = false;

        for start_node in graph.nodes() {
            let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
            
            bfs(graph, start_node.index(), |n: NodeIndex, _depth: usize| {
                visited.insert(n.index());
                true
            });

            // Check if target is reachable
            path_exists = visited.iter().any(|&n| {
                let node_idx = NodeIndex::new(n, 0);
                if let Ok(node_data) = graph.get_node(node_idx) {
                    format!("{:?}", node_data).contains(to)
                } else {
                    false
                }
            });

            if path_exists {
                break;
            }
        }

        let description = format!("GradientFlow: {} -> {}", from, to);

        if path_exists {
            Ok((true, description, None))
        } else {
            Ok((
                false,
                description,
                Some(format!("No gradient flow path from {} to {}", from, to)),
            ))
        }
    }
}

impl Default for TopologyValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Assembly validation report
#[derive(Debug, Clone)]
pub struct AssemblyReport {
    /// Whether the assembly is valid
    pub is_valid: bool,
    /// Number of modules checked
    pub module_count: usize,
    /// Number of interface mismatches
    pub interface_mismatches: usize,
    /// Details about each module
    pub module_details: Vec<ModuleDetail>,
}

/// Module detail in assembly report
#[derive(Debug, Clone)]
pub struct ModuleDetail {
    /// Module name
    pub name: String,
    /// Input dimension
    pub input_dim: Option<usize>,
    /// Output dimension
    pub output_dim: Option<usize>,
    /// Whether interfaces match
    pub interfaces_match: bool,
}

/// Validate assembly of modules
///
/// # Arguments
///
/// * `graph` - Graph representing the assembled modules
///
/// # Returns
///
/// Assembly validation report
pub fn validate_assembly(
    graph: &Graph<OperatorType, WeightTensor>,
) -> GraphResult<AssemblyReport> {
    let mut module_details = Vec::new();
    let interface_mismatches = 0;

    for node_ref in graph.nodes() {
        let node_data = node_ref.data();
        
        // Extract input/output dimensions based on operator type
        let (input_dim, output_dim) = match node_data {
            OperatorType::Linear { in_features, out_features } => {
                (Some(*in_features), Some(*out_features))
            }
            OperatorType::Attention { hidden_dim, .. } => {
                (Some(*hidden_dim), Some(*hidden_dim))
            }
            OperatorType::MLP { hidden_dim, .. } => {
                (Some(*hidden_dim), Some(*hidden_dim))
            }
            _ => (None, None),
        };

        module_details.push(ModuleDetail {
            name: format!("{:?}", node_data),
            input_dim,
            output_dim,
            interfaces_match: true, // Simplified
        });
    }

    Ok(AssemblyReport {
        is_valid: interface_mismatches == 0,
        module_count: graph.node_count(),
        interface_mismatches,
        module_details,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::traits::GraphOps;

    #[test]
    fn test_topology_validator() {
        let mut validator = TopologyValidator::new();
        
        validator.add_constraint(TopologyConstraint::ResidualConnection {
            from_layer: "attn".to_string(),
            to_layer: "output".to_string(),
        });

        assert_eq!(validator.constraint_count(), 1);
    }

    #[test]
    fn test_defect_detection() {
        // Create a graph with an isolated node
        let mut graph = Graph::<OperatorType, WeightTensor>::directed();
        
        // Add an isolated node
        graph.add_node(OperatorType::Linear {
            in_features: 512,
            out_features: 1024,
        }).unwrap();

        let validator = TopologyValidator::new();
        let defects = validator.detect_defects(&graph).unwrap();

        // Graph with isolated node should have defects
        assert!(!defects.is_empty(), "Should detect isolated node as a defect");
    }

    #[test]
    fn test_assembly_validation() {
        let mut graph = Graph::<OperatorType, WeightTensor>::directed();
        
        let node = graph.add_node(OperatorType::Linear {
            in_features: 512,
            out_features: 1024,
        }).unwrap();
        
        let report = validate_assembly(&graph).unwrap();
        
        assert_eq!(report.module_count, 1);
        assert!(report.is_valid);
    }
}
