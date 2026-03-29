//! Node types for graph-structured Transformer

use crate::tensor::DenseTensor;
use crate::tensor::traits::TensorBase;

/// Type of graph node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphNodeType {
    /// Token embedding node
    TokenEmbedding,
    /// Hidden state node
    HiddenState,
    /// Attention output node
    AttentionOutput,
    /// FFN output node
    FFNOutput,
}

/// Token embedding node data
#[derive(Debug, Clone)]
pub struct TokenEmbeddingNode {
    /// Token ID
    pub token_id: usize,
    /// Position in sequence
    pub position: usize,
    /// Embedding vector [1, hidden_dim]
    pub embedding: DenseTensor,
}

impl TokenEmbeddingNode {
    /// Create a new token embedding node
    pub fn new(token_id: usize, position: usize, embedding: DenseTensor) -> Self {
        Self {
            token_id,
            position,
            embedding,
        }
    }

    /// Get the hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.embedding.shape()[1]
    }
}

/// Hidden state node data
#[derive(Debug, Clone)]
pub struct HiddenStateNode {
    /// Layer number
    pub layer: usize,
    /// Position in sequence
    pub position: usize,
    /// Hidden state vector [1, hidden_dim]
    pub state: DenseTensor,
}

impl HiddenStateNode {
    /// Create a new hidden state node
    pub fn new(layer: usize, position: usize, state: DenseTensor) -> Self {
        Self {
            layer,
            position,
            state,
        }
    }

    /// Get the hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.state.shape()[1]
    }
}

/// Attention output node data
#[derive(Debug, Clone)]
pub struct AttentionOutputNode {
    /// Layer number
    pub layer: usize,
    /// Attention head
    pub head: usize,
    /// Query position
    pub query_pos: usize,
    /// Attended positions
    pub attended_positions: Vec<usize>,
    /// Attention weights
    pub weights: Vec<f64>,
    /// Output vector [1, head_dim]
    pub output: DenseTensor,
}

impl AttentionOutputNode {
    /// Create a new attention output node
    pub fn new(
        layer: usize,
        head: usize,
        query_pos: usize,
        attended_positions: Vec<usize>,
        weights: Vec<f64>,
        output: DenseTensor,
    ) -> Self {
        Self {
            layer,
            head,
            query_pos,
            attended_positions,
            weights,
            output,
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.output.shape()[1]
    }

    /// Get number of attended positions
    pub fn num_attended(&self) -> usize {
        self.attended_positions.len()
    }
}

/// FFN output node data
#[derive(Debug, Clone)]
pub struct FFNOutputNode {
    /// Layer number
    pub layer: usize,
    /// Position in sequence
    pub position: usize,
    /// FFN output vector [1, hidden_dim]
    pub output: DenseTensor,
}

impl FFNOutputNode {
    /// Create a new FFN output node
    pub fn new(layer: usize, position: usize, output: DenseTensor) -> Self {
        Self {
            layer,
            position,
            output,
        }
    }

    /// Get the hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.output.shape()[1]
    }
}

/// Graph node wrapper
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node type
    pub node_type: GraphNodeType,
    /// Unique node ID
    pub id: usize,
    /// Layer number (for layer-specific nodes)
    pub layer: usize,
    /// Position in sequence
    pub position: usize,
    /// Optional token embedding data
    pub token_embedding: Option<TokenEmbeddingNode>,
    /// Optional hidden state data
    pub hidden_state: Option<HiddenStateNode>,
    /// Optional attention output data
    pub attention_output: Option<AttentionOutputNode>,
    /// Optional FFN output data
    pub ffn_output: Option<FFNOutputNode>,
}

impl GraphNode {
    /// Create a token embedding node
    pub fn token_embedding(id: usize, token_id: usize, position: usize, embedding: DenseTensor) -> Self {
        Self {
            node_type: GraphNodeType::TokenEmbedding,
            id,
            layer: 0,
            position,
            token_embedding: Some(TokenEmbeddingNode::new(token_id, position, embedding)),
            hidden_state: None,
            attention_output: None,
            ffn_output: None,
        }
    }

    /// Create a hidden state node
    pub fn hidden_state(id: usize, layer: usize, position: usize, state: DenseTensor) -> Self {
        Self {
            node_type: GraphNodeType::HiddenState,
            id,
            layer,
            position,
            token_embedding: None,
            hidden_state: Some(HiddenStateNode::new(layer, position, state)),
            attention_output: None,
            ffn_output: None,
        }
    }

    /// Create an attention output node
    pub fn attention_output(
        id: usize,
        layer: usize,
        head: usize,
        query_pos: usize,
        attended_positions: Vec<usize>,
        weights: Vec<f64>,
        output: DenseTensor,
    ) -> Self {
        Self {
            node_type: GraphNodeType::AttentionOutput,
            id,
            layer,
            position: query_pos,
            token_embedding: None,
            hidden_state: None,
            attention_output: Some(AttentionOutputNode::new(
                layer,
                head,
                query_pos,
                attended_positions,
                weights,
                output,
            )),
            ffn_output: None,
        }
    }

    /// Create a FFN output node
    pub fn ffn_output(id: usize, layer: usize, position: usize, output: DenseTensor) -> Self {
        Self {
            node_type: GraphNodeType::FFNOutput,
            id,
            layer,
            position,
            token_embedding: None,
            hidden_state: None,
            attention_output: None,
            ffn_output: Some(FFNOutputNode::new(layer, position, output)),
        }
    }

    /// Get the embedding if this is a token embedding node
    pub fn get_embedding(&self) -> Option<&TokenEmbeddingNode> {
        self.token_embedding.as_ref()
    }

    /// Get the hidden state if this is a hidden state node
    pub fn get_hidden_state(&self) -> Option<&HiddenStateNode> {
        self.hidden_state.as_ref()
    }

    /// Get the attention output if this is an attention output node
    pub fn get_attention_output(&self) -> Option<&AttentionOutputNode> {
        self.attention_output.as_ref()
    }

    /// Get the FFN output if this is a FFN output node
    pub fn get_ffn_output(&self) -> Option<&FFNOutputNode> {
        self.ffn_output.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_embedding_node() {
        let embedding = DenseTensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4]);
        let node = GraphNode::token_embedding(0, 10, 0, embedding);

        assert_eq!(node.node_type, GraphNodeType::TokenEmbedding);
        assert_eq!(node.id, 0);
        assert_eq!(node.layer, 0);
        assert_eq!(node.position, 0);

        let emb = node.get_embedding().unwrap();
        assert_eq!(emb.token_id, 10);
        assert_eq!(emb.position, 0);
        assert_eq!(emb.hidden_dim(), 4);
    }

    #[test]
    fn test_hidden_state_node() {
        let state = DenseTensor::new(vec![0.1, 0.2, 0.3], vec![1, 3]);
        let node = GraphNode::hidden_state(1, 5, 2, state);

        assert_eq!(node.node_type, GraphNodeType::HiddenState);
        assert_eq!(node.layer, 5);
        assert_eq!(node.position, 2);

        let hidden = node.get_hidden_state().unwrap();
        assert_eq!(hidden.layer, 5);
        assert_eq!(hidden.position, 2);
        assert_eq!(hidden.hidden_dim(), 3);
    }

    #[test]
    fn test_attention_output_node() {
        let output = DenseTensor::new(vec![0.1, 0.2], vec![1, 2]);
        let node = GraphNode::attention_output(
            10,
            3,
            2,
            5,
            vec![3, 4, 5],
            vec![0.3, 0.5, 0.2],
            output,
        );

        assert_eq!(node.node_type, GraphNodeType::AttentionOutput);
        assert_eq!(node.layer, 3);

        let attn = node.get_attention_output().unwrap();
        assert_eq!(attn.layer, 3);
        assert_eq!(attn.head, 2);
        assert_eq!(attn.query_pos, 5);
        assert_eq!(attn.num_attended(), 3);
        assert_eq!(attn.head_dim(), 2);
    }

    #[test]
    fn test_ffn_output_node() {
        let output = DenseTensor::new(vec![0.1, 0.2, 0.3], vec![1, 3]);
        let node = GraphNode::ffn_output(20, 7, 4, output);

        assert_eq!(node.node_type, GraphNodeType::FFNOutput);
        assert_eq!(node.layer, 7);
        assert_eq!(node.position, 4);

        let ffn = node.get_ffn_output().unwrap();
        assert_eq!(ffn.layer, 7);
        assert_eq!(ffn.position, 4);
        assert_eq!(ffn.hidden_dim(), 3);
    }
}
