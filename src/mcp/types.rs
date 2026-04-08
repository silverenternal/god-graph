//! MCP type definitions
//!
//! This module defines the core types used in the MCP protocol.

use serde::{Deserialize, Serialize};

/// MCP protocol version
pub const MCP_VERSION: &str = "1.0";

/// JSON-RPC version
pub const JSON_RPC_VERSION: &str = "2.0";

/// MCP Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
    /// Protocol version
    #[serde(default = "default_protocol_version")]
    pub protocol_version: String,
}

fn default_protocol_version() -> String {
    MCP_VERSION.to_string()
}

impl Default for ServerInfo {
    fn default() -> Self {
        Self {
            name: "god-graph-mcp".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_version: MCP_VERSION.to_string(),
        }
    }
}

/// Tool input schema type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JsonSchemaType {
    /// Object type (dictionary/map)
    Object,
    /// Array type (list)
    Array,
    /// String type
    String,
    /// Number type (floating point)
    Number,
    /// Integer type (whole number)
    Integer,
    /// Boolean type (true/false)
    Boolean,
    /// Null type
    Null,
}

/// JSON Schema for tool parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// The type of the schema
    #[serde(rename = "type")]
    pub schema_type: JsonSchemaType,
    /// Description of the field
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Properties for object types
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<std::collections::HashMap<String, Box<JsonSchema>>>,
    /// Required property names for object types
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    /// Schema for array items
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JsonSchema>>,
    /// Default value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    /// Minimum value for numbers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Maximum value for numbers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
}

impl JsonSchema {
    /// Create a string schema
    pub fn string() -> Self {
        Self {
            schema_type: JsonSchemaType::String,
            description: None,
            properties: None,
            required: None,
            items: None,
            default: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a number schema
    pub fn number() -> Self {
        Self {
            schema_type: JsonSchemaType::Number,
            description: None,
            properties: None,
            required: None,
            items: None,
            default: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create an integer schema
    pub fn integer() -> Self {
        Self {
            schema_type: JsonSchemaType::Integer,
            description: None,
            properties: None,
            required: None,
            items: None,
            default: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a boolean schema
    pub fn boolean() -> Self {
        Self {
            schema_type: JsonSchemaType::Boolean,
            description: None,
            properties: None,
            required: None,
            items: None,
            default: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create an object schema
    pub fn object() -> Self {
        Self {
            schema_type: JsonSchemaType::Object,
            description: None,
            properties: Some(std::collections::HashMap::new()),
            required: None,
            items: None,
            default: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Add a description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a property to an object schema
    pub fn with_property(mut self, name: impl Into<String>, schema: JsonSchema) -> Self {
        match self.properties {
            Some(ref mut props) => {
                props.insert(name.into(), Box::new(schema));
            }
            None => {
                let mut props = std::collections::HashMap::new();
                props.insert(name.into(), Box::new(schema));
                self.properties = Some(props);
            }
        }
        self
    }

    /// Mark properties as required
    pub fn with_required(mut self, names: Vec<&str>) -> Self {
        self.required = Some(names.into_iter().map(String::from).collect());
        self
    }

    /// Set the default value
    pub fn with_default(mut self, value: serde_json::Value) -> Self {
        self.default = Some(value);
        self
    }
}

/// Tool definition for MCP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool name (snake_case, no spaces)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Input schema (JSON Schema)
    pub input_schema: JsonSchema,
}

impl Tool {
    /// Create a new tool definition
    pub fn new(name: impl Into<String>, description: impl Into<String>, input_schema: JsonSchema) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Result content (text, image, or resource)
    pub content: Vec<Content>,
    /// Whether the result is an error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ToolResult {
    /// Create a text result
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![Content::text(text)],
            is_error: None,
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![Content::text(message)],
            is_error: Some(true),
        }
    }
}

/// Content type for tool results
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Content {
    /// Text content
    Text {
        /// Text content string
        text: String,
    },
    /// Image content (base64 encoded)
    Image {
        /// Base64 encoded image data
        data: String,
        /// MIME type of the image
        mime_type: String,
    },
    /// Resource content
    Resource {
        /// Resource reference
        resource: Resource,
    },
}

impl Content {
    /// Create a text content
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create an image content (base64 encoded)
    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Image {
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }
}

/// Resource definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// Resource URI
    pub uri: String,
    /// Resource name
    pub name: String,
    /// Resource content
    #[serde(flatten)]
    pub content: ResourceContent,
}

/// Resource content type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResourceContent {
    /// Text resource
    Text {
        /// Text content string
        text: String,
    },
    /// Blob resource (base64 encoded)
    Blob {
        /// Base64 encoded blob data
        blob: String,
        /// MIME type of the blob
        mime_type: String,
    },
}

/// JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,
    /// Request ID
    pub id: Option<serde_json::Value>,
    /// Method name
    pub method: String,
    /// Method parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,
    /// Response ID (matches request ID)
    pub id: Option<serde_json::Value>,
    /// Response result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Error information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// MCP capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    /// Tools capability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    /// Resources capability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    /// Prompts capability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
}

/// Tools capability for listing available tools
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    /// Whether the tool list has changed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Resources capability for resource management
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesCapability {
    /// Whether subscription is supported
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subscribe: Option<bool>,
    /// Whether the resource list has changed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Prompts capability for prompt templates
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsCapability {
    /// Whether the prompt list has changed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}
