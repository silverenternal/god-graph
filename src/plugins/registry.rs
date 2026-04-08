//! 插件注册表
//!
//! 管理插件的注册、查找和执行
//!
//! # 架构设计
//!
//! 插件注册表采用类型擦除技术来解决 dyn 兼容性问题：
//! - 使用 `Box<dyn Any + Send + Sync>` 存储插件实例
//! - 通过 `downcast_ref` 进行类型转换
//! - `execute_by_name` 方法支持按名称执行，无需指定类型参数
//!
//! # 示例
//!
//! ```
//! use god_graph::plugins::PluginRegistry;
//!
//! let registry = PluginRegistry::new();
//! assert!(registry.is_empty());
//! ```

use crate::plugins::algorithm::{AlgorithmResult, FastHashMap, GraphAlgorithm, PluginContext, PluginInfo};
use crate::vgi::VgiResult;
use crate::vgi::VirtualGraph;
use std::any::Any;

/// 插件元数据
pub struct PluginMetadata {
    /// 插件信息
    pub info: PluginInfo,
    /// 插件实例（类型擦除）
    pub instance: Box<dyn Any + Send + Sync>,
}

impl PluginMetadata {
    /// 创建新的插件元数据
    pub fn new<A: GraphAlgorithm + 'static>(info: PluginInfo, instance: A) -> Self {
        Self {
            info,
            instance: Box::new(instance),
        }
    }

    /// 尝试获取算法实例的引用
    pub fn as_algorithm<A: GraphAlgorithm + 'static>(&self) -> Option<&A> {
        self.instance.downcast_ref::<A>()
    }
}

/// 插件注册表
///
/// 用于注册、查找和管理图算法插件
///
/// # 示例
///
/// ```
/// use god_graph::plugins::PluginRegistry;
///
/// let registry = PluginRegistry::new();
/// assert!(registry.is_empty());
/// ```
pub struct PluginRegistry {
    /// 已注册的插件元数据
    plugins: FastHashMap<String, PluginMetadata>,
    /// 插件标签索引
    tag_index: FastHashMap<String, Vec<String>>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// 创建新的插件注册表
    pub fn new() -> Self {
        Self {
            plugins: FastHashMap::default(),
            tag_index: FastHashMap::default(),
        }
    }

    /// 注册算法插件
    ///
    /// # Arguments
    ///
    /// * `name` - 插件名称（唯一标识符）
    /// * `algorithm` - 算法实例
    ///
    /// # Returns
    ///
    /// 注册成功返回 Ok，如果名称冲突返回 Err
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let mut registry = PluginRegistry::new();
    /// registry.register_algorithm("pagerank", PageRankPlugin::default());
    /// ```
    pub fn register_algorithm<A: GraphAlgorithm + 'static>(
        &mut self,
        name: impl Into<String>,
        algorithm: A,
    ) -> VgiResult<()> {
        let name = name.into();

        if self.plugins.contains_key(&name) {
            return Err(crate::vgi::VgiError::PluginRegistrationFailed {
                plugin_name: name.clone(),
                reason: "Plugin with this name already exists".to_string(),
            });
        }

        let info = algorithm.info();

        // 更新标签索引
        for tag in &info.tags {
            self.tag_index
                .entry(tag.clone())
                .or_default()
                .push(name.clone());
        }

        let metadata = PluginMetadata::new(info, algorithm);
        self.plugins.insert(name, metadata);

        Ok(())
    }

    /// 按名称执行算法（推荐方法）
    ///
    /// 这个方法不需要指定算法类型参数，内部使用类型擦除技术。
    ///
    /// # Arguments
    ///
    /// * `name` - 算法名称
    /// * `graph` - 要执行算法的图
    /// * `ctx` - 插件上下文
    ///
    /// # Returns
    ///
    /// 算法执行结果
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let mut registry = PluginRegistry::new();
    /// registry.register_algorithm("pagerank", PageRankPlugin::default());
    ///
    /// let mut ctx = PluginContext::new(&graph);
    /// let result = registry.execute_by_name("pagerank", &graph, &mut ctx)?;
    /// ```
    pub fn execute_by_name<G: VirtualGraph + ?Sized>(
        &self,
        name: &str,
        _graph: &G,
        ctx: &mut PluginContext<'_, G>,
    ) -> VgiResult<AlgorithmResult> {
        let metadata =
            self.plugins
                .get(name)
                .ok_or_else(|| crate::vgi::VgiError::PluginNotFound {
                    plugin_name: name.to_string(),
                })?;

        // 使用宏来尝试 downcast 到所有可能的算法类型并执行
        // 这是 Rust 中处理 trait object 调用泛型方法的标准模式
        macro_rules! try_downcast_and_execute {
            ($($t:ty),*) => {
                $(
                    if let Some(algo) = metadata.instance.downcast_ref::<$t>() {
                        algo.validate(ctx)?;
                        algo.before_execute(ctx)?;
                        let result = algo.execute(ctx)?;
                        algo.after_execute(ctx, &result)?;
                        return Ok(result);
                    }
                )*
            };
        }

        // 尝试 downcast 到所有已知的算法类型
        // 注意：这需要手动维护列表，但这是 Rust 类型系统的限制
        try_downcast_and_execute!(
            crate::plugins::PageRankPlugin,
            crate::plugins::BfsPlugin,
            crate::plugins::DfsPlugin,
            crate::plugins::ConnectedComponentsPlugin,
            crate::plugins::DijkstraPlugin,
            crate::plugins::BellmanFordPlugin,
            crate::plugins::BetweennessCentralityPlugin,
            crate::plugins::ClosenessCentralityPlugin,
            crate::plugins::LouvainPlugin,
            crate::plugins::TopologicalSortPlugin
        );

        // 如果都不匹配，返回错误
        Err(crate::vgi::VgiError::Internal {
            message: format!(
                "Failed to downcast plugin '{}' to any known algorithm type",
                name
            ),
        })
    }

    /// 执行算法（泛型版本，已弃用）
    ///
    /// # Type Parameters
    ///
    /// * `G` - 图类型
    /// * `A` - 算法类型（必须与注册时的类型匹配）
    ///
    /// # Arguments
    ///
    /// * `name` - 算法名称
    /// * `graph` - 要执行算法的图
    /// * `ctx` - 插件上下文
    ///
    /// # Returns
    ///
    /// 算法执行结果
    #[deprecated(
        since = "0.6.1",
        note = "Use `execute_by_name` instead for simpler API"
    )]
    pub fn execute<G, A>(
        &self,
        name: &str,
        graph: &G,
        ctx: &mut PluginContext<'_, G>,
    ) -> VgiResult<AlgorithmResult>
    where
        G: VirtualGraph + ?Sized,
        A: GraphAlgorithm + 'static,
    {
        // 委托给 execute_by_name
        self.execute_by_name(name, graph, ctx)
    }

    /// 获取插件元数据
    pub fn get_metadata(&self, name: &str) -> Option<&PluginMetadata> {
        self.plugins.get(name)
    }

    /// 检查插件是否已注册
    pub fn is_registered(&self, name: &str) -> bool {
        self.plugins.contains_key(name)
    }

    /// 列出所有已注册的插件名称
    pub fn list_plugins(&self) -> Vec<&String> {
        self.plugins.keys().collect()
    }

    /// 按标签查找插件
    pub fn find_by_tag(&self, tag: &str) -> Vec<&String> {
        self.tag_index
            .get(tag)
            .map(|names| names.iter().collect())
            .unwrap_or_default()
    }

    /// 获取插件数量
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// 清空注册表
    pub fn clear(&mut self) {
        self.plugins.clear();
        self.tag_index.clear();
    }

    /// 获取算法信息
    pub fn get_algorithm_info(&self, name: &str) -> Option<&PluginInfo> {
        self.plugins.get(name).map(|m| &m.info)
    }

    /// 注销插件
    pub fn unregister(&mut self, name: &str) -> VgiResult<Option<PluginMetadata>> {
        if let Some(metadata) = self.plugins.remove(name) {
            // 清理标签索引
            for tag in &metadata.info.tags {
                if let Some(names) = self.tag_index.get_mut(tag) {
                    names.retain(|n| n != name);
                }
            }
            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }
}

/// 插件构建器
///
/// 用于链式构建和注册插件元数据
pub struct PluginMetadataBuilder {
    name: String,
    info: Option<PluginInfo>,
}

impl PluginMetadataBuilder {
    /// 创建新的构建器
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            info: None,
        }
    }

    /// 设置插件信息
    pub fn info(mut self, info: PluginInfo) -> Self {
        self.info = Some(info);
        self
    }

    /// 注册到注册表
    pub fn register<A: GraphAlgorithm + 'static>(
        self,
        registry: &mut PluginRegistry,
        algorithm: A,
    ) -> VgiResult<()> {
        let info = self
            .info
            .ok_or_else(|| crate::vgi::VgiError::PluginRegistrationFailed {
                plugin_name: self.name.clone(),
                reason: "No plugin info provided".to_string(),
            })?;

        let name = self.name.clone();
        registry.register_algorithm(name.clone(), algorithm)?;

        // 更新已注册插件的 info（如果算法实例的 info 不同）
        if let Some(metadata) = registry.plugins.get_mut(&name) {
            metadata.info = info;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::graph::traits::GraphOps;
    use crate::plugins::algorithm::{AlgorithmResult, PluginInfo};
    use crate::vgi::VirtualGraph;
    use std::any::Any;

    struct TestAlgorithm;

    impl GraphAlgorithm for TestAlgorithm {
        fn info(&self) -> PluginInfo {
            PluginInfo::new("test", "1.0.0", "Test Algorithm").with_tags(&["test", "demo"])
        }

        fn execute<G>(&self, _ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
        where
            G: VirtualGraph + ?Sized,
        {
            Ok(AlgorithmResult::scalar(42.0))
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn test_plugin_registry() {
        let mut registry = PluginRegistry::new();

        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        registry.register_algorithm("test", TestAlgorithm).unwrap();

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.is_registered("test"));
        assert!(registry.get_metadata("test").is_some());
    }

    #[test]
    fn test_plugin_find_by_tag() {
        let mut registry = PluginRegistry::new();

        struct TaggedAlgorithm {
            tags: Vec<&'static str>,
        }
        impl GraphAlgorithm for TaggedAlgorithm {
            fn info(&self) -> PluginInfo {
                PluginInfo::new("tagged", "1.0.0", "Tagged Algorithm")
                    .with_tags(&self.tags.to_vec())
            }
            fn execute<G>(&self, _ctx: &mut PluginContext<G>) -> VgiResult<AlgorithmResult>
            where
                G: VirtualGraph + ?Sized,
            {
                Ok(AlgorithmResult::scalar(1.0))
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }

        registry
            .register_algorithm(
                "algo1",
                TaggedAlgorithm {
                    tags: vec!["tag1", "common"],
                },
            )
            .unwrap();
        registry
            .register_algorithm(
                "algo2",
                TaggedAlgorithm {
                    tags: vec!["tag2", "common"],
                },
            )
            .unwrap();

        // 单标签查找
        assert_eq!(registry.find_by_tag("tag1").len(), 1);
        assert_eq!(registry.find_by_tag("common").len(), 2);

        // 不存在的标签
        assert_eq!(registry.find_by_tag("nonexistent").len(), 0);
    }

    #[test]
    fn test_plugin_execute() {
        use crate::plugins::PageRankPlugin;

        let mut registry = PluginRegistry::new();
        // 使用内置的 PageRankPlugin 测试
        registry
            .register_algorithm("pagerank", PageRankPlugin::default())
            .unwrap();

        let graph = Graph::<String, f64>::directed();
        let mut ctx = PluginContext::new(&graph);

        // 使用新的 execute_by_name 方法（不需要类型参数）
        let result = registry.execute_by_name("pagerank", &graph, &mut ctx);
        assert!(result.is_ok());
        let result = result.unwrap();
        // PageRank 返回的是 NodeValues
        assert!(result.data.as_node_values().is_some());
    }

    #[test]
    fn test_plugin_unregister() {
        let mut registry = PluginRegistry::new();
        registry.register_algorithm("test", TestAlgorithm).unwrap();
        assert!(registry.is_registered("test"));

        registry.unregister("test").unwrap();
        assert!(!registry.is_registered("test"));
    }
}
