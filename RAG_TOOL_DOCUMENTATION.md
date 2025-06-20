# Enhanced RAG Tool Documentation

## 概述

增强的RAG（Retrieval Augmented Generation）工具现在支持：
- 🗄️ **多数据库注册和管理**
- 🔍 **通用文本数据查询**（不仅限于相似issue匹配）
- 🤖 **LLM智能分析和使用建议**（借鉴Similar Issues Tool的模式）
- ⚙️ **灵活的知识检索配置**
- 🎯 **跨数据库搜索功能**

## 主要功能

### 1. 数据库注册管理

```python
from analyzer_core.tools.rag_tool import RAGTool

tool = RAGTool()

# 注册新数据库
tool.register_database(
    name="custom_db",
    path="/path/to/database",
    description="自定义知识库",
    collection_name="documents",
    set_as_default=False
)

# 列出所有数据库
databases = tool.list_databases()
print(f"已注册数据库: {list(databases.keys())}")

# 设置默认数据库
tool.set_default_database("custom_db")
```

### 2. 基本查询（纯检索模式）

```python
# 纯检索，不生成使用建议
query_data = {
    "query_text": "Office add-in loading problem",
    "n_results": 5,
    "generate_suggestions": False  # 关闭LLM分析
}

result = await tool.execute(query_data)
print(f"找到 {result['results_count']} 个相关结果")
```

### 3. 智能分析查询（包含LLM建议）

```python
# 包含LLM分析和使用建议
query_data = {
    "query_text": "Office add-in loading problem",
    "n_results": 5,
    "generate_suggestions": True  # 启用LLM分析
}

result = await tool.execute(query_data)
print(f"找到 {result['results_count']} 个相关结果")

# 获取智能建议
if 'use_suggestion' in result:
    suggestion = result['use_suggestion']
    print(f"相关性: {suggestion['relevance']}")
    print(f"建议摘要: {suggestion['summary']}")
    for insight in suggestion['actionable_insights']:
        print(f"- {insight}")
```

### 4. 指定数据库查询

```python
# 查询特定数据库
query_data = {
    "query_text": "Excel API usage",
    "database_name": "api_docs",
    "n_results": 3,
    "generate_suggestions": True  # 可选：生成使用建议
}

result = await tool.execute(query_data)
```

### 5. 跨数据库搜索

```python
# 在所有数据库中搜索并生成综合分析
query_data = {
    "query_text": "troubleshooting guide",
    "search_all": True,
    "n_results": 5,
    "exclude_databases": ["test_db"],  # 排除某些数据库
    "generate_suggestions": True  # 生成跨数据库综合建议
}

result = await tool.execute(query_data)
print(f"搜索了 {len(result['databases_searched'])} 个数据库")
print(f"总共找到 {result['total_results']} 个结果")

# 获取跨数据库综合分析
if 'use_suggestion' in result:
    suggestion = result['use_suggestion']
    print(f"综合建议: {suggestion['recommended_approach']}")
```

### 6. 获取数据库信息

```python
# 获取所有数据库信息
all_info = tool.get_database_info()
print(f"默认数据库: {all_info['default_database']}")
print(f"总数据库数: {all_info['total_databases']}")

# 获取特定数据库信息
db_info = tool.get_database_info("main")
print(f"文档数量: {db_info['document_count']}")
print(f"状态: {db_info['status']}")
```

## 配置文件

工具会自动创建 `rag_databases.json` 配置文件来保存数据库注册信息：

```json
{
  "databases": {
    "main": {
      "path": "./chroma_db",
      "description": "Main issue database",
      "collection_name": "issue_collection",
      "registered_at": "1640995200.0"
    },
    "custom_db": {
      "path": "./custom_chroma_db",
      "description": "Custom knowledge base",
      "collection_name": "documents",
      "registered_at": "1640995300.0"
    }
  },
  "default_database": "main"
}
```

## 返回格式

### 单数据库查询结果（包含LLM分析）
```json
{
  "query": "查询文本",
  "database": "数据库名称",
  "results_count": 3,
  "confidence": 0.7,
  "is_successful": true,
  "results": [
    {
      "content": "文档内容...",
      "similarity_score": 0.85,
      "metadata": {"source": "document1.txt"}
    }
  ],
  "use_suggestion": {
    "summary": "找到了3个相关的技术文档，主要涉及API使用和故障排除",
    "relevance": "high",
    "actionable_insights": [
      "文档显示此问题通常由配置错误引起，建议检查配置文件",
      "参考文档2中的解决方案，已被验证有效",
      "注意文档3中提到的已知限制和解决方法"
    ],
    "recommended_approach": "基于检索到的文档，建议按以下步骤排查：1) 检查配置文件格式 2) 验证API调用参数 3) 查看日志文件",
    "user_friendly_summary": "找到了相关的技术文档，主要说明了如何解决类似的配置和API调用问题"
  }
}
```

### 纯检索结果（无LLM分析）
```json
{
  "query": "查询文本",
  "database": "数据库名称",
  "results_count": 3,
  "confidence": 0.7,
  "is_successful": true,
  "results": [
    {
      "content": "文档内容...",
      "similarity_score": 0.85,
      "metadata": {"source": "document1.txt"}
    }
  ]
}
```

### 跨数据库搜索结果（包含综合分析）
```json
{
  "query": "查询文本",
  "total_results": 5,
  "confidence": 0.8,
  "databases_searched": ["main", "custom_db"],
  "combined_results": [
    {
      "content": "文档内容...",
      "similarity_score": 0.90,
      "source_database": "main",
      "metadata": {}
    }
  ],
  "database_breakdown": {
    "main": {"results_count": 3, "is_successful": true},
    "custom_db": {"results_count": 2, "is_successful": true}
  },
  "use_suggestion": {
    "summary": "跨多个数据库找到了5个相关结果，信息来源多样化",
    "relevance": "high",
    "actionable_insights": [
      "主数据库提供了核心技术文档，包含详细的解决步骤",
      "自定义数据库补充了实际案例和最佳实践",
      "两个数据源的信息相互印证，提高了解决方案的可靠性"
    ],
    "recommended_approach": "结合多个数据源的信息，建议采用综合解决方案：先参考主数据库的标准流程，再结合自定义数据库的实践经验",
    "user_friendly_summary": "从多个知识库中找到了相关信息，提供了更全面的解决思路"
  }
}
```

## 与 Similar Issues Tool 的区别

| 功能 | RAG Tool | Similar Issues Tool |
|------|----------|-------------------|
| 用途 | 通用文本数据查询 | 专门的相似issue匹配 |
| 数据库支持 | 多数据库注册管理 | 固定数据库 |
| 查询类型 | 任意文本查询 | Issue标题和内容 |
| 输出格式 | 灵活的结果格式 | 特定的issue分析格式 |
| LLM分析 | 可选的智能分析和建议 | 固定的issue分析和GitHub链接生成 |
| 使用模式 | 纯检索 + 可选智能分析 | 固定的LLM分析流程 |

## RAG Tool的两种工作模式

### 📊 纯检索模式 (`generate_suggestions=False`)
- 快速返回相关文档
- 不消耗LLM资源
- 适合需要原始数据的场景
- 返回格式简洁

### 🤖 智能分析模式 (`generate_suggestions=True`)
- 对检索结果进行LLM分析
- 生成使用建议和洞察
- 提供用户友好的总结
- 适合需要指导的场景

## 使用场景

### 🔍 纯检索场景
1. **快速查找**: 需要快速获取相关文档片段
2. **数据提取**: 为其他系统提供原始数据
3. **批量处理**: 处理大量查询时节省LLM成本
4. **API集成**: 作为其他服务的数据源

### 🤖 智能分析场景  
1. **决策支持**: 需要基于知识库的建议和指导
2. **问题解决**: 用户需要结构化的解决方案
3. **知识发现**: 发现数据间的关联和模式
4. **用户交互**: 为最终用户提供友好的解释

### 🗄️ 多数据库场景
1. **知识整合**: 从多个专业领域获取信息
2. **信息验证**: 通过多源对比验证信息准确性
3. **全面分析**: 获得更完整的视角和解决方案
4. **领域切换**: 根据查询内容自动选择最相关的数据库

## 最佳实践

### 🗄️ 数据库管理
1. **数据库命名**: 使用描述性的数据库名称
2. **集合管理**: 为不同类型的文档使用不同的集合
3. **定期维护**: 定期检查数据库状态和文档数量

### 🔍 查询优化
1. **模式选择**: 根据需求选择纯检索或智能分析模式
2. **结果数量**: 根据用途调整返回结果数量
3. **相似度阈值**: 可以根据相似度分数过滤结果
4. **查询精度**: 使用具体的关键词提高查询精度

### 🤖 LLM使用
1. **成本控制**: 在不需要分析时关闭LLM功能
2. **prompt优化**: 根据业务需求调整LLM分析的prompt
3. **结果验证**: 对LLM生成的建议进行人工验证
4. **缓存策略**: 对常见查询缓存LLM分析结果

### 🎯 性能优化
1. **数据库分区**: 将不同类型的数据存储在不同数据库中
2. **索引优化**: 合理设置ChromaDB的索引参数
3. **批量查询**: 对于多个相关查询，考虑使用批量处理
4. **资源监控**: 监控数据库和LLM的资源使用情况
