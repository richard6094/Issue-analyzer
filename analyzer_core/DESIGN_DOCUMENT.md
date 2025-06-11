# GitHub Issue Agent Architecture Design Document (GitHub Actions Based)

## 1. 概述

GitHub Issue Agent 是一个基于GitHub Actions的智能问题处理系统，通过Issue评论进行交互，实现对新创建的GitHub Issues和用户回复的自主分析、分类、引导和解决方案推荐。

### 1.1 设计目标

- **GitHub Actions集成**：完全基于GitHub Actions工作流运行
- **评论交互**：通过Issue评论与用户进行智能对话
- **自动触发**：响应Issue创建、评论添加等GitHub事件
- **状态持续**：通过评论元数据维护对话状态
- **现有功能复用**：扩展现有的analyze_issue.py功能

### 1.2 核心原则

- **事件驱动**：基于GitHub webhooks和Actions触发
- **无状态设计**：每次执行都是独立的，状态通过评论保存
- **渐进式处理**：从简单分类到复杂解决方案的分步处理
- **用户友好**：清晰的评论格式和交互提示
- **智能交互控制**：基于用户角色和交互上下文的智能回复机制

## 2. 交互逻辑与触发机制

### 2.1 触发条件详细设计

#### 2.1.1 自动触发条件（默认启用）

**Issue创建时 (issues.opened)**
- **触发对象**: 所有新创建的Issue
- **行为**: 自动进行初始分析并发布分析评论
- **例外**: 如果Issue描述中包含 `@issue-agent disable` 则不触发初始分析

**Issue Owner评论时 (issue_comment.created)**
- **触发对象**: Issue的创建者(owner)的所有评论
- **触发条件**: 
  - 评论者必须是Issue的创建者
  - Issue未被禁用Agent交互
  - 评论者不是Bot账户
- **行为**: 自动分析评论内容并进行回复
- **特殊处理**: 
  - 检测禁用/启用指令：`@issue-agent disable` 或 `@issue-agent enable`
  - 检测帮助指令：`@issue-agent help`
  - 分析评论意图：提问、反馈、补充信息等

#### 2.1.2 条件触发（被动响应）

**其他用户引用Agent时 (issue_comment.created)**
- **触发条件**: 
  - 评论者不是Issue Owner
  - 评论者不是Bot账户
  - 满足以下任一引用条件：
    - 直接mention: `@issue-agent` 或 `@github-actions[bot]`
    - 回复Agent评论: 使用GitHub的reply功能
    - 引用Agent评论: 使用 `>` 语法引用Agent之前的回复
    - 评论中包含指向Agent评论的链接
- **行为**: 
  - 分析引用内容和用户意图
  - 提供针对性回复（权限受限）
  - 不主动提供敏感操作建议
  - 必要时引导用户向Issue Owner寻求帮助

#### 2.1.3 严格排除条件

**Bot评论完全排除**
- **GitHub Actions Bot**: 所有 `github-actions[bot]` 的评论
- **其他CI/CD Bot**: 如 `codecov[bot]`, `dependabot[bot]` 等
- **第三方Bot**: 所有用户名以 `[bot]` 结尾的账户
- **自身评论**: Agent生成的评论不会触发新的响应

**系统保护机制**
- **循环检测**: 防止与其他Bot形成无限循环
- **频率限制**: 同一用户在短时间内的多次评论合并处理
- **异常状态**: 检测到异常交互模式时自动禁用

### 2.2 用户权限分级系统

#### 2.2.1 权限等级定义

**Level 1: Issue Owner (完全控制权)**
```
权限范围:
├── 默认启用所有自动交互
├── 可以禁用/启用Agent (`@issue-agent disable/enable`)
├── 所有评论都触发Agent回复
├── 可以请求所有类型的分析和建议
├── 可以要求Agent执行自动化操作（标签、关联等）
└── 接收完整的分析报告和解决方案

交互特点:
├── 回复速度: 立即响应
├── 回复详细度: 完整详细
├── 操作权限: 无限制
└── 建议类型: 包括敏感操作
```

**Level 2: Repository Collaborators (协作权限)**
```
权限范围:
├── 仅在明确引用时回复
├── 可以获得技术建议和分析
├── 不能控制Agent的启用/禁用状态
├── 可以提供补充信息协助分析
└── 接收适度详细的技术回复

交互特点:
├── 回复速度: 正常响应
├── 回复详细度: 适中
├── 操作权限: 受限
└── 建议类型: 技术性建议为主
```

**Level 3: External Contributors (受限权限)**
```
权限范围:
├── 仅在明确引用且合理的情况下回复
├── 主要提供公开信息和通用建议
├── 不提供项目特定的敏感信息
├── 引导向Issue Owner或维护者寻求帮助
└── 接收简化的通用性回复

交互特点:
├── 回复速度: 延迟响应（防止滥用）
├── 回复详细度: 简化
├── 操作权限: 严格受限
└── 建议类型: 通用性建议
```

#### 2.2.2 权限检查实现

**身份验证流程**
```python
def check_user_permission_level(issue_data, comment_data):
    """
    检查用户权限等级
    返回: 'owner', 'collaborator', 'contributor', 'bot', 'blocked'
    """
    commenter = comment_data['user']
    issue_owner = issue_data['user']
    
    # 1. Bot检查 - 最高优先级
    if commenter['login'].endswith('[bot]'):
        return 'bot'
    
    # 2. Issue Owner检查
    if commenter['id'] == issue_owner['id']:
        return 'owner'
    
    # 3. 协作者权限检查
    if is_repository_collaborator(commenter['login']):
        return 'collaborator'
    
    # 4. 默认为外部贡献者
    return 'contributor'
```

### 2.3 交互控制指令系统

#### 2.3.1 Owner专用控制指令

**禁用Agent指令**
- **语法**: `@issue-agent disable` 或 `@issue-agent off`
- **权限**: 仅Issue Owner
- **作用域**: 当前Issue
- **效果**: 
  - 停止对Owner评论的自动回复
  - 保留被引用时的回复功能
  - 在Issue中添加禁用状态标识
  - 记录禁用时间和原因

**启用Agent指令**
- **语法**: `@issue-agent enable` 或 `@issue-agent on`  
- **权限**: 仅Issue Owner
- **作用域**: 当前Issue
- **效果**: 
  - 恢复对Owner评论的自动回复
  - 清除禁用状态标识
  - 发送重新启用确认消息

**临时静默指令**
- **语法**: `@issue-agent silent` 或 `@issue-agent quiet`
- **权限**: 仅Issue Owner
- **效果**: 
  - 临时禁用自动回复（24小时）
  - 保留紧急情况下的回复能力
  - 自动恢复正常模式

#### 2.3.2 通用指令

**帮助指令**
- **语法**: `@issue-agent help`
- **权限**: 所有用户
- **效果**: 根据用户权限显示可用指令和功能

**状态查询指令**
- **语法**: `@issue-agent status`
- **权限**: 所有用户  
- **效果**: 显示当前Issue的Agent交互状态

### 2.4 直接引用检测系统

#### 2.4.1 引用检测逻辑

**直接引用检测 (唯一检测方式)**
```regex
# @mention检测
@issue-agent|@github-actions\[bot\]

# 直接回复检测  
检查comment.in_reply_to_id是否指向Agent评论
```

**引用验证流程**
```python
def is_referencing_agent(comment_data, issue_comments):
    """
    检查评论是否直接引用了Agent
    返回: True/False
    """
    comment_body = comment_data['body'].lower()
    
    # 1. @issue-agent检测
    if '@issue-agent' in comment_body or '@github-actions[bot]' in comment_body:
        return True
    
    # 2. 直接回复检测
    if comment_data.get('in_reply_to_id'):
        replied_comment = find_comment_by_id(issue_comments, comment_data['in_reply_to_id'])
        if replied_comment and is_agent_comment(replied_comment):
            return True
    
    return False
```

### 2.5 状态持久化与会话管理

#### 2.5.1 禁用状态管理

**状态数据结构**
```json
{
  "interaction_settings": {
    "auto_reply_enabled": true,
    "disabled_by": null,
    "disabled_at": null,
    "disable_reason": null,
    "silent_until": null,
    "last_interaction": "2025-05-29T10:30:00Z"
  },
  "user_preferences": {
    "notification_level": "normal",
    "response_detail": "full",
    "auto_labeling": true
  }
}
```

**状态检查优先级**
1. **全局禁用检查** - Issue级别的禁用设置
2. **临时静默检查** - 时间限制的静默状态  
3. **用户权限检查** - 基于角色的权限验证
4. **系统保护检查** - 防止循环和滥用
5. **默认规则应用** - 标准交互逻辑

#### 2.5.2 状态存储实现

**状态嵌入方式**
```html
<!-- 用户可见的回复内容 -->

<!-- ISSUE_AGENT_STATE:START
{
  "conversation_state": {...},
  "interaction_settings": {...},
  "analysis_history": [...]
}
ISSUE_AGENT_STATE:END -->
```

**状态提取与更新**
```python
def extract_latest_state(issue_comments):
    """从最新的Agent评论中提取状态"""
    for comment in reversed(issue_comments):
        if is_agent_comment(comment):
            return parse_embedded_state(comment['body'])
    return create_initial_state()

def update_conversation_state(current_state, user_comment, analysis_result):
    """更新对话状态"""
    new_state = current_state.copy()
    new_state['last_user_input'] = user_comment
    new_state['analysis_results'].append(analysis_result)
    new_state['conversation_stage'] = determine_next_stage(analysis_result)
    return new_state
```

### 2.6 防护机制与异常处理

#### 2.6.1 循环检测与防护

**Bot循环检测**
- 检测连续的Bot交互模式
- 识别潜在的无限循环风险
- 自动暂停交互并通知管理员

**频率限制**
- 单用户短时间内评论合并处理
- 异常高频交互自动限制
- 恶意使用检测与阻止

#### 2.6.2 错误恢复机制

**状态恢复**
- 状态损坏时的自动重建
- 历史对话上下文的智能恢复
- 优雅降级到基础功能

**交互异常处理**
- API调用失败的重试机制
- 分析服务不可用时的备用方案
- 用户反馈的错误信息收集

## 3. 工作流程设计

### 3.1 Issue创建流程

```
新Issue创建
    ↓
GitHub Actions触发 (issue-created.yml)
    ↓
获取Issue内容和上下文
    ↓
执行综合分析 (复用现有analyze_issue逻辑)
    ├── 回归分析检查
    ├── 严重程度评估
    ├── 问题类型分类
    └── 信息完整性检查
    ↓
生成初始分析评论
    ↓
嵌入对话状态到评论中
    ↓
发布评论到Issue
    ↓
执行立即行动 (标签、通知等)
```

### 3.2 评论交互流程和触发逻辑

```
用户添加评论
    ↓
GitHub Actions触发 (comment-added.yml)
    ↓
检查评论触发条件
    ├── Issue Owner评论：默认触发回复
    ├── 其他用户Quote Agent：检测引用后触发回复
    ├── Bot评论：跳过处理
    └── Owner设置不介入：跳过处理
    ↓
从历史评论中提取对话状态
    ↓
LLM分析用户回复内容
    ├── 意图识别分析
    ├── 信息完整性评估
    ├── 情感倾向分析
    └── 下一步行动建议
    ↓
根据LLM分析结果决定响应策略
    ├── 继续信息收集
    ├── 提供解决方案
    ├── 升级处理
    └── 结束对话
    ↓
生成响应评论
    ↓
更新对话状态
    ↓
发布响应评论
```

### 3.3 交互触发逻辑

#### 3.3.1 触发条件优先级

1. **最高优先级：直接引用检测**
   - 检测其他用户是否quote了Agent的回复
   - 通过GitHub API的comment引用关系检测
   - 触发条件：非Issue Owner用户引用Agent评论

2. **默认触发：Issue Owner回复**
   - Issue Owner的任何评论都会触发Agent回复
   - 前提：Owner未设置"不介入"选项

3. **跳过条件：Bot和不相关评论**
   - 所有Bot账户的评论（包括GitHub Actions）
   - Issue Owner设置不希望Agent介入的情况

#### 3.3.2 Owner控制机制

- **介入控制选项**：在Agent首次回复中提供控制选项
- **退出指令**：Owner可通过特定回复请求Agent停止介入
- **重新激活**：Owner可通过特定回复重新激活Agent

### 3.4 LLM驱动的意图识别

#### 3.4.1 意图分析提示策略

- **上下文感知**：将完整的Issue历史和当前对话状态提供给LLM
- **意图分类**：让LLM识别用户回复的主要意图类型
- **置信度评估**：LLM提供对意图识别的置信度评分
- **行动建议**：LLM基于意图分析建议下一步行动

#### 3.4.2 意图类型框架

通过LLM识别的主要意图类型：
- **信息提供**：用户提供了请求的额外信息
- **问题澄清**：用户对问题描述进行了澄清或修正
- **解决方案反馈**：用户对提供的解决方案进行反馈
- **寻求帮助**：用户明确表示需要进一步帮助
- **问题解决确认**：用户确认问题已经解决
- **不满或投诉**：用户表达不满情绪
- **退出请求**：用户希望结束与Agent的交互

#### 3.4.3 LLM分析输出格式

LLM需要输出结构化的分析结果：
- **主要意图**：识别的主要意图类型
- **次要意图**：可能存在的次要意图
- **情感倾向**：积极、中性、消极
- **信息完整性**：评估用户是否提供了足够信息
- **建议行动**：基于分析建议的下一步行动
- **置信度**：对整体分析的置信度评分

### 3.5 智能功能调度系统

#### 3.5.1 功能调度模块设计

功能调度模块是一个智能评估层，它使LLM能够在收到用户新评论后，自动判断是否需要调用特定功能（如RAG、图像分析等）来更好地帮助用户，而不是在每次交互中都调用所有可用功能。

**核心组件**
```
功能调度系统:
├── 意图分析器: 分析用户评论意图
├── 功能注册表: 维护可用功能清单
├── 调度决策引擎: 根据意图选择功能
├── 功能执行管理: 按优先级执行选定功能
└── 结果整合器: 合并多个功能的结果
```

**工作流程**
```
用户提交评论
    ↓
LLM分析用户意图和需求
    ↓
调度决策引擎评估功能需求
    ├── 是否需要RAG检索知识？
    ├── 是否需要图像分析？
    ├── 是否需要代码分析？
    ├── 是否需要问题诊断？
    └── 是否只需一般对话？
    ↓
按需调用所需功能
    ↓
整合功能结果
    ↓
生成最终回复
```

#### 3.5.2 功能调度决策机制

**调度决策标准**
- **意图相关性**: 用户意图与功能目的的匹配度
- **上下文需求**: 当前对话上下文是否需要特定功能
- **置信度阈值**: 只有当需求置信度超过阈值时才调用功能
- **资源优化**: 避免不必要的功能调用以节省资源

**动态优先级规则**
```
优先级确定:
├── 高优先级: 用户明确要求的功能
├── 中优先级: 高置信度推断需要的功能
├── 低优先级: 可能有帮助但非必需的功能
└── 不调用: 置信度低于阈值的功能
```

#### 3.5.3 可插拔功能设计

**功能注册机制**
```json
{
  "功能ID": {
    "name": "知识库检索",
    "description": "从知识库检索相关内容",
    "invoke_condition": {
      "primary_intents": ["search_knowledge", "issue_analysis"],
      "keywords": ["查找", "搜索", "类似问题"],
      "confidence_threshold": 0.65
    },
    "resource_requirement": "medium",
    "output_format": "text",
    "priority_level": "high"
  }
}
```

**易扩展架构**
- 简单的功能注册接口，支持运行时注册新功能
- 标准化的功能调用接口和结果格式
- 配置化的调度规则，支持自定义调度逻辑
- 完整的调试和监控支持

#### 3.5.4 结果整合策略

**多功能结果整合**
- **顺序整合**: 按优先级顺序组织结果
- **补充整合**: 后续功能结果补充前面功能的不足
- **冲突解决**: 当不同功能结果冲突时采用高置信度结果
- **格式统一**: 将不同格式的结果转换为统一回复格式

**响应质量提升**
- 结果整合前进行冗余信息去除
- 确保信息的一致性和连贯性
- 保持回复的清晰和简洁
- 提供功能调用的透明度（可选择性展示）