#!/usr/bin/env python3
"""
示例：如何使用重构后的Issue Analyzer架构

这个文件展示了如何在不同场景下使用新的模块化架构。
"""

import asyncio
import logging
from typing import Dict, Any

# 导入重构后的组件
from analyzer_core import IntelligentDispatcher, get_trigger_decision
from analyzer_core.models import AvailableTools
from analyzer_core.tools import get_tool_registry

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """基本使用示例"""
    print("=" * 50)
    print("示例 1: 基本使用")
    print("=" * 50)
    
    # 配置（通常来自环境变量）
    config = {
        "github_token": "fake_token_for_demo",
        "repo_owner": "test_owner", 
        "repo_name": "test_repo",
        "issue_number": "123",
        "event_name": "issues",
        "event_action": "opened"
    }
    
    # 模拟issue数据
    issue_data = {
        "title": "Bug: Application crashes when clicking submit button",
        "body": """
## Bug Description
The application crashes when I click the submit button on the contact form.

## Steps to Reproduce
1. Open the contact form page
2. Fill in all required fields
3. Click the submit button
4. Application crashes immediately

## Expected Behavior
Form should submit successfully and show confirmation message.

## Actual Behavior
Application crashes with no error message.

## Environment
- Browser: Chrome 91.0.4472.124
- OS: Windows 10
- Application Version: 2.1.0
        """,
        "labels": [],
        "comments": 0
    }
    
    try:
        # 创建调度器
        dispatcher = IntelligentDispatcher(config)
        
        # 运行分析（这里会因为缺少真实的Azure配置而失败，但展示了用法）
        print("✓ 调度器创建成功")
        print("✓ 准备运行分析...")
        
        # 在真实环境中，这里会执行完整的分析
        # result = await dispatcher.analyze(issue_data, None)
        
    except Exception as e:
        print(f"⚠️  预期错误（缺少Azure配置）: {e}")


async def example_trigger_logic():
    """触发逻辑示例"""
    print("\n" + "=" * 50)
    print("示例 2: 触发逻辑测试")
    print("=" * 50)
    
    # 测试不同的触发场景
    test_cases = [
        {
            "name": "新issue创建",
            "issue_data": {
                "title": "New bug report",
                "body": "Something is broken",
                "user": {"login": "user123"}
            },
            "comment_data": None
        },
        {
            "name": "机器人评论",
            "issue_data": {
                "title": "Existing issue", 
                "body": "Original issue",
                "user": {"login": "user123"}
            },
            "comment_data": {
                "body": "Automated comment",
                "user": {"login": "github-actions[bot]"}
            }
        },
        {
            "name": "用户提及agent",
            "issue_data": {
                "title": "Need help",
                "body": "I need assistance", 
                "user": {"login": "user123"}
            },
            "comment_data": {
                "body": "@issue-agent please help with this",
                "user": {"login": "other_user"}
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        decision = get_trigger_decision(
            test_case["issue_data"],
            test_case["comment_data"]
        )
        
        print(f"  应该触发: {decision.should_trigger}")
        print(f"  原因: {decision.reason}")
        print(f"  触发类型: {decision.trigger_type}")


def example_tool_registry():
    """工具注册表示例"""
    print("\n" + "=" * 50)
    print("示例 3: 工具注册表")
    print("=" * 50)
    
    # 获取工具注册表
    registry = get_tool_registry()
    
    print(f"可用工具数量: {len(registry)}")
    print("\n可用工具列表:")
    
    for tool_enum, tool_instance in registry.items():
        print(f"  ✓ {tool_enum.value}: {tool_instance.__class__.__name__}")
    
    # 展示如何使用特定工具
    print(f"\n示例：如何访问RAG搜索工具")
    rag_tool = registry.get(AvailableTools.RAG_SEARCH)
    if rag_tool:
        print(f"  工具名称: {rag_tool.name}")
        print(f"  工具类型: {type(rag_tool).__name__}")


async def example_custom_tool():
    """自定义工具示例"""
    print("\n" + "=" * 50)
    print("示例 4: 创建自定义工具")
    print("=" * 50)
    
    from analyzer_core.tools.base_tool import BaseTool
    
    class CustomAnalysisTool(BaseTool):
        """自定义分析工具示例"""
        
        def __init__(self):
            super().__init__("custom_analysis")
        
        async def execute(self, issue_data: Dict[str, Any], 
                         comment_data=None) -> Dict[str, Any]:
            """执行自定义分析"""
            
            # 简单的关键词分析
            title = issue_data.get('title', '').lower()
            body = issue_data.get('body', '').lower()
            content = f"{title} {body}"
            
            keywords = {
                'bug_keywords': ['bug', 'error', 'crash', 'fail', 'broken'],
                'feature_keywords': ['feature', 'enhancement', 'improve', 'add'],
                'performance_keywords': ['slow', 'performance', 'speed', 'lag']
            }
            
            analysis = {}
            for category, words in keywords.items():
                count = sum(1 for word in words if word in content)
                analysis[category] = count
            
            # 确定主要类别
            main_category = max(analysis.keys(), key=lambda k: analysis[k])
            
            return {
                "keyword_analysis": analysis,
                "suggested_category": main_category,
                "confidence": analysis[main_category] / len(keywords[main_category])
            }
    
    # 使用自定义工具
    custom_tool = CustomAnalysisTool()
    
    issue_data = {
        "title": "Performance issue: App is very slow",
        "body": "The application performance is terrible. It's very slow to load."
    }
    
    result = await custom_tool.execute(issue_data)
    
    print("自定义工具分析结果:")
    print(f"  关键词分析: {result['keyword_analysis']}")
    print(f"  建议类别: {result['suggested_category']}")
    print(f"  置信度: {result['confidence']:.2f}")


async def main():
    """运行所有示例"""
    print("🧪 Issue Analyzer 架构使用示例")
    print("📚 展示重构后组件的使用方法\n")
    
    # 运行各个示例
    await example_basic_usage()
    await example_trigger_logic()
    example_tool_registry()
    await example_custom_tool()
    
    print("\n" + "=" * 50)
    print("✅ 所有示例执行完成！")
    print("=" * 50)
    print("\n💡 关键要点:")
    print("1. 新架构模块化程度高，每个组件职责清晰")
    print("2. 触发逻辑智能化，避免不必要的处理")
    print("3. 工具系统可扩展，易于添加新功能") 
    print("4. 配置统一管理，便于部署和维护")
    print("5. 错误处理健壮，系统稳定性高")


if __name__ == "__main__":
    asyncio.run(main())
