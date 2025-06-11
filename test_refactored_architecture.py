#!/usr/bin/env python3
"""
Test script to verify the refactored architecture
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def test_imports():
    """Test that all modules can be imported successfully"""
    try:
        print("Testing imports...")
        
        # Test core imports
        from analyzer_core import IntelligentDispatcher, get_trigger_decision
        print("✓ Core dispatcher and trigger logic imported")
        
        # Test model imports
        from analyzer_core.models import AvailableTools, ToolResult, DecisionStep
        print("✓ Data models imported")
        
        # Test tool imports
        from analyzer_core.tools import get_tool_registry
        print("✓ Tool registry imported")
        
        # Test analyzer imports
        from analyzer_core.analyzers import InitialAssessor, ResultAnalyzer, FinalAnalyzer
        print("✓ Analyzer components imported")
        
        # Test action imports
        from analyzer_core.actions import ActionExecutor
        print("✓ Action executor imported")
        
        # Test utility imports
        from analyzer_core.utils import assess_user_provided_information, fetch_issue_data
        print("✓ Utility functions imported")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_tool_registry():
    """Test that tool registry works correctly"""
    try:
        print("\nTesting tool registry...")
        
        from analyzer_core.tools import get_tool_registry
        from analyzer_core.models import AvailableTools
        
        registry = get_tool_registry()
        print(f"✓ Tool registry created with {len(registry)} tools")
        
        # Check that expected tools are present
        expected_tools = [
            AvailableTools.RAG_SEARCH,
            AvailableTools.IMAGE_ANALYSIS,
            AvailableTools.REGRESSION_ANALYSIS,
            AvailableTools.SIMILAR_ISSUES,
            AvailableTools.TEMPLATE_GENERATION
        ]
        
        for tool in expected_tools:
            if tool in registry:
                print(f"✓ {tool.value} tool available")
            else:
                print(f"❌ {tool.value} tool missing")
                return False
        
        print("✅ Tool registry test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tool registry test failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation"""
    try:
        print("\nTesting configuration...")
        
        # Mock configuration
        config = {
            "github_token": "fake_token",
            "repo_owner": "test_owner",
            "repo_name": "test_repo",
            "issue_number": "123",
            "event_name": "issues",
            "event_action": "opened"
        }
        
        from analyzer_core import IntelligentDispatcher
        
        # This should not fail with proper config
        dispatcher = IntelligentDispatcher(config)
        print("✓ Dispatcher created successfully with mock config")
        
        print("✅ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Refactored Architecture\n")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_tool_registry, 
        test_config_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        
        print("-" * 30)
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Architecture is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
