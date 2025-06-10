#!/usr/bin/env python3
"""
Test script to verify JSON response parsing fix
"""

import json
import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def test_json_parsing():
    """Test the JSON parsing helper methods"""
    
    # Import the dispatcher class
    from intelligent_dispatch_action import IntelligentGitHubDispatcher
    
    # Create a test instance
    dispatcher = IntelligentGitHubDispatcher()
    
    # Test case 1: Clean JSON response
    clean_json = '{"issue_type": "bug_report", "user_comment": "This is a clean comment", "confidence": 0.8}'
    print("Test 1: Clean JSON")
    try:
        result = dispatcher._extract_and_parse_json_response(clean_json)
        print(f"✓ Parsed successfully: {result}")
        cleaned_comment = dispatcher._clean_user_comment(result['user_comment'])
        print(f"✓ Cleaned comment: {cleaned_comment}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test case 2: Nested JSON in user_comment (the problematic case)
    nested_json = '''{"issue_type": "bug_report", "user_comment": "{\\"nested\\": \\"This is nested JSON that should be extracted\\", \\"user_comment\\": \\"This is the actual comment\\"}", "confidence": 0.8}'''
    print("\nTest 2: Nested JSON in user_comment")
    try:
        result = dispatcher._extract_and_parse_json_response(nested_json)
        print(f"✓ Parsed successfully: {result}")
        cleaned_comment = dispatcher._clean_user_comment(result['user_comment'])
        print(f"✓ Cleaned comment: {cleaned_comment}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test case 3: JSON with markdown formatting
    markdown_json = '''```json
    {
        "issue_type": "bug_report", 
        "user_comment": "This is a markdown formatted comment",
        "confidence": 0.9
    }
    ```'''
    print("\nTest 3: Markdown formatted JSON")
    try:
        result = dispatcher._extract_and_parse_json_response(markdown_json)
        print(f"✓ Parsed successfully: {result}")
        cleaned_comment = dispatcher._clean_user_comment(result['user_comment'])
        print(f"✓ Cleaned comment: {cleaned_comment}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test case 4: Malformed response (fallback scenario)
    malformed_response = '''This is not JSON but contains some text.
    There might be a "user_comment": "extracted comment" somewhere in here.
    The rest is just random text.'''
    print("\nTest 4: Malformed response (fallback)")
    try:
        comment = dispatcher._extract_user_comment_from_response(malformed_response)
        print(f"✓ Extracted comment: {comment}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n" + "="*50)
    print("JSON Parsing Fix Test Completed!")
    print("The fix should prevent complete JSON structures from being posted as user comments.")

if __name__ == "__main__":
    test_json_parsing()
