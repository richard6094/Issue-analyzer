# analyzer_core/utils/json_utils.py
"""
JSON processing utilities
"""

import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def extract_json_from_llm_response(response_text: str, logger_instance=None) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM response that might contain extra text or markdown
    
    Args:
        response_text: Raw LLM response text
        logger_instance: Optional logger for debug output
        
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    if logger_instance:
        logger_instance.debug(f"Attempting to extract JSON from response of length: {len(response_text)}")
    
    # Strategy 1: Try direct parsing
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        if logger_instance:
            logger_instance.debug("Direct JSON parsing failed, trying other strategies")
    
    # Strategy 2: Extract from markdown code blocks
    markdown_patterns = [
        r'```(?:json)?\s*\n?(.*?)\n?```',  # ```json ... ``` or ``` ... ```
        r'```(?:JSON)?\s*\n?(.*?)\n?```',  # ```JSON ... ```
    ]
    
    for pattern in markdown_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1).strip()
                if logger_instance:
                    logger_instance.debug(f"Found JSON in markdown block, length: {len(json_str)}")
                return json.loads(json_str)
            except json.JSONDecodeError:
                if logger_instance:
                    logger_instance.debug("Failed to parse JSON from markdown block")
    
    # Strategy 3: Find JSON by looking for outermost braces
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            json_str = response_text[start_idx:end_idx + 1]
            if logger_instance:
                logger_instance.debug(f"Extracted JSON by braces, length: {len(json_str)}")
            return json.loads(json_str)
        except json.JSONDecodeError:
            if logger_instance:
                logger_instance.debug("Failed to parse JSON extracted by braces")
    
    # Strategy 4: Clean common issues and retry
    if start_idx != -1 and end_idx != -1:
        json_str = response_text[start_idx:end_idx + 1]
        # Remove common issues
        cleaned = json_str.strip()
        # Replace single quotes with double quotes (common LLM mistake)
        cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)
        # Remove trailing commas
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        try:
            if logger_instance:
                logger_instance.debug("Trying cleaned JSON")
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if logger_instance:
                logger_instance.debug("Failed to parse cleaned JSON")
    
    if logger_instance:
        logger_instance.warning("All JSON extraction strategies failed")
    
    return None


def extract_and_parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response, handling nested JSON structures
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    
    # First try to parse the entire response as JSON
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON within the response using regex
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    json_match = re.search(json_pattern, response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON block without markdown formatting
    json_pattern2 = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.findall(json_pattern2, response_text, re.DOTALL)
    
    for json_str in json_matches:
        try:
            parsed = json.loads(json_str)
            # Verify it has expected structure
            if isinstance(parsed, dict) and any(key in parsed for key in ['issue_type', 'user_comment', 'summary']):
                return parsed
        except json.JSONDecodeError:
            continue
    
    # If all else fails, raise an error
    raise json.JSONDecodeError("Could not extract valid JSON from response", response_text, 0)


def clean_user_comment(user_comment: str) -> str:
    """
    Clean and validate user comment, handling nested JSON structures
    
    Args:
        user_comment: Raw user comment string
        
    Returns:
        Cleaned user comment text
    """
    if not user_comment:
        return "I've analyzed this issue. Please let me know if you need any clarification."
    
    # Check if the user_comment itself contains JSON
    try:
        # Try to parse as JSON to see if it's a nested structure
        parsed_comment = json.loads(user_comment)
        if isinstance(parsed_comment, dict):
            # Extract the actual comment from nested JSON
            if "user_comment" in parsed_comment:
                return clean_user_comment(parsed_comment["user_comment"])
            elif "message" in parsed_comment:
                return str(parsed_comment["message"])
            elif "content" in parsed_comment:
                return str(parsed_comment["content"])
            else:
                # If it's a JSON object but no clear comment field, return a fallback
                logger.warning(f"Nested JSON found in user_comment but no clear text field: {parsed_comment}")
                return "I've completed the analysis of this issue. The results have been processed successfully."
    except json.JSONDecodeError:
        # Not JSON, treat as regular string
        pass
    
    # Clean up common formatting issues
    cleaned = str(user_comment).strip()
    
    # Remove markdown code blocks if they contain JSON
    json_block_pattern = r'```json\s*\{.*?\}\s*```'
    if re.search(json_block_pattern, cleaned, re.DOTALL):
        # This appears to be a JSON block, extract meaningful text
        logger.warning(f"Removing JSON code block from user comment: {cleaned[:100]}...")
        return "I've analyzed this issue and the results are available. Please let me know if you need any additional information."
    
    # Remove any remaining JSON-like structures
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    cleaned = re.sub(json_pattern, '', cleaned).strip()
    
    # Ensure we have some meaningful content
    if len(cleaned) < 10:
        return "I've completed the analysis of this issue. The details have been recorded successfully."
    
    return cleaned


def extract_user_comment_from_response(response_text: str) -> str:
    """
    Extract a user-friendly comment from raw LLM response when JSON parsing fails
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        User-friendly comment text
    """
    
    # Try to find any user_comment field in the text
    user_comment_pattern = r'"user_comment":\s*"([^"]*)"'
    match = re.search(user_comment_pattern, response_text)
    if match:
        return clean_user_comment(match.group(1))
    
    # Try to find analysis or summary text that could serve as a comment
    lines = response_text.split('\n')
    meaningful_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, JSON brackets, and technical formatting
        if (line and 
            not line.startswith('{') and 
            not line.startswith('}') and 
            not line.startswith('"') and
            not line.startswith('[') and
            not line.startswith(']') and
            len(line) > 20):  # Minimum length for meaningful content
            meaningful_lines.append(line)
    
    if meaningful_lines:
        # Take the first meaningful line as the comment
        comment = meaningful_lines[0]
        # Clean up any JSON fragments
        comment = re.sub(r'[{}"\[\]]', '', comment).strip()
        if len(comment) > 10:
            return comment
    
    # Fallback comment
    return "I've analyzed this issue. The analysis has been completed and the results are being processed."
