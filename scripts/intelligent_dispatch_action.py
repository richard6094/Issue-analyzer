#!/usr/bin/env python3
"""
Intelligent Function Dispatcher for GitHub Actions - Entry Point

This script serves as the entry point for the intelligent issue analyzer system.
It has been refactored to use the modular analyzer_core components.
"""

import os
import sys
import json
import asyncio
import logging

# Add parent directory to sys.path to allow importing sibling packages
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the new modular components
from analyzer_core import IntelligentDispatcher
from analyzer_core.utils import fetch_issue_data, fetch_comment_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('intelligent_dispatch_logs.txt')
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for GitHub Actions"""
    try:
        logger.info("Starting Intelligent Function Dispatcher (Refactored Version)...")
        
        # Load configuration from environment
        config = {
            "github_token": os.environ.get("GITHUB_TOKEN"),
            "repo_owner": os.environ.get("REPO_OWNER"),
            "repo_name": os.environ.get("REPO_NAME"),
            "repo_full_name": os.environ.get("REPO_FULL_NAME"),
            "issue_number": os.environ.get("ISSUE_NUMBER"),
            "event_name": os.environ.get("GITHUB_EVENT_NAME"),
            "event_action": os.environ.get("GITHUB_EVENT_ACTION"),
            "sender_login": os.environ.get("SENDER_LOGIN"),
            "sender_type": os.environ.get("SENDER_TYPE"),
            "force_analysis": os.environ.get("FORCE_ANALYSIS", "false").lower() == "true"
        }
        
        # Validate configuration
        required_fields = ["github_token", "repo_owner", "repo_name", "issue_number"]
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
        
        # Fetch data
        logger.info("Fetching issue and comment data...")
        issue_data = await fetch_issue_data(config)
        if not issue_data:
            raise ValueError("Failed to fetch issue data")
        
        comment_data = await fetch_comment_data(config)
        
        # Create and run dispatcher
        logger.info("Initializing intelligent dispatcher...")
        dispatcher = IntelligentDispatcher(config)
        
        logger.info("Running analysis workflow...")
        result = await dispatcher.analyze(issue_data, comment_data)
        
        # Save results
        output_file = "intelligent_dispatch_results.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        logger.info("Intelligent Function Dispatcher completed successfully")
        
        # Print summary
        if result.get("trigger_decision", {}).get("should_trigger"):
            logger.info(f"Analysis completed. Trigger type: {result.get('trigger_decision', {}).get('trigger_type')}")
            logger.info(f"Actions taken: {len(result.get('actions_taken', []))}")
        else:
            logger.info(f"Analysis skipped. Reason: {result.get('trigger_decision', {}).get('reason')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"Intelligent Function Dispatcher failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))