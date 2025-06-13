# analyzer_core/actions/action_executor.py
"""
Action executor that coordinates GitHub actions based on analysis results
"""

import logging
from typing import Dict, Any, List, Set
from .github_actions import GitHubActionExecutor

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Coordinates action execution based on analysis results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.github_executor = GitHubActionExecutor(config)
    
    async def execute(self, final_analysis: Dict[str, Any], issue_data: Dict[str, Any], 
                     strategy_actions: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute actions based on the final analysis and strategy recommendations
        
        Args:
            final_analysis: Final analysis from LLM
            issue_data: Original issue data
            strategy_actions: Strategy-recommended actions (optional)
            
        Returns:
            List of actions taken
        """
        actions_taken = []
        all_labels_to_add: Set[str] = set()  # Use set to avoid duplicate labels
        
        try:
            # Step 1: Collect all labels from final analysis
            recommended_labels = final_analysis.get("recommended_labels", [])
            if recommended_labels:
                all_labels_to_add.update(recommended_labels)
            
            # Step 2: Process final analysis recommended actions
            recommended_actions = final_analysis.get("recommended_actions", [])
            
            # Step 3: Merge with strategy-recommended actions if provided
            if strategy_actions:
                logger.info(f"Merging {len(strategy_actions)} strategy-recommended actions")
                recommended_actions.extend(strategy_actions)
            
            # Process all actions to collect labels and other actions
            non_label_actions = []
            
            for action in recommended_actions:
                action_type = action.get("action", "")
                action_details = action.get("details", "")
                
                if action_type == "add_label" and action_details:
                    # Collect label for batch addition
                    labels_list = list(all_labels_to_add)
                    success = await self.github_executor.add_labels(labels_list)
                    actions_taken.append({
                        "action": "labels_added",
                        "details": f"Added labels: {', '.join(labels_list)}",
                        "success": success
                    })

                elif action_type == "add_comment":
                    logger.info("Skipping add_comment action - using main user_comment instead")
                    actions_taken.append({
                        "action": "comment_action_merged",
                        "details": "Comment action merged into main user_comment",
                        "success": True
                    })
                else:
                    # Other actions (assign_user, close_issue, etc.)
                    non_label_actions.append(action)
            
            # Step 2: Add labels if any
            # if all_labels_to_add:
            #     labels_list = list(all_labels_to_add)
            #     success = await self.github_executor.add_labels(labels_list)
            #     actions_taken.append({
            #         "action": "labels_added",
            #         "details": f"Added labels: {', '.join(labels_list)}",
            #         "success": success
            #     })
            
            # Step 3: Add the main user comment
            user_comment = final_analysis.get("user_comment", "")
            if user_comment:
                success = await self.github_executor.add_comment(user_comment)
                actions_taken.append({
                    "action": "comment_added",
                    "details": "Analysis comment posted",
                    "success": success
                })
            
            # Step 4: Execute other non-label, non-comment actions
            for action in non_label_actions:
                try:
                    action_type = action.get("action", "")
                    action_details = action.get("details", "")
                    
                    if action_type == "assign_user" and action_details:
                        success = await self.github_executor.assign_user(action_details)
                        actions_taken.append({
                            "action": "user_assigned",
                            "details": f"Assigned to {action_details}",
                            "success": success
                        })
                    elif action_type == "close_issue":
                        success = await self.github_executor.close_issue(action_details)
                        actions_taken.append({
                            "action": "issue_closed",
                            "details": f"Issue closed: {action_details}",
                            "success": success
                        })
                    elif action_type == "request_info":
                        # This would be handled through the user_comment
                        logger.info("Info request handled through main comment")
                        actions_taken.append({
                            "action": "info_request_handled",
                            "details": "Information request included in main comment",
                            "success": True
                        })
                    else:
                        logger.warning(f"Unknown action type: {action_type}")
                        actions_taken.append({
                            "action": "unknown_action",
                            "details": f"Unknown action type: {action_type}",
                            "success": False
                        })
                    
                except Exception as e:
                    logger.error(f"Error executing action {action}: {str(e)}")
                    actions_taken.append({
                        "action": action.get("action", "unknown"),
                        "details": f"Failed: {str(e)}",
                        "success": False
                    })
            
        except Exception as e:
            logger.error(f"Error taking intelligent actions: {str(e)}")
            actions_taken.append({
                "action": "error",
                "details": str(e),
                "success": False
            })
        
        return actions_taken
