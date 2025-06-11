# analyzer_core/tools/image_tool.py
"""
Image analysis tool for visual content
"""

import re
from typing import Dict, Any, Optional, List
from .base_tool import BaseTool
from image_recognition.image_recognition_provider import get_image_recognition_model, analyze_image


class ImageAnalysisTool(BaseTool):
    """Image analysis tool for visual content"""
    
    def __init__(self):
        super().__init__("image_analysis")
    
    async def execute(self, issue_data: Dict[str, Any], 
                     comment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute image analysis for visual content"""
        try:
            body = issue_data.get('body', '')
            if comment_data:
                body += f"\n{comment_data.get('body', '')}"
                
            image_urls = self._extract_image_urls(body)
            
            if not image_urls:
                return {"message": "No images found", "confidence": 0.0}
            
            image_model = get_image_recognition_model(provider="azure")
            analysis_results = []
            
            for url in image_urls[:3]:  # Limit to 3 images
                try:
                    result = analyze_image(
                        image_url=url,
                        prompt=f"Analyze this image in the context of a GitHub issue: {issue_data.get('title', '')}",
                        llm=image_model
                    )
                    analysis_results.append({
                        "url": url,
                        "analysis": result
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to analyze image {url}: {str(e)}")
            
            return {
                "images_analyzed": len(analysis_results),
                "results": analysis_results,
                "confidence": 0.8 if analysis_results else 0.2
            }
        except Exception as e:
            self.logger.error(f"Image analysis failed: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    def _extract_image_urls(self, text: str) -> List[str]:
        """Extract image URLs from text"""
        patterns = [
            r'!\[.*?\]\((https?://[^\s\)]+\.(?:png|jpg|jpeg|gif|svg))\)',  # Markdown images
            r'<img[^>]+src=["\']([^"\']+)["\']',  # HTML img tags
            r'(https?://[^\s]+\.(?:png|jpg|jpeg|gif|svg))',  # Direct URLs
        ]
        
        urls = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            urls.extend(matches)
        
        return list(set(urls))  # Remove duplicates
