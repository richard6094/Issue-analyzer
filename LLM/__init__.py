"""
LLM module based on LangChain architecture for generating text using language models.
"""

from .llm_provider import (
    get_llm,
    get_openai_llm,
    get_azure_llm,
    generate_text,
    generate_chat_response,
    generate_structured_output
)

__all__ = [
    'get_llm',
    'get_openai_llm',
    'get_azure_llm',
    'generate_text', 
    'generate_chat_response',
    'generate_structured_output'
]