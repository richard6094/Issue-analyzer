"""
Demonstration of how to use the LLM module based on LangChain architecture.
"""
import argparse
import os
import sys
import json
from typing import Dict, Any, List

# Add parent directory to sys.path to import the LLM module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the LLM module
from LLM import (
    get_llm,
    get_openai_llm,
    get_azure_llm,
    generate_text,
    generate_chat_response,
    generate_structured_output
)

# Import LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough


def run_text_generation_demo(args):
    """Run text generation demonstration"""
    print(f"Using LangChain to generate text response: '{args.prompt}'")
    
    if args.use_langchain_chain:
        # Method 1: Using complete LangChain chain
        llm = get_llm(
            provider="azure" if args.use_azure_openai else "openai",
            model=args.azure_deployment if args.use_azure_openai else args.openai_model,
            temperature=args.temperature,
            endpoint=args.azure_endpoint if args.use_azure_openai else None,
            api_key=args.openai_api_key if not args.use_azure_openai else args.azure_key,
            max_tokens=args.max_tokens
        )
        
        chain = llm | StrOutputParser()
        response = chain.invoke(args.prompt)
    else:
        # Method 2: Using the wrapper function interface
        response = generate_text(
            prompt=args.prompt,
            provider="azure" if args.use_azure_openai else "openai",
            model=args.azure_deployment if args.use_azure_openai else args.openai_model,
            temperature=args.temperature,
            endpoint=args.azure_endpoint if args.use_azure_openai else None,
            api_key=args.openai_api_key if not args.use_azure_openai else args.azure_key,
            max_tokens=args.max_tokens
        )
    
    print("\n--- Generated Text ---")
    print(response)


def run_chat_demo(args):
    """Run chat demonstration"""
    print(f"Using LangChain to generate chat response: '{args.prompt}'")
    
    # Create message list
    system_prompt = "You are a helpful AI assistant providing clear and concise information."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.prompt}
    ]
    
    if args.use_langchain_chain:
        # Method 1: Using complete LangChain chain with ChatPromptTemplate
        llm = get_llm(
            provider="azure" if args.use_azure_openai else "openai",
            model=args.azure_deployment if args.use_azure_openai else args.openai_model,
            temperature=args.temperature,
            endpoint=args.azure_endpoint if args.use_azure_openai else None,
            api_key=args.openai_api_key if not args.use_azure_openai else args.azure_key,
            max_tokens=args.max_tokens
        )
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        chain = chat_prompt | llm | StrOutputParser()
        response = chain.invoke({"input": args.prompt})
        
        print("\n--- LangChain Chat Response ---")
        print(response)
    else:
        # Method 2: Using the wrapper function interface
        response = generate_chat_response(
            messages=messages,
            provider="azure" if args.use_azure_openai else "openai",
            model=args.azure_deployment if args.use_azure_openai else args.openai_model,
            temperature=args.temperature,
            endpoint=args.azure_endpoint if args.use_azure_openai else None,
            api_key=args.openai_api_key if not args.use_azure_openai else args.azure_key,
            max_tokens=args.max_tokens
        )
        
        print("\n--- Chat Response ---")
        print(f"Content: {response['content']}")
        if 'model' in response:
            print(f"Model: {response['model']}")
        if 'usage' in response:
            print(f"Token Usage: {response['usage']}")


def run_structured_output_demo(args):
    """Run structured output demonstration"""
    print(f"Using LangChain to generate structured output: '{args.prompt}'")
    
    # Define output schema
    output_schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A concise summary of the response"
            },
            "key_points": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of key points from the response"
            },
            "difficulty": {
                "type": "string",
                "enum": ["beginner", "intermediate", "advanced"],
                "description": "The difficulty level of the content"
            }
        },
        "required": ["summary", "key_points", "difficulty"]
    }
    
    if args.use_langchain_chain:
        # Method 1: Using complete LangChain chain
        llm = get_llm(
            provider="azure" if args.use_azure_openai else "openai",
            model=args.azure_deployment if args.use_azure_openai else args.openai_model,
            temperature=args.temperature,
            endpoint=args.azure_endpoint if args.use_azure_openai else None,
            api_key=args.openai_api_key if not args.use_azure_openai else args.azure_key,
            max_tokens=args.max_tokens,
            response_format={"type": "json_object"}
        )
        
        # Create system prompt
        system_prompt = f"""
        You will be given a prompt. Please respond with a JSON object that matches this schema:
        {json.dumps(output_schema, indent=2)}
        
        Ensure your response is valid JSON and conforms to the schema.
        Only return the JSON object, do not add any additional text.
        """
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create chain
        chain = (
            {"input": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
            | (lambda x: json.loads(x))
        )
        
        response = chain.invoke(args.prompt)
    else:
        # Method 2: Using the wrapper function interface
        response = generate_structured_output(
            prompt=args.prompt,
            output_schema=output_schema,
            provider="azure" if args.use_azure_openai else "openai",
            model=args.azure_deployment if args.use_azure_openai else args.openai_model,
            temperature=args.temperature,
            endpoint=args.azure_endpoint if args.use_azure_openai else None,
            api_key=args.openai_api_key if not args.use_azure_openai else args.azure_key,
            max_tokens=args.max_tokens
        )
    
    print("\n--- Structured Output ---")
    print(json.dumps(response, indent=2))


def main():
    """
    Main function demonstrating the use of the LLM module
    """
    parser = argparse.ArgumentParser(description="Demonstrate LLM module based on LangChain")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms.",
                      help="Text prompt for generation")
    
    # Generation mode
    parser.add_argument("--mode", type=str, choices=["text", "chat", "structured"], default="text",
                      help="Generation mode: text, chat, or structured output")
    
    # LangChain options
    parser.add_argument("--use-langchain-chain", action="store_true",
                      help="Use complete LangChain chain instead of wrapper functions")
    
    # OpenAI options
    parser.add_argument("--openai-api-key", type=str,
                      help="OpenAI API key (if not set as environment variable)")
    parser.add_argument("--openai-model", type=str, default="gpt-4o",
                      help="OpenAI model for text generation")
    
    # Azure OpenAI options
    parser.add_argument("--use-azure-openai", action="store_true",
                      help="Use Azure OpenAI instead of regular OpenAI")
    parser.add_argument("--azure-endpoint", type=str,
                      help="Azure OpenAI endpoint URL")
    parser.add_argument("--azure-key", type=str,
                      help="Azure OpenAI API key")
    parser.add_argument("--azure-deployment", type=str,
                      help="Azure OpenAI deployment name")
    parser.add_argument("--azure-api-version", type=str, default="2025-01-01-preview",
                      help="Azure OpenAI API version")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Temperature for text generation (higher = more random)")
    parser.add_argument("--max-tokens", type=int, default=1000,
                      help="Maximum number of tokens to generate")
    
    args = parser.parse_args()
    
    try:
        # Run different demos based on mode
        if args.mode == "text":
            run_text_generation_demo(args)
        elif args.mode == "chat":
            run_chat_demo(args)
        elif args.mode == "structured":
            run_structured_output_demo(args)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())