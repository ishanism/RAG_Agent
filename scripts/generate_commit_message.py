#!/usr/bin/env python3
import sys
import argparse
from ollama import Client

SYSTEM_PROMPT = """You are a highly skilled software developer responsible for generating clear and informative git commit messages.
Generate a commit message with the following format, being sure to include the newlines:

<title>

<detailed explanation>

The title should be 50 chars max, imperative mood, capitalized, no period.
The explanation should be wrapped at 72 chars and explain what and why."""

def format_commit_message(response):
    """Format the LLM response into a proper commit message."""
    # Remove any XML-like tags and extra whitespace
    message = response.replace('<title>', '').replace('<detailed explanation>', '')
    parts = message.strip().split('\n\n', 1)
    
    if len(parts) == 2:
        title, body = parts
        return f"{title.strip()}\n\n{body.strip()}"
    return message.strip()

def generate_commit_message(host, model, prompt_file):
    try:
        client = Client(host=host)
        
        with open(prompt_file, 'r') as f:
            prompt = f.read()
        
        response = client.chat(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        return format_commit_message(response['message']['content'])
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate commit message using Ollama")
    parser.add_argument("--host", required=True, help="Ollama API host")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--prompt-file", required=True, help="File containing the prompt")
    
    args = parser.parse_args()
    print(generate_commit_message(args.host, args.model, args.prompt_file))