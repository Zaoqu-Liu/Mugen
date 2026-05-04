#!/usr/bin/env python3
"""
Mugen API Server — Python client example.

Demonstrates non-streaming and streaming chat completions using the
official OpenAI Python SDK against a local Mugen server.

Prerequisites:
    pip install openai

Usage:
    1. Start the server:  mugen-cli serve --model ./models/llama-2-7b-chat.Q4_K_M.gguf
    2. Run this script:   python chat.py
"""

from openai import OpenAI

MUGEN_BASE_URL = "http://127.0.0.1:8080/v1"
MODEL_NAME = "llama-2-7b-chat"

client = OpenAI(
    base_url=MUGEN_BASE_URL,
    api_key="not-needed",  # Omit --api-key on the server to skip auth.
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain unified memory in two sentences."},
]


def non_streaming():
    """Send a request and receive the full response at once."""
    print("=== Non-streaming ===\n")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
    )
    print(response.choices[0].message.content)
    print(f"\nTokens: {response.usage.total_tokens}\n")


def streaming():
    """Send a request and print tokens as they arrive."""
    print("=== Streaming ===\n")
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    non_streaming()
    streaming()
