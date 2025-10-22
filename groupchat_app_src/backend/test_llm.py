"""
Test script to verify the Llama-3.2-3B-Instruct model integration
"""
import asyncio
from llm import chat_completion

async def test_model():
    print("Testing Llama-3.2-3B-Instruct-IQ3_M.gguf model...")
    print("-" * 60)

    # Test 1: Simple question
    print("\nTest 1: Simple greeting")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you introduce yourself?"}
    ]

    try:
        response = await chat_completion(messages, temperature=0.7, max_tokens=200)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Question with question mark (triggers bot in app)
    print("\n" + "-" * 60)
    print("\nTest 2: Question about Python")
    messages = [
        {"role": "system", "content": "You are a helpful assistant participating in a small group chat. Provide concise, accurate answers suitable for a shared chat context."},
        {"role": "user", "content": "What is Python used for?"}
    ]

    try:
        response = await chat_completion(messages, temperature=0.2, max_tokens=150)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "-" * 60)
    print("\nModel integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_model())
