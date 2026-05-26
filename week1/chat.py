import httpx
import os
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")

conversation_history = []

def call_groq(messages):
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": messages
        }
    )
    return response.json()

def chat(user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    result = call_groq(conversation_history)
    
    assistant_message = result["choices"][0]["message"]["content"]
    
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    print(f"\nAssistant: {assistant_message}")
    print(f"[Tokens used - Input: {result['usage']['prompt_tokens']} | Output: {result['usage']['completion_tokens']}]")

def save_conversation():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"week1/conversation_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(conversation_history, f, indent=2)
    print(f"\nConversation saved to {filename}")

print("Placement Prep Assistant (type 'quit' to exit)")
print("=" * 50)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        save_conversation()
        break
    chat(user_input)