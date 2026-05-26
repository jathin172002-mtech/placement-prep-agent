import httpx
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Token limits and pricing
MAX_CONTEXT_TOKENS = 128000
PRICE_PER_INPUT_TOKEN = 0.00000059   # $ per token (Groq Llama 3.3 70B)
PRICE_PER_OUTPUT_TOKEN = 0.00000079  # $ per token
USD_TO_INR = 83.5

# Session tracking
total_input_tokens = 0
total_output_tokens = 0
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

def calculate_cost(input_tokens, output_tokens):
    cost_usd = (input_tokens * PRICE_PER_INPUT_TOKEN) + \
               (output_tokens * PRICE_PER_OUTPUT_TOKEN)
    cost_inr = cost_usd * USD_TO_INR
    return cost_usd, cost_inr

def get_context_percentage(input_tokens):
    return (input_tokens / MAX_CONTEXT_TOKENS) * 100

def chat(user_message):
    global total_input_tokens, total_output_tokens

    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    result = call_groq(conversation_history)

    assistant_message = result["choices"][0]["message"]["content"]
    input_tokens = result["usage"]["prompt_tokens"]
    output_tokens = result["usage"]["completion_tokens"]

    # Update session totals
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    # Calculate costs
    turn_cost_usd, turn_cost_inr = calculate_cost(input_tokens, output_tokens)
    total_cost_usd, total_cost_inr = calculate_cost(
        total_input_tokens, total_output_tokens
    )

    # Context window percentage
    context_pct = get_context_percentage(input_tokens)

    # Print response
    print(f"\nAssistant: {assistant_message}")
    print("-" * 50)
    print(f"📊 This turn:  {input_tokens} in + {output_tokens} out tokens | "
          f"Cost: ₹{turn_cost_inr:.4f}")
    print(f"📈 Session total: {total_input_tokens + total_output_tokens} tokens | "
          f"Cost: ₹{total_cost_inr:.4f}")
    print(f"🪟 Context window: {context_pct:.1f}% used "
          f"({input_tokens}/{MAX_CONTEXT_TOKENS} tokens)")

    # Warning at 80%
    if context_pct >= 80:
        print("⚠️  WARNING: You are at 80% of the context window!")
    print("-" * 50)

# Main loop
print("💬 Placement Prep Assistant with Cost Tracker")
print("Type 'stats' to see session summary, 'quit' to exit")
print("=" * 50)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        total_cost_usd, total_cost_inr = calculate_cost(
            total_input_tokens, total_output_tokens
        )
        print(f"\n📊 Session Summary:")
        print(f"Total input tokens:  {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"Total tokens:        {total_input_tokens + total_output_tokens}")
        print(f"Total cost:          ₹{total_cost_inr:.4f} "
              f"(${total_cost_usd:.6f})")
        break

    elif user_input.lower() == "stats":
        total_cost_usd, total_cost_inr = calculate_cost(
            total_input_tokens, total_output_tokens
        )
        print(f"\n📊 Current Stats:")
        print(f"Tokens used: {total_input_tokens + total_output_tokens}")
        print(f"Cost so far: ₹{total_cost_inr:.4f}")

    else:
        chat(user_input)
