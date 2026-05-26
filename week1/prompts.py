import httpx
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")

def call_groq(prompt):
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# TECHNIQUE 1 - Zero Shot (just ask directly)
zero_shot = """
What is the time complexity of binary search?
"""

# TECHNIQUE 2 - Few Shot (give examples first)
few_shot = """
Classify the time complexity of these algorithms:
- Linear search: O(n)
- Accessing array element: O(1)
- Bubble sort: O(n²)

Now classify this:
- Binary search: ?
"""

# TECHNIQUE 3 - Chain of Thought (ask it to think step by step)
chain_of_thought = """
What is the time complexity of binary search?
Think through this step by step:
1. First explain what binary search does
2. Then count how many steps it takes for n elements
3. Then derive the time complexity
"""

print("=== ZERO SHOT ===")
print(call_groq(zero_shot))

print("\n=== FEW SHOT ===")
print(call_groq(few_shot))

print("\n=== CHAIN OF THOUGHT ===")
print(call_groq(chain_of_thought))
# TECHNIQUE 4 - Structured JSON Output
json_prompt = """
Extract information about this company and return ONLY a JSON object.
No explanation, no markdown, just pure JSON.

Company: Zomato

Return this exact structure:
{
    "company": "company name",
    "founded": "year",
    "headquarters": "city, country",
    "tech_stack": ["tech1", "tech2"],
    "known_for": "one sentence",
    "typical_roles": ["role1", "role2"]
}
"""

import json

print("\n=== STRUCTURED JSON OUTPUT ===")
result = call_groq(json_prompt)
print(result)

# Now parse it
try:
    parsed = json.loads(result)
    print("\n✅ Successfully parsed as JSON!")
    print(f"Company: {parsed['company']}")
    print(f"Founded: {parsed['founded']}")
    print(f"Tech Stack: {', '.join(parsed['tech_stack'])}")
except:
    print("\n❌ Failed to parse as JSON")
