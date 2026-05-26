import httpx
import os
import json
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

def get_company_info(company_name):
    prompt = f"""
    Return information about {company_name} as a JSON object.
    Return ONLY the JSON object, no explanation, no markdown, no backticks.
    
    Use exactly this structure:
    {{
        "company": "company name",
        "founded": "year",
        "hq": "city, country",
        "tech_stack": ["tech1", "tech2", "tech3"],
        "known_for": "one sentence description",
        "typical_roles": ["role1", "role2", "role3"]
    }}
    """
    
    # Retry up to 3 times if JSON parsing fails
    for attempt in range(1, 4):
        print(f"Attempt {attempt}...")
        result = call_groq(prompt)
        
        try:
            parsed = json.loads(result)
            print("✅ Successfully parsed JSON!")
            return parsed
        except json.JSONDecodeError:
            print(f"❌ Attempt {attempt} failed - invalid JSON returned")
            if attempt == 3:
                print("All 3 attempts failed.")
                return None

    return None

# Ask user for company name
company = input("Enter company name: ")
info = get_company_info(company)

if info:
    print("\n--- Company Profile ---")
    print(f"Company:       {info['company']}")
    print(f"Founded:       {info['founded']}")
    print(f"HQ:            {info['hq']}")
    print(f"Tech Stack:    {', '.join(info['tech_stack'])}")
    print(f"Known For:     {info['known_for']}")
    print(f"Typical Roles: {', '.join(info['typical_roles'])}")
