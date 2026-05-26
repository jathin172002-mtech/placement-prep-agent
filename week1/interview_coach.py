import httpx
import os
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")

conversation_history = []

SYSTEM_PROMPT = """You are a strict technical interview coach helping Indian 
engineering students prepare for tech company placements.

STRICT RULES:
1. Ask ONLY ONE question at a time
2. After the candidate answers, give brief feedback (2-3 sentences)
3. Then ask the NEXT question
4. NEVER answer your own questions
5. NEVER simulate candidate responses
6. ALWAYS wait for the human to respond before continuing
7. Ask exactly 5 questions total, then give the JSON summary

After all 5 answers, give ONLY this JSON with no extra text:
{
    "company": "company name",
    "role": "role name",
    "total_score": 7,
    "max_score": 10,
    "strengths": ["strength1", "strength2"],
    "improvements": ["improvement1", "improvement2"],
    "verdict": "Ready / Almost Ready / Needs More Prep"
}"""

def call_groq(messages):
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages
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
    input_tokens = result["usage"]["prompt_tokens"]
    output_tokens = result["usage"]["completion_tokens"]

    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    print(f"\nCoach: {assistant_message}")
    print(f"[Tokens: {input_tokens} in / {output_tokens} out]")

    return assistant_message

def save_session(company, role, summary):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"week1/interview_{company}_{role}_{timestamp}.json".replace(" ", "_")

    session_data = {
        "timestamp": timestamp,
        "company": company,
        "role": role,
        "conversation": conversation_history,
        "summary": summary
    }

    with open(filename, "w") as f:
        json.dump(session_data, f, indent=2)

    print(f"\n💾 Session saved to {filename}")

def parse_summary(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except:
        return None

# ── Main Program ──────────────────────────────────────────

print("=" * 60)
print("🎯 PLACEMENT PREP — TERMINAL INTERVIEW COACH")
print("=" * 60)

# Step 1 — Get company and role
company = input("\nWhich company are you preparing for? ")
role = input("Which role? ")

print(f"\n✅ Starting interview prep for {role} at {company}")
print("Type your answers and press Enter. Type 'quit' to exit.\n")

# Step 2 — Start the interview
opening = f"I want to practice for {role} role at {company}. Ask me the first question only."
chat(opening)

# Step 3 — Interview loop
question_count = 0
summary = None

while question_count < 5:
    user_answer = input("\nYou: ")

    if user_answer.lower() == "quit":
        print("Interview ended early.")
        break

    # After 5th answer ask for summary
    if question_count == 4:
        response = chat(user_answer)
        print("\n⏳ Generating your performance summary...")
        summary_response = chat(
            "All 5 questions are done. Give me ONLY the JSON performance summary now. No other text."
        )
        summary = parse_summary(summary_response)
    else:
        response = chat(user_answer)

    question_count += 1

# Step 4 — Show summary
if summary:
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Company:  {summary.get('company', company)}")
    print(f"Role:     {summary.get('role', role)}")
    print(f"Score:    {summary.get('total_score')}/{summary.get('max_score')}")
    print(f"Verdict:  {summary.get('verdict')}")
    print(f"\nStrengths:")
    for s in summary.get('strengths', []):
        print(f"  ✅ {s}")
    print(f"\nImprovement Areas:")
    for i in summary.get('improvements', []):
        print(f"  📌 {i}")
    save_session(company, role, summary)
else:
    print("\n⚠️ Could not parse summary. Saving raw conversation...")
    save_session(company, role, {})