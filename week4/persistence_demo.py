import json
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b")
parser = StrOutputParser()

# ── Pydantic Model ─────────────────────────────────────────

class CompanyProfile(BaseModel):
    company_name: str = Field(description="Name of company")
    difficulty: Literal["low", "medium", "high"] = Field(
        description="Interview difficulty"
    )
    key_topics: list[str] = Field(description="Top 5 topics to prepare")
    rounds: list[str] = Field(description="Interview rounds")
    recommendation: str = Field(description="One sentence recommendation")

# ── State ──────────────────────────────────────────────────

class AgentState(TypedDict):
    company: str
    role: str
    research_data: str
    profile: dict
    status: str

# ── Nodes ──────────────────────────────────────────────────

def research_node(state: AgentState) -> AgentState:
    print(f"\n🔍 Researching {state['company']} {state['role']}...")

    prompt = ChatPromptTemplate.from_template(
        "Research the interview process for {role} at {company}. "
        "Include difficulty, rounds, and key topics to prepare."
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "company": state["company"],
        "role": state["role"]
    })

    # Remove <think> tags if present
    if "<think>" in result:
        result = result.split("</think>")[-1].strip()

    print(f"   ✅ Research complete!")
    return {"research_data": result}

def profile_node(state: AgentState) -> AgentState:
    print(f"\n🧠 Creating profile for {state['company']}...")

    json_parser = JsonOutputParser(pydantic_object=CompanyProfile)

    prompt = ChatPromptTemplate.from_template("""
Based on this research about {company} for {role}:
{research_data}

{format_instructions}
""")

    chain = prompt | llm | parser

    try:
        raw = chain.invoke({
            "company": state["company"],
            "role": state["role"],
            "research_data": state["research_data"][:2000],
            "format_instructions": json_parser.get_format_instructions()
        })

        # Handle think tags
        if isinstance(raw, str):
            if "<think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            # Find JSON in the string
            start = raw.find("{")
            end = raw.rfind("}") + 1
            result = json.loads(raw[start:end])
        else:
            result = raw

        print(f"   ✅ Profile created!")
        return {"profile": result, "status": "complete"}

    except Exception as e:
        print(f"   ⚠️ Profile creation failed: {e}")
        return {"profile": {}, "status": "error"}

# ── Build Graph ────────────────────────────────────────────

graph_builder = StateGraph(AgentState)
graph_builder.add_node("research_node", research_node)
graph_builder.add_node("profile_node", profile_node)
graph_builder.add_edge(START, "research_node")
graph_builder.add_edge("research_node", "profile_node")
graph_builder.add_edge("profile_node", END)

# SQLite checkpointer — saves to file
conn = sqlite3.connect("week4/agent_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = graph_builder.compile(checkpointer=checkpointer)

# ── Research a Company ─────────────────────────────────────

def research_company(company, role, thread_id):
    print(f"\n{'='*60}")
    print(f"🎯 Researching: {company} {role}")
    print(f"   Thread ID: {thread_id}")
    print(f"{'='*60}")

    config = {"configurable": {"thread_id": thread_id}}

    # Check if already researched
    existing = checkpointer.get(config)
    if existing:
        state = existing.get("channel_values", {})
        if state.get("status") == "complete":
            print(f"   ✅ Already researched! Loading from database...")
            return state

    # Run graph
    result = graph.invoke({
        "company": company,
        "role": role,
        "research_data": "",
        "profile": {},
        "status": ""
    }, config=config)

    return result

# ── Compare Companies ──────────────────────────────────────

def compare_companies(results):
    print(f"\n{'='*60}")
    print("📊 COMPANY COMPARISON")
    print(f"{'='*60}")

    profiles = []
    for company, result in results.items():
        profile = result.get("profile", {})
        if profile:
            profiles.append(profile)
            print(f"\n🏢 {profile.get('company_name', company)}")
            print(f"   Difficulty:  {profile.get('difficulty', 'N/A')}")
            print(f"   Rounds:      {', '.join(profile.get('rounds', []))}")
            print(f"   Key Topics:  {', '.join(profile.get('key_topics', [])[:3])}")
            print(f"   Tip:         {profile.get('recommendation', 'N/A')}")

    # Ask LLM to compare
    if len(profiles) >= 2:
        print(f"\n{'='*60}")
        print("🤖 AI RECOMMENDATION")
        print(f"{'='*60}")

        prompt = ChatPromptTemplate.from_template("""
Compare these company profiles and recommend which is best for someone
who prefers system design interviews and wants strong compensation:

{profiles}

Give a clear recommendation with reasoning in 3-4 sentences.
Do not include any thinking tags or extra explanation.
""")
        chain = prompt | llm | parser
        recommendation = chain.invoke({
            "profiles": json.dumps(profiles, indent=2)
        })

        # Remove think tags
        if "<think>" in recommendation:
            recommendation = recommendation.split("</think>")[-1].strip()

        print(recommendation)

# ── Main ───────────────────────────────────────────────────

print("🚀 PERSISTENCE & MULTI-COMPANY DEMO")
print("Researching 3 companies and saving to SQLite...\n")

# Research 3 companies in separate threads
companies = [
    ("Google", "SDE-1", "thread_google"),
    ("Amazon", "SDE-1", "thread_amazon"),
    ("Microsoft", "SDE-1", "thread_microsoft"),
]

results = {}
for company, role, thread_id in companies:
    results[company] = research_company(company, role, thread_id)

# Compare all 3
compare_companies(results)

print(f"\n✅ All data saved to week4/agent_memory.db")
print(f"   Run again — it will load from database instead of researching again!")