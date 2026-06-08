from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# ── Step 1: Define State ───────────────────────────────────
# State is the data that flows through the graph
# Every node reads from state and writes to state

class AgentState(TypedDict):
    company: str          # company name
    role: str             # job role
    research_data: str    # raw research results
    company_profile: str  # synthesized profile
    questions: str        # generated questions
    errors: list          # any errors
    retry_count: int      # how many times we retried

# ── Step 2: Define Nodes ───────────────────────────────────
# Each node is a function that takes state and returns updated state

def metadata_node(state: AgentState) -> AgentState:
    print(f"\n📋 Node 1: Getting metadata for {state['company']}...")

    prompt = ChatPromptTemplate.from_template(
        "Give me basic facts about {company} in 2 sentences. "
        "Include when it was founded and what it does."
    )
    chain = prompt | llm | parser
    result = chain.invoke({"company": state["company"]})

    print(f"   ✅ Got metadata")
    return {"research_data": result}

def research_node(state: AgentState) -> AgentState:
    print(f"\n🔍 Node 2: Researching {state['company']} {state['role']}...")

    prompt = ChatPromptTemplate.from_template(
        "What are the key interview topics and process for "
        "{role} role at {company}? Be specific."
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "company": state["company"],
        "role": state["role"]
    })

    # Append to existing research data
    existing = state.get("research_data", "")
    combined = existing + "\n\n" + result

    print(f"   ✅ Research done (attempt {state.get('retry_count', 0) + 1})")
    return {
        "research_data": combined,
        "retry_count": state.get("retry_count", 0) + 1
    }

def synthesize_node(state: AgentState) -> AgentState:
    print(f"\n🧠 Node 3: Synthesizing company profile...")

    prompt = ChatPromptTemplate.from_template("""
Based on this research about {company} for {role} role:
{research_data}

Create a structured profile with:
1. Interview rounds
2. Key topics to prepare
3. Difficulty level
4. Top 3 tips
""")
    chain = prompt | llm | parser
    result = chain.invoke({
        "company": state["company"],
        "role": state["role"],
        "research_data": state["research_data"]
    })

    print(f"   ✅ Profile synthesized")
    return {"company_profile": result}

def question_node(state: AgentState) -> AgentState:
    print(f"\n❓ Node 4: Generating interview questions...")

    prompt = ChatPromptTemplate.from_template("""
Based on this profile for {company} {role}:
{company_profile}

Generate 5 specific interview questions.
Mix DSA, system design, and behavioral questions.
""")
    chain = prompt | llm | parser
    result = chain.invoke({
        "company": state["company"],
        "role": state["role"],
        "company_profile": state["company_profile"]
    })

    print(f"   ✅ Questions generated")
    return {"questions": result}

# ── Step 3: Define Router ──────────────────────────────────
# Router decides which node to go to next

def research_router(state: AgentState) -> str:
    retry_count = state.get("retry_count", 0)
    research_data = state.get("research_data", "")

    # If too many retries → end with error
    if retry_count >= 3:
        print(f"\n⚠️  Max retries reached!")
        return "end"

    # If not enough data → research again
    if len(research_data) < 200:
        print(f"\n🔄 Not enough data, searching again...")
        return "research_again"

    # Enough data → proceed
    print(f"\n✅ Enough data found, proceeding...")
    return "proceed"

# ── Step 4: Build the Graph ────────────────────────────────

graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("metadata_node", metadata_node)
graph_builder.add_node("research_node", research_node)
graph_builder.add_node("synthesize_node", synthesize_node)
graph_builder.add_node("question_node", question_node)

# Add edges
graph_builder.add_edge(START, "metadata_node")
graph_builder.add_edge("metadata_node", "research_node")

# Conditional edge — router decides next step
graph_builder.add_conditional_edges(
    "research_node",
    research_router,
    {
        "research_again": "research_node",  # loop back
        "proceed": "synthesize_node",        # move forward
        "end": END                           # stop
    }
)

graph_builder.add_edge("synthesize_node", "question_node")
graph_builder.add_edge("question_node", END)

# Compile the graph
graph = graph_builder.compile()

# ── Step 5: Run the Graph ──────────────────────────────────

print("=" * 60)
print("🎯 LANGGRAPH — PLACEMENT PREP AGENT")
print("=" * 60)

company = input("Enter company: ")
role = input("Enter role: ")

# Initial state
initial_state = {
    "company": company,
    "role": role,
    "research_data": "",
    "company_profile": "",
    "questions": "",
    "errors": [],
    "retry_count": 0
}

# Run graph
print(f"\n🚀 Running graph for {company} {role}...")
final_state = graph.invoke(initial_state)

# Show results
print("\n" + "=" * 60)
print("📊 COMPANY PROFILE")
print("=" * 60)
print(final_state["company_profile"])

print("\n" + "=" * 60)
print("❓ INTERVIEW QUESTIONS")
print("=" * 60)
print(final_state["questions"])
