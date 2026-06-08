from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# ── State ──────────────────────────────────────────────────

class AgentState(TypedDict):
    company: str
    role: str
    research_data: str
    retry_count: int
    search_queries: list
    status: str

# ── Nodes ──────────────────────────────────────────────────

def research_node(state: AgentState) -> AgentState:
    retry = state.get("retry_count", 0)
    company = state["company"]
    role = state["role"]

    # Each retry uses a broader query
    queries = [
        f"{company} {role} interview process",
        f"{company} software engineer interview tips",
        f"{company} hiring process technical rounds"
    ]

    query = queries[min(retry, len(queries)-1)]
    print(f"\n🔍 Research attempt {retry + 1}: '{query}'")

    prompt = ChatPromptTemplate.from_template(
        "Research this query and provide detailed information: {query}"
    )
    chain = prompt | llm | parser
    result = chain.invoke({"query": query})

    existing = state.get("research_data", "")
    combined = existing + "\n\n" + result

    print(f"   Data length: {len(combined)} characters")

    return {
        "research_data": combined,
        "retry_count": retry + 1,
        "search_queries": state.get("search_queries", []) + [query]
    }

def success_node(state: AgentState) -> AgentState:
    print(f"\n✅ Success! Enough data collected after "
          f"{state['retry_count']} attempt(s)")
    print(f"   Total data: {len(state['research_data'])} characters")
    print(f"   Queries used: {state['search_queries']}")
    return {"status": "success"}

def error_node(state: AgentState) -> AgentState:
    print(f"\n❌ Failed after {state['retry_count']} attempts")
    print(f"   Data collected: {len(state['research_data'])} characters")
    return {"status": "error"}

# ── Router ─────────────────────────────────────────────────

def research_router(state: AgentState) -> str:
    retry_count = state.get("retry_count", 0)
    data_length = len(state.get("research_data", ""))

    print(f"\n🤔 Router checking:")
    print(f"   Retry count: {retry_count}")
    print(f"   Data length: {data_length} characters")
    print(f"   Minimum needed: 500 characters")

    # Too many retries → error
    if retry_count >= 3:
        print(f"   Decision: MAX RETRIES reached → error")
        return "error"

    # Not enough data → retry
    if data_length < 500:
        print(f"   Decision: NOT ENOUGH DATA → retry")
        return "retry"

    # Enough data → success
    print(f"   Decision: ENOUGH DATA → success")
    return "success"

# ── Build Graph ────────────────────────────────────────────

graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("research_node", research_node)
graph_builder.add_node("success_node", success_node)
graph_builder.add_node("error_node", error_node)

# Add edges
graph_builder.add_edge(START, "research_node")

# Conditional edges with loop
graph_builder.add_conditional_edges(
    "research_node",
    research_router,
    {
        "retry": "research_node",    # ← LOOP BACK
        "success": "success_node",
        "error": "error_node"
    }
)

graph_builder.add_edge("success_node", END)
graph_builder.add_edge("error_node", END)

graph = graph_builder.compile()

# ── Test 1: Normal company (should succeed quickly) ────────

print("=" * 60)
print("TEST 1: Normal company — Google")
print("=" * 60)

result1 = graph.invoke({
    "company": "Google",
    "role": "SDE-1",
    "research_data": "",
    "retry_count": 0,
    "search_queries": [],
    "status": ""
})
print(f"\nFinal status: {result1['status']}")
print(f"Retries needed: {result1['retry_count']}")

# ── Test 2: Obscure company (may need more retries) ────────

print("\n" + "=" * 60)
print("TEST 2: Less known company — Zepto")
print("=" * 60)

result2 = graph.invoke({
    "company": "Zepto",
    "role": "Backend Engineer",
    "research_data": "",
    "retry_count": 0,
    "search_queries": [],
    "status": ""
})
print(f"\nFinal status: {result2['status']}")
print(f"Retries needed: {result2['retry_count']}")
