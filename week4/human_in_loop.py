from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b")
parser = StrOutputParser()

# ── State ──────────────────────────────────────────────────

class AgentState(TypedDict):
    company: str
    role: str
    research_data: str
    focus_area: str        # set by human input
    questions: str
    status: str

# ── Nodes ──────────────────────────────────────────────────

def research_node(state: AgentState) -> AgentState:
    print(f"\n🔍 Researching {state['company']} {state['role']}...")

    prompt = ChatPromptTemplate.from_template(
        "Research the interview process for {role} at {company}. "
        "Cover DSA, System Design, and Behavioral aspects."
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "company": state["company"],
        "role": state["role"]
    })

    print(f"   ✅ Research complete!")
    return {"research_data": result}

def question_node(state: AgentState) -> AgentState:
    focus = state.get("focus_area", "general")
    print(f"\n❓ Generating {focus} questions...")

    prompt = ChatPromptTemplate.from_template("""
Based on this research about {company} {role}:
{research_data}

Generate 5 {focus_area} specific interview questions.
Make them challenging and relevant.
""")
    chain = prompt | llm | parser
    result = chain.invoke({
        "company": state["company"],
        "role": state["role"],
        "research_data": state["research_data"],
        "focus_area": focus
    })

    print(f"   ✅ Questions generated!")
    return {"questions": result, "status": "complete"}

# ── Build Graph with Interrupt ─────────────────────────────

graph_builder = StateGraph(AgentState)

graph_builder.add_node("research_node", research_node)
graph_builder.add_node("question_node", question_node)

graph_builder.add_edge(START, "research_node")
graph_builder.add_edge("research_node", "question_node")
graph_builder.add_edge("question_node", END)

# MemorySaver needed for interrupts
checkpointer = MemorySaver()

# interrupt_before pauses BEFORE this node runs
graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["question_node"]
)

# ── Run with Human Input ───────────────────────────────────

print("=" * 60)
print("🎯 HUMAN IN THE LOOP DEMO")
print("=" * 60)

company = input("Enter company: ")
role = input("Enter role: ")

# Thread ID — identifies this specific run
config = {"configurable": {"thread_id": "session_1"}}

initial_state = {
    "company": company,
    "role": role,
    "research_data": "",
    "focus_area": "",
    "questions": "",
    "status": ""
}

# ── Phase 1: Run until interrupt ───────────────────────────
print(f"\n🚀 Phase 1: Running until interrupt...")
graph.invoke(initial_state, config=config)

# Graph paused before question_node!
print("\n" + "=" * 60)
print("⏸️  AGENT PAUSED — Waiting for your input!")
print("=" * 60)
print("What should the agent focus on?")
print("1. DSA")
print("2. System Design")
print("3. Behavioral")
print("4. General")

choice = input("\nYour choice (type the focus area): ").strip()

# ── Phase 2: Update state and resume ──────────────────────
print(f"\n▶️  Phase 2: Resuming with focus: {choice}")

# Update state with human input
graph.update_state(
    config,
    {"focus_area": choice}
)

# Resume from where it paused
final_state = graph.invoke(None, config=config)

# ── Show Results ───────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)
print(f"Company:    {company}")
print(f"Role:       {role}")
print(f"Focus Area: {choice}")
print(f"\n❓ Questions:")
print(final_state["questions"])
