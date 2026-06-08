from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated
from datetime import datetime
import operator
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b")

# ── Tools ──────────────────────────────────────────────────

@tool
def get_current_date() -> str:
    """Get today's date."""
    return datetime.now().strftime("Today is %A, %B %d, %Y")

@tool
def calculate(expression: str) -> str:
    """Calculate a math expression. Example: calculate('2 + 2')"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_company_info(company: str) -> str:
    """Get interview information about a tech company."""
    companies = {
        "google": "Google: Very hard interviews. Focus on DSA and System Design. 4-5 rounds.",
        "amazon": "Amazon: Hard interviews. Focus on Leadership Principles and DSA. 4-6 rounds.",
        "microsoft": "Microsoft: Medium-hard. Focus on DSA and Problem Solving. 4-5 rounds.",
        "infosys": "Infosys: Medium difficulty. Focus on Aptitude and Basic Coding. 3-4 rounds.",
    }
    key = company.lower().strip()
    return companies.get(key, f"No data found for {company}")

tools = [get_current_date, calculate, get_company_info]
model_with_tools = llm.bind_tools(tools)

# ── State ──────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# ── Nodes ──────────────────────────────────────────────────

def llm_node(state: AgentState) -> AgentState:
    print(f"\n🧠 LLM thinking...")
    messages = state["messages"]
    response = model_with_tools.invoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"   Wants to call: {[t['name'] for t in response.tool_calls]}")
    else:
        print(f"   Has final answer")

    return {"messages": [response]}

def tool_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\n🔧 Calling: {tool_name} with {tool_args}")

        if tool_name == "get_current_date":
            result = get_current_date.invoke({})
        elif tool_name == "calculate":
            result = calculate.invoke(tool_args)
        elif tool_name == "get_company_info":
            result = get_company_info.invoke(tool_args)
        else:
            result = f"Unknown tool: {tool_name}"

        print(f"   Result: {result}")

        tool_results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": tool_results}

# ── Router ─────────────────────────────────────────────────

def router(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# ── Build Graph ────────────────────────────────────────────

graph_builder = StateGraph(AgentState)
graph_builder.add_node("llm_node", llm_node)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_edge(START, "llm_node")
graph_builder.add_conditional_edges(
    "llm_node",
    router,
    {"tools": "tool_node", "end": END}
)
graph_builder.add_edge("tool_node", "llm_node")
graph = graph_builder.compile()

# ── Run ────────────────────────────────────────────────────

def ask(question):
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    result = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })
    print(f"\n✅ Final Answer:")
    print(result["messages"][-1].content)

ask("What is 2847 multiplied by 39?")
ask("What company is easier to crack - Google or Amazon?")
ask("What is today's date?")