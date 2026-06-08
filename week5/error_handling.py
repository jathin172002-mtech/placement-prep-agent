import json
import time
import logging
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

llm = ChatGroq(model="qwen/qwen3-32b")
parser = StrOutputParser()

# State

class AgentState(TypedDict):
    company: str
    role: str
    research_data: str
    errors: list
    retry_count: int
    status: str
    timestamp: str

# Helper

def clean(text):
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    return text

# Retry decorator

def with_retry(func, max_retries=3, base_delay=1):
    def wrapper(*args, **kwargs):
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                wait_time = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed: {str(e)[:50]}"
                    f" | Retrying in {wait_time}s..."
                )
                if attempt == max_retries:
                    raise e
                time.sleep(wait_time)
    return wrapper

# Nodes

def research_node(state: AgentState) -> AgentState:
    logger.info(f"research_node: Starting for {state['company']} {state['role']}")
    errors = state.get("errors", [])

    try:
        # Simulate occasional failure
        retry_count = state.get("retry_count", 0)

        prompt = ChatPromptTemplate.from_template(
            "Research interview process for {role} at {company} in 3 bullet points."
        )

        # Use retry wrapper on the API call
        def make_api_call():
            chain = prompt | llm | parser
            return chain.invoke({
                "company": state["company"],
                "role": state["role"]
            })

        result = with_retry(make_api_call, max_retries=3, base_delay=1)()
        result = clean(result)

        logger.info(f"research_node: Success! Got {len(result)} chars")
        return {
            "research_data": result,
            "retry_count": retry_count + 1,
            "status": "research_complete"
        }

    except Exception as e:
        error_msg = f"research_node failed: {str(e)[:100]}"
        logger.error(error_msg)
        errors.append({
            "node": "research_node",
            "error": error_msg,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        return {
            "errors": errors,
            "status": "error"
        }

def synthesis_node(state: AgentState) -> AgentState:
    logger.info("synthesis_node: Synthesizing research...")
    errors = state.get("errors", [])

    try:
        if not state.get("research_data"):
            raise ValueError("No research data available!")

        prompt = ChatPromptTemplate.from_template(
            "Based on this research:\n{research}\n\n"
            "Give 3 specific interview tips for {role} at {company}."
        )

        def make_api_call():
            chain = prompt | llm | parser
            return chain.invoke({
                "company": state["company"],
                "role": state["role"],
                "research": state["research_data"]
            })

        result = with_retry(make_api_call, max_retries=3, base_delay=1)()
        result = clean(result)

        logger.info("synthesis_node: Success!")
        return {
            "research_data": state["research_data"] + "\n\nTIPS:\n" + result,
            "status": "complete"
        }

    except Exception as e:
        error_msg = f"synthesis_node failed: {str(e)[:100]}"
        logger.error(error_msg)
        errors.append({
            "node": "synthesis_node",
            "error": error_msg,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        return {
            "errors": errors,
            "status": "error"
        }

def error_handler_node(state: AgentState) -> AgentState:
    errors = state.get("errors", [])
    logger.error(f"error_handler_node: Handling {len(errors)} error(s)")

    print("\n ERROR REPORT")
    print("="*50)
    for i, error in enumerate(errors):
        print(f"\nError {i+1}:")
        print(f"  Node:      {error.get('node')}")
        print(f"  Time:      {error.get('timestamp')}")
        print(f"  Message:   {error.get('error')}")

    return {"status": "handled"}

def display_node(state: AgentState) -> AgentState:
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Company:  {state['company']}")
    print(f"Role:     {state['role']}")
    print(f"Status:   {state['status']}")
    print(f"\nResearch:\n{state['research_data']}")

    if state.get("errors"):
        print(f"\nWarnings: {len(state['errors'])} error(s) occurred but were handled")

    return {}

# Router

def status_router(state: AgentState) -> str:
    status = state.get("status", "")
    if status == "error":
        return "error"
    return "continue"

def research_router(state: AgentState) -> str:
    status = state.get("status", "")
    if status == "error":
        return "error"
    return "continue"

# Build Graph

graph_builder = StateGraph(AgentState)

graph_builder.add_node("research_node", research_node)
graph_builder.add_node("synthesis_node", synthesis_node)
graph_builder.add_node("error_handler_node", error_handler_node)
graph_builder.add_node("display_node", display_node)

graph_builder.add_edge(START, "research_node")

graph_builder.add_conditional_edges(
    "research_node",
    research_router,
    {
        "continue": "synthesis_node",
        "error": "error_handler_node"
    }
)

graph_builder.add_conditional_edges(
    "synthesis_node",
    status_router,
    {
        "continue": "display_node",
        "error": "error_handler_node"
    }
)

graph_builder.add_edge("error_handler_node", END)
graph_builder.add_edge("display_node", END)

graph = graph_builder.compile()

# Test 1 - Normal run

print("="*60)
print("TEST 1: Normal Run")
print("="*60)

result1 = graph.invoke({
    "company": "Google",
    "role": "SDE-1",
    "research_data": "",
    "errors": [],
    "retry_count": 0,
    "status": "",
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
})

# Test 2 - Forced error

print("\n" + "="*60)
print("TEST 2: Forced Error (empty company)")
print("="*60)

result2 = graph.invoke({
    "company": "",
    "role": "",
    "research_data": "",
    "errors": [],
    "retry_count": 0,
    "status": "",
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
})
