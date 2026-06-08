import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# LangSmith is enabled automatically via .env variables
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=your-key
# LANGSMITH_PROJECT=placement-prep-agent

print("LangSmith Project:", os.getenv("LANGSMITH_PROJECT"))
print("LangSmith Tracing:", os.getenv("LANGSMITH_TRACING"))

llm = ChatGroq(model="qwen/qwen3-32b")
parser = StrOutputParser()
search = TavilySearch(max_results=2)

# State

class AgentState(TypedDict):
    company: str
    role: str
    research: str
    questions: str

# Helper

def clean(text):
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    return text

# Nodes

def research_node(state: AgentState) -> AgentState:
    print(f"\n🔍 Researching {state['company']}...")

    results = search.invoke(f"{state['company']} {state['role']} interview")
    content = ""
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                content += r.get("content", "") + "\n"

    prompt = ChatPromptTemplate.from_template(
        "Summarize this interview info for {role} at {company} in 3 points:\n{content}"
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "company": state["company"],
        "role": state["role"],
        "content": content[:1500]
    })

    print("   Research done!")
    return {"research": clean(result)}

def question_node(state: AgentState) -> AgentState:
    print(f"\n❓ Generating questions...")

    prompt = ChatPromptTemplate.from_template(
        "Based on this research:\n{research}\n\n"
        "Generate 3 interview questions for {role} at {company}."
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "company": state["company"],
        "role": state["role"],
        "research": state["research"]
    })

    print("   Questions generated!")
    return {"questions": clean(result)}

# Build Graph

graph_builder = StateGraph(AgentState)
graph_builder.add_node("research_node", research_node)
graph_builder.add_node("question_node", question_node)
graph_builder.add_edge(START, "research_node")
graph_builder.add_edge("research_node", "question_node")
graph_builder.add_edge("question_node", END)

graph = graph_builder.compile()

# Run

print("\n" + "="*60)
print("LANGSMITH TRACING DEMO")
print("="*60)

result = graph.invoke({
    "company": "Flipkart",
    "role": "SDE-1",
    "research": "",
    "questions": ""
})

print("\n--- RESEARCH ---")
print(result["research"])

print("\n--- QUESTIONS ---")
print(result["questions"])

print("\n" + "="*60)
print("Check your LangSmith dashboard at smith.langchain.com")
print("Project: placement-prep-agent")
print("You should see this run traced there!")
print("="*60)
