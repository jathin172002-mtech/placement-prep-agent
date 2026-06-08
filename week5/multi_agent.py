import json
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b")
parser = StrOutputParser()
search = TavilySearch(max_results=3)

# Pydantic Models

class ResearchResult(BaseModel):
    company: str = Field(description="Company name")
    role: str = Field(description="Job role")
    summary: str = Field(description="Research summary")
    difficulty: Literal["low", "medium", "high"] = Field(description="Interview difficulty")
    key_topics: list[str] = Field(description="Top 5 topics")
    rounds: list[str] = Field(description="Interview rounds")

class QuestionSet(BaseModel):
    questions: list[str] = Field(description="List of 10 interview questions")
    topics_covered: list[str] = Field(description="Topics covered")

class FeedbackReport(BaseModel):
    score: int = Field(description="Score out of 10")
    strengths: list[str] = Field(description="What was good")
    improvements: list[str] = Field(description="What to improve")
    verdict: str = Field(description="Overall verdict")

# State

class SupervisorState(TypedDict):
    user_input: str
    agent_type: str
    company: str
    role: str
    research_result: dict
    questions: dict
    feedback: dict
    answer: str
    final_response: str

# Helper functions

def clean(text):
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    return text

def parse_json(text):
    text = clean(text)
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])

# Supervisor Node

def supervisor_node(state):
    print("\nSupervisor analyzing: " + state["user_input"])

    prompt = ChatPromptTemplate.from_template(
        "You are a supervisor that routes requests to the right agent.\n"
        "User input: {user_input}\n\n"
        "Decide which agent should handle this:\n"
        "- research: if user wants to research a company\n"
        "- questions: if user wants interview questions\n"
        "- feedback: if user wants feedback on their answer\n\n"
        "Also extract company name, role, and answer if provided.\n\n"
        "Respond in JSON only with no extra text:\n"
        "{{\"agent_type\": \"research/questions/feedback\", "
        "\"company\": \"name\", \"role\": \"role\", \"answer\": \"answer or none\"}}"
    )

    chain = prompt | llm | parser
    result = chain.invoke({"user_input": state["user_input"]})
    parsed = parse_json(result)

    print("   Routes to: " + parsed.get("agent_type", "research"))
    print("   Company: " + parsed.get("company", "unknown"))

    return {
        "agent_type": parsed.get("agent_type", "research"),
        "company": parsed.get("company", "unknown"),
        "role": parsed.get("role", "unknown"),
        "answer": parsed.get("answer", "none")
    }

# Research Agent

def research_agent(state):
    company = state["company"]
    role = state["role"]
    print("\nResearchAgent: Researching " + company + " " + role)

    query = company + " " + role + " interview process 2024"
    results = search.invoke(query)

    content = ""
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                content += r.get("content", "") + "\n\n"
    else:
        content = str(results)

    json_parser = JsonOutputParser(pydantic_object=ResearchResult)

    prompt = ChatPromptTemplate.from_template(
        "Based on this research about {company} {role}:\n"
        "{content}\n\n"
        "{format_instructions}"
    )

    chain = prompt | llm | parser
    raw = chain.invoke({
        "company": company,
        "role": role,
        "content": content[:2000],
        "format_instructions": json_parser.get_format_instructions()
    })

    result = parse_json(raw)
    print("   Research complete!")

    rounds_text = "\n".join(["- " + r for r in result.get("rounds", [])])
    topics_text = "\n".join(["- " + t for t in result.get("key_topics", [])])

    response = (
        "\n## " + company + " - " + role + " Research\n\n"
        "**Difficulty:** " + result.get("difficulty", "N/A") + "\n\n"
        "**Rounds:**\n" + rounds_text + "\n\n"
        "**Key Topics:**\n" + topics_text + "\n\n"
        "**Summary:** " + result.get("summary", "N/A")
    )

    return {
        "research_result": result,
        "final_response": response
    }

# Question Agent

def question_agent(state):
    company = state["company"]
    role = state["role"]
    research = state.get("research_result", {})
    print("\nQuestionAgent: Generating questions for " + company + " " + role)

    json_parser = JsonOutputParser(pydantic_object=QuestionSet)

    prompt = ChatPromptTemplate.from_template(
        "Generate 10 challenging interview questions for {role} at {company}.\n"
        "Context: {context}\n"
        "Mix DSA, System Design, and Behavioral questions.\n"
        "{format_instructions}"
    )

    chain = prompt | llm | parser
    raw = chain.invoke({
        "company": company,
        "role": role,
        "context": json.dumps(research) if research else company + " " + role,
        "format_instructions": json_parser.get_format_instructions()
    })

    result = parse_json(raw)
    print("   Questions generated!")

    questions_list = result.get("questions", [])
    questions_text = "\n".join([
        str(i+1) + ". " + q
        for i, q in enumerate(questions_list)
    ])

    response = (
        "\n## Interview Questions - " + company + " " + role + "\n\n"
        + questions_text + "\n\n"
        "**Topics:** " + ", ".join(result.get("topics_covered", []))
    )

    return {
        "questions": result,
        "final_response": response
    }

# Feedback Agent

def feedback_agent(state):
    answer = state.get("answer", "")
    company = state["company"]
    role = state["role"]
    print("\nFeedbackAgent: Evaluating answer...")

    json_parser = JsonOutputParser(pydantic_object=FeedbackReport)

    prompt = ChatPromptTemplate.from_template(
        "Evaluate this interview answer for {role} at {company}:\n"
        "Answer: {answer}\n"
        "Be strict but constructive. Score out of 10.\n"
        "{format_instructions}"
    )

    chain = prompt | llm | parser
    raw = chain.invoke({
        "company": company,
        "role": role,
        "answer": answer,
        "format_instructions": json_parser.get_format_instructions()
    })

    result = parse_json(raw)
    print("   Feedback generated! Score: " + str(result.get("score")) + "/10")

    strengths = "\n".join(["+ " + s for s in result.get("strengths", [])])
    improvements = "\n".join(["* " + i for i in result.get("improvements", [])])

    response = (
        "\n## Feedback Report\n\n"
        "**Score:** " + str(result.get("score")) + "/10\n"
        "**Verdict:** " + result.get("verdict", "") + "\n\n"
        "**Strengths:**\n" + strengths + "\n\n"
        "**Improvements:**\n" + improvements
    )

    return {
        "feedback": result,
        "final_response": response
    }

# Router

def agent_router(state):
    agent_type = state.get("agent_type", "research")
    print("   Routing to: " + agent_type)
    return agent_type

# Build Graph

graph_builder = StateGraph(SupervisorState)

graph_builder.add_node("supervisor_node", supervisor_node)
graph_builder.add_node("research", research_agent)
graph_builder.add_node("questions", question_agent)
graph_builder.add_node("feedback", feedback_agent)

graph_builder.add_edge(START, "supervisor_node")

graph_builder.add_conditional_edges(
    "supervisor_node",
    agent_router,
    {
        "research": "research",
        "questions": "questions",
        "feedback": "feedback"
    }
)

graph_builder.add_edge("research", END)
graph_builder.add_edge("questions", END)
graph_builder.add_edge("feedback", END)

graph = graph_builder.compile()

# Run function

def run(user_input):
    print("\n" + "="*60)
    print("Input: " + user_input)
    print("="*60)

    result = graph.invoke({
        "user_input": user_input,
        "agent_type": "",
        "company": "",
        "role": "",
        "research_result": {},
        "questions": {},
        "feedback": {},
        "answer": "",
        "final_response": ""
    })

    print(result["final_response"])
    return result

# Test all 3 routing paths
run("Research Razorpay SDE-2 interview process")
run("Give me 10 system design questions for Meesho backend engineer")
run("I answered this for Google SDE-1: A hash table uses array with hash function. Time complexity O(1). How did I do?")