import json
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b")
parser = StrOutputParser()

# ── Pydantic Models ────────────────────────────────────────

class Question(BaseModel):
    text: str = Field(description="The question text")
    category: Literal["DSA", "System Design", "Behavioral"] = Field(
        description="Question category"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Question difficulty"
    )

class QuestionSet(BaseModel):
    company: str = Field(description="Company name")
    role: str = Field(description="Job role")
    questions: list[Question] = Field(description="List of 5 questions")

class EvaluationReport(BaseModel):
    relevance_score: int = Field(description="Relevance score 1-10")
    difficulty_score: int = Field(description="Difficulty spread score 1-10")
    coverage_score: int = Field(description="Topic coverage score 1-10")
    overall_score: int = Field(description="Overall score 1-10")
    feedback: str = Field(description="Detailed feedback")
    approved: bool = Field(description="Whether questions are approved")

# ── State ──────────────────────────────────────────────────

class AgentState(TypedDict):
    company: str
    role: str
    question_set: dict
    evaluation: dict
    status: str

# ── Helper ─────────────────────────────────────────────────

def clean_and_parse(text):
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])

# ── Question Agent ─────────────────────────────────────────

def question_agent(state: AgentState) -> AgentState:
    company = state["company"]
    role = state["role"]
    print(f"\n❓ QuestionAgent: Generating questions for {company} {role}...")

    json_parser = JsonOutputParser(pydantic_object=QuestionSet)

    prompt = ChatPromptTemplate.from_template("""
Generate exactly 5 interview questions for {role} at {company}.

Rules:
- Mix DSA, System Design, and Behavioral categories
- Mix easy, medium, and hard difficulties
- Each question must be specific and relevant

{format_instructions}
""")

    chain = prompt | llm | parser
    raw = chain.invoke({
        "company": company,
        "role": role,
        "format_instructions": json_parser.get_format_instructions()
    })

    result = clean_and_parse(raw)
    print(f"   Generated {len(result.get('questions', []))} questions")

    # Print each question
    for i, q in enumerate(result.get("questions", [])):
        if isinstance(q, dict):
            print(f"   {i+1}. [{q.get('category')} - {q.get('difficulty')}] {q.get('text', '')[:60]}...")

    return {"question_set": result}

# ── Evaluator Agent ────────────────────────────────────────

def evaluator_agent(state: AgentState) -> AgentState:
    question_set = state["question_set"]
    company = state["company"]
    role = state["role"]
    print(f"\n🔍 EvaluatorAgent: Grading questions...")

    json_parser = JsonOutputParser(pydantic_object=EvaluationReport)

    prompt = ChatPromptTemplate.from_template("""
Evaluate these interview questions for {role} at {company}:

Questions:
{questions}

Grade on:
1. Relevance (1-10): Are questions relevant to the role?
2. Difficulty Spread (1-10): Good mix of easy/medium/hard?
3. Topic Coverage (1-10): Good mix of DSA/System Design/Behavioral?
4. Overall Score (1-10): Overall quality

Also decide if questions are approved (score >= 7 overall).

{format_instructions}
""")

    chain = prompt | llm | parser
    raw = chain.invoke({
        "company": company,
        "role": role,
        "questions": json.dumps(question_set.get("questions", []), indent=2),
        "format_instructions": json_parser.get_format_instructions()
    })

    result = clean_and_parse(raw)
    print(f"   Evaluation complete!")
    print(f"   Overall score: {result.get('overall_score')}/10")
    print(f"   Approved: {result.get('approved')}")

    return {
        "evaluation": result,
        "status": "approved" if result.get("approved") else "rejected"
    }

# ── Display Results ────────────────────────────────────────

def display_node(state: AgentState) -> AgentState:
    question_set = state["question_set"]
    evaluation = state["evaluation"]

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    print(f"\nCompany: {state['company']} | Role: {state['role']}")
    print(f"Status: {state['status'].upper()}")

    print("\n--- QUESTIONS ---")
    for i, q in enumerate(question_set.get("questions", [])):
        if isinstance(q, dict):
            print(f"\n{i+1}. [{q.get('category')} - {q.get('difficulty')}]")
            print(f"   {q.get('text', '')}")

    print("\n--- EVALUATION ---")
    print(f"Relevance:    {evaluation.get('relevance_score')}/10")
    print(f"Difficulty:   {evaluation.get('difficulty_score')}/10")
    print(f"Coverage:     {evaluation.get('coverage_score')}/10")
    print(f"Overall:      {evaluation.get('overall_score')}/10")
    print(f"Feedback:     {evaluation.get('feedback')}")

    return {}

# ── Build Graph ────────────────────────────────────────────

graph_builder = StateGraph(AgentState)

graph_builder.add_node("question_agent", question_agent)
graph_builder.add_node("evaluator_agent", evaluator_agent)
graph_builder.add_node("display_node", display_node)

graph_builder.add_edge(START, "question_agent")
graph_builder.add_edge("question_agent", "evaluator_agent")
graph_builder.add_edge("evaluator_agent", "display_node")
graph_builder.add_edge("display_node", END)

graph = graph_builder.compile()

# ── Run ────────────────────────────────────────────────────

print("="*60)
print("STRUCTURED AGENTS DEMO")
print("="*60)

result = graph.invoke({
    "company": "Google",
    "role": "SDE-1",
    "question_set": {},
    "evaluation": {},
    "status": ""
})
