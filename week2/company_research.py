import json
import os
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# ── Setup ─────────────────────────────────────────────────

model = ChatGroq(model="llama-3.3-70b-versatile")
search = TavilySearch(max_results=5)

# ── Pydantic Models ───────────────────────────────────────

class CompanyProfile(BaseModel):
    company_name: str = Field(description="Name of the company")
    founded: str = Field(description="Year the company was founded")
    headquarters: str = Field(description="City and country of headquarters")
    tech_stack: list[str] = Field(description="Technologies the company uses")
    known_for: str = Field(description="One sentence about what company is known for")
    interview_rounds: list[str] = Field(description="List of interview rounds")
    key_topics: list[str] = Field(description="Key topics to prepare for interviews")
    difficulty: Literal["low", "medium", "high"] = Field(
        description="Overall interview difficulty"
    )
    tips: list[str] = Field(description="3 specific tips for interview preparation")

class QuestionSet(BaseModel):
    questions: list[str] = Field(description="List of 10 interview questions")
    topics_covered: list[str] = Field(description="Topics these questions cover")

# ── Chain 1: Search ───────────────────────────────────────

def chain1_search(company, role):
    print(f"\n🔍 Chain 1: Searching for {company} {role} information...")

    search_queries = [
        f"{company} {role} interview process 2024",
        f"{company} tech stack engineering",
        f"{company} interview experience {role}"
    ]

    all_results = []
    for query in search_queries:
        print(f"   Searching: {query}")
        results = search.invoke(query)
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    all_results.append(r.get("content", ""))
                else:
                    all_results.append(str(r))
        else:
            all_results.append(str(results))

    combined = "\n\n".join(all_results[:6])
    print(f"   ✅ Found {len(all_results)} results")
    return combined

# ── Chain 2: Synthesize ───────────────────────────────────

def chain2_synthesize(company, role, search_results):
    print(f"\n🧠 Chain 2: Synthesizing company profile...")

    parser = JsonOutputParser(pydantic_object=CompanyProfile)

    prompt = ChatPromptTemplate.from_template("""
You are a placement preparation expert. Based on the search results below,
create a detailed company profile for {company} targeting {role} candidates.

Search Results:
{search_results}

{format_instructions}
""")

    chain = prompt | model | parser

    result = chain.invoke({
        "company": company,
        "role": role,
        "search_results": search_results[:3000],
        "format_instructions": parser.get_format_instructions()
    })

    print(f"   ✅ Profile created for {result['company_name']}")
    return result

# ── Chain 3: Generate Questions ───────────────────────────

def chain3_questions(company, role, profile):
    print(f"\n❓ Chain 3: Generating interview questions...")

    parser = JsonOutputParser(pydantic_object=QuestionSet)

    prompt = ChatPromptTemplate.from_template("""
You are an expert interviewer at {company}.
Generate exactly 10 challenging and relevant interview questions for a {role} candidate.

Company Profile:
- Tech Stack: {tech_stack}
- Key Topics: {key_topics}
- Interview Rounds: {interview_rounds}
- Difficulty: {difficulty}

Mix questions across: DSA, System Design, OOP, Behavioral, and {company} specific topics.

{format_instructions}
""")

    chain = prompt | model | parser

    result = chain.invoke({
        "company": company,
        "role": role,
        "tech_stack": ", ".join(profile["tech_stack"]),
        "key_topics": ", ".join(profile["key_topics"]),
        "interview_rounds": ", ".join(profile["interview_rounds"]),
        "difficulty": profile["difficulty"],
        "format_instructions": parser.get_format_instructions()
    })

    print(f"   ✅ Generated {len(result['questions'])} questions")
    return result

# ── Save Results ──────────────────────────────────────────

def save_results(company, role, profile, questions):
    filename = f"week2/{company}_{role}_profile.json".replace(" ", "_")

    data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "company": company,
        "role": role,
        "profile": profile,
        "questions": questions
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 Saved to {filename}")
    return filename

# ── Main ──────────────────────────────────────────────────

def research_company(company, role):
    print("=" * 60)
    print(f"🎯 COMPANY RESEARCH CHAIN")
    print(f"   Company: {company} | Role: {role}")
    print("=" * 60)

    # Run all 3 chains
    search_results = chain1_search(company, role)
    profile = chain2_synthesize(company, role, search_results)
    questions = chain3_questions(company, role, profile)

    # Save everything
    filename = save_results(company, role, profile, questions)

    # Display results
    print("\n" + "=" * 60)
    print("📊 COMPANY PROFILE")
    print("=" * 60)
    print(f"Company:    {profile['company_name']}")
    print(f"Founded:    {profile['founded']}")
    print(f"HQ:         {profile['headquarters']}")
    print(f"Difficulty: {profile['difficulty']}")
    print(f"Known For:  {profile['known_for']}")
    print(f"\nTech Stack: {', '.join(profile['tech_stack'])}")
    print(f"Key Topics: {', '.join(profile['key_topics'])}")
    print(f"\nTips:")
    for i, tip in enumerate(profile['tips'], 1):
        print(f"  {i}. {tip}")

    print("\n" + "=" * 60)
    print("❓ INTERVIEW QUESTIONS")
    print("=" * 60)
    for i, q in enumerate(questions['questions'], 1):
        print(f"\n{i}. {q}")

    print(f"\n✅ Everything saved to {filename}")
    return profile, questions

# ── Run ───────────────────────────────────────────────────

company = input("\nEnter company name: ")
role = input("Enter role: ")
research_company(company, role)
