import os
import sentry_sdk

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=1.0,
)

import streamlit as st
import json
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b")
parser = StrOutputParser()
search = TavilySearch(max_results=3)

# Rate limiting constants
MAX_REQUESTS_PER_SESSION = 10
MAX_TOKENS_PER_SESSION = 50000
COST_PER_TOKEN = 0.00000059

class CompanyProfile(BaseModel):
    company_name: str = Field(description="Company name")
    difficulty: Literal["low", "medium", "high"] = Field(description="Interview difficulty")
    rounds: list[str] = Field(description="Interview rounds")
    key_topics: list[str] = Field(description="Key topics to prepare")
    tips: list[str] = Field(description="Top 3 preparation tips")

def clean(text):
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    return text

def parse_json(text):
    text = clean(text)
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])

def estimate_tokens(text):
    return len(text) // 4

def research_company(company, role):
    query = f"{company} {role} interview process 2024"
    results = search.invoke(query)
    content = ""
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                content += r.get("content", "") + "\n\n"
    json_parser = JsonOutputParser(pydantic_object=CompanyProfile)
    prompt = ChatPromptTemplate.from_template(
        "Research {company} {role} interview.\n"
        "Content: {content}\n\n"
        "{format_instructions}"
    )
    chain = prompt | llm | parser
    raw = chain.invoke({
        "company": company,
        "role": role,
        "content": content[:2000],
        "format_instructions": json_parser.get_format_instructions()
    })
    tokens = estimate_tokens(content[:2000] + raw)
    st.session_state.total_tokens += tokens
    return parse_json(raw)

@st.cache_data
def generate_questions(company, role, profile_str):
    prompt = ChatPromptTemplate.from_template(
        "Generate 10 interview questions for {role} at {company}.\n"
        "Profile: {profile}\n"
        "Mix DSA, System Design, and Behavioral. Number them 1-10."
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "company": company,
        "role": role,
        "profile": profile_str
    })
    tokens = estimate_tokens(profile_str + result)
    st.session_state.total_tokens += tokens
    return clean(result)

def chat_with_agent(company, role, question, history):
    history_text = "\n".join([
        f"Human: {m.content}" if isinstance(m, HumanMessage)
        else f"Assistant: {m.content}"
        for m in history[-6:]
    ])
    prompt = ChatPromptTemplate.from_template(
        "You are an expert on {company} {role} interviews.\n"
        "Conversation history:\n{history}\n\n"
        "Answer this question: {question}"
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "company": company,
        "role": role,
        "history": history_text,
        "question": question
    })
    tokens = estimate_tokens(history_text + question + result)
    st.session_state.total_tokens += tokens
    return clean(result)

st.set_page_config(
    page_title="Placement Prep Agent",
    page_icon="🎯",
    layout="wide"
)

if "researched_companies" not in st.session_state:
    st.session_state.researched_companies = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "current_role" not in st.session_state:
    st.session_state.current_role = None
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

st.sidebar.title("Placement Prep Agent")
page = st.sidebar.radio("Navigate", ["Research", "Chat", "My Companies"])

if st.session_state.current_company:
    st.sidebar.success(
        f"Current: {st.session_state.current_company} "
        f"{st.session_state.current_role}"
    )

# Cost dashboard in sidebar
st.sidebar.divider()
st.sidebar.subheader("📊 Usage Dashboard")
tokens_used = st.session_state.total_tokens
cost = tokens_used * COST_PER_TOKEN
st.sidebar.write(f"🔢 Tokens used: {tokens_used:,}")
st.sidebar.write(f"💰 Est. cost: ${cost:.4f}")
st.sidebar.write(f"🔍 Requests: {st.session_state.request_count}/{MAX_REQUESTS_PER_SESSION}")
st.sidebar.progress(min(tokens_used / MAX_TOKENS_PER_SESSION, 1.0))

if page == "Research":
    st.title("Research a Company")
    st.write("Enter a company and role to get a complete interview preparation guide.")

    col1, col2 = st.columns(2)
    with col1:
        company = st.text_input("Company Name", placeholder="e.g. Google")
    with col2:
        role = st.text_input("Role", placeholder="e.g. SDE-1")

    if st.button("Research", type="primary"):
        if not company or not role:
            st.error("Please enter both company and role!")
        elif st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
            st.error("⚠️ You've reached the maximum of 10 research requests per session. Please refresh to start a new session.")
        elif st.session_state.total_tokens >= MAX_TOKENS_PER_SESSION:
            st.error("⚠️ Token limit reached for this session. Please refresh to start a new session.")
        else:
            key = f"{company}_{role}"
            if key in st.session_state.researched_companies:
                st.info("Loaded from cache!")
                data = st.session_state.researched_companies[key]
            else:
                with st.status(f"Researching {company} {role}...") as status:
                    st.write("Searching the web...")
                    try:
                        profile = research_company(company, role)
                        st.write("Generating questions...")
                        questions = generate_questions(company, role, json.dumps(profile))
                        status.update(label="Research complete!", state="complete")
                        data = {"profile": profile, "questions": questions}
                        st.session_state.researched_companies[key] = data
                        st.session_state.request_count += 1
                    except Exception as e:
                        sentry_sdk.capture_exception(e)
                        st.error(f"Error: {str(e)}")
                        st.stop()

            st.session_state.current_company = company
            st.session_state.current_role = role
            st.session_state.chat_history = []

            profile = data["profile"]
            questions = data["questions"]

            st.subheader(f"{company} - {role} Profile")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Difficulty", profile.get("difficulty", "N/A").upper())
            with col2:
                st.metric("Interview Rounds", len(profile.get("rounds", [])))
            with col3:
                st.metric("Key Topics", len(profile.get("key_topics", [])))

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Interview Rounds")
                for i, round_name in enumerate(profile.get("rounds", []), 1):
                    st.write(f"{i}. {round_name}")
                st.subheader("Key Topics")
                for topic in profile.get("key_topics", []):
                    st.write(f"• {topic}")
            with col2:
                st.subheader("Preparation Tips")
                for i, tip in enumerate(profile.get("tips", []), 1):
                    st.info(f"Tip {i}: {tip}")

            st.subheader("Interview Questions")
            st.write(questions)
            st.success(f"Go to Chat page to ask follow-up questions about {company}!")

elif page == "Chat":
    st.title("Chat with Your Agent")

    if not st.session_state.current_company:
        st.warning("Please research a company first on the Research page!")
        st.stop()

    company = st.session_state.current_company
    role = st.session_state.current_role

    st.subheader(f"Asking about: {company} - {role}")

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

    if question := st.chat_input("Ask anything about this company's interviews..."):
        st.session_state.chat_history.append(HumanMessage(content=question))
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chat_with_agent(
                        company, role, question,
                        st.session_state.chat_history
                    )
                except Exception as e:
                    sentry_sdk.capture_exception(e)
                    response = f"Error: {str(e)}"
            st.write(response)
            st.session_state.chat_history.append(AIMessage(content=response))

elif page == "My Companies":
    st.title("My Researched Companies")

    if not st.session_state.researched_companies:
        st.warning("No companies researched yet. Go to Research page!")
        st.stop()

    companies = st.session_state.researched_companies
    st.write(f"You have researched {len(companies)} companies:")

    for key, data in companies.items():
        profile = data.get("profile", {})
        company_name = profile.get("company_name", key.split("_")[0])

        with st.expander(f"{company_name} - {key.split('_', 1)[1] if '_' in key else ''}"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Difficulty", profile.get("difficulty", "N/A").upper())
                st.write("**Rounds:**")
                for r in profile.get("rounds", []):
                    st.write(f"• {r}")
            with col2:
                st.write("**Key Topics:**")
                for t in profile.get("key_topics", []):
                    st.write(f"• {t}")
            st.write("**Tips:**")
            for tip in profile.get("tips", []):
                st.info(tip)

    if len(companies) > 1:
        st.subheader("Comparison Table")
        comparison_data = []
        for key, data in companies.items():
            profile = data.get("profile", {})
            comparison_data.append({
                "Company": profile.get("company_name", key),
                "Difficulty": profile.get("difficulty", "N/A"),
                "Rounds": len(profile.get("rounds", [])),
                "Topics": len(profile.get("key_topics", []))
            })
        st.table(comparison_data)

        if st.button("Get AI Recommendation"):
            with st.spinner("Analyzing companies..."):
                profiles = [d.get("profile", {}) for d in companies.values()]
                prompt = ChatPromptTemplate.from_template(
                    "Compare these companies and recommend which to apply first "
                    "for someone who prefers system design:\n{profiles}\n"
                    "Give a clear recommendation in 3-4 sentences."
                )
                chain = prompt | llm | parser
                try:
                    recommendation = chain.invoke({
                        "profiles": json.dumps(profiles, indent=2)
                    })
                except Exception as e:
                    sentry_sdk.capture_exception(e)
                    recommendation = f"Error: {str(e)}"
                st.success(clean(recommendation))