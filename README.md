# 🎯 Placement Prep Agent

![CI Pipeline](https://github.com/jathin172002-mtech/placement-prep-agent/actions/workflows/ci.yml/badge.svg)

An AI-powered web application that helps engineering students prepare for tech company interviews.

## 🚀 Live Demo
👉 [placement-prep-agent.streamlit.app](https://placement-prep-agent.streamlit.app)

## What it does
Give it a company name and role and it will:
- 🔍 Search the web for real interview experiences
- 🧠 Generate targeted interview questions
- 💬 Let you have a follow-up conversation
- 📊 Track all companies across sessions
- 🔒 Rate limiting and security hardening

## Architecture

User Input → Streamlit UI → Multi-Agent System → Response to User

**Agents:**
- ResearchAgent → Tavily web search
- QuestionAgent → generates interview questions
- FeedbackAgent → evaluates your answers

**Storage:**
- ChromaDB Vector Database → stores company research
- Session state → caches results

## Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Groq API | LLM (qwen3-32b) |
| LangChain | LLM framework |
| LangGraph | Agent state machines |
| ChromaDB | Vector database |
| Tavily | Web search |
| Streamlit | Web UI |
| Docker | Containerization |
| GitHub Actions | CI/CD pipeline |
| Sentry | Error monitoring |

## Setup

### 1. Clone the repository
git clone https://github.com/jathin172002-mtech/placement-prep-agent
cd placement-prep-agent

### 2. Install dependencies
pip install -r requirements.txt

### 3. Set up environment variables
Create a .env file with:
GROQ_API_KEY=your-groq-key
TAVILY_API_KEY=your-tavily-key
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_TRACING=false

### 4. Run the app
python -m streamlit run week5/app.py

### 5. Run with Docker
docker compose up

## Project Structure
week1 - Raw API calls
week2 - LangChain chains
week3 - ChromaDB and RAG
week4 - LangGraph agents
week5 - Multi-agent and Streamlit UI
week6 - Tests and Documentation
.github - CI/CD workflows
Dockerfile - Container setup
docker-compose.yml
requirements.txt

## Running Tests
python -m pytest week6/ -v