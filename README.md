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
```bash
git clone https://github.com/jathin172002-mtech/placement-prep-agent
cd placement-prep-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file:
### 4. Run the app
```bash
python -m streamlit run week5/app.py
```

### 5. Run with Docker
```bash
docker compose up
```

## Project Structure
## Running Tests
```bash
python -m pytest week6/ -v
```