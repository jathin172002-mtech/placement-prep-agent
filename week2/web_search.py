from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

# Setup model
model = ChatGroq(model="llama-3.3-70b-versatile")

# Setup Tavily search tool
search = TavilySearch(max_results=3)
tools = [search]

# Create agent using langgraph
agent = create_react_agent(model, tools)

# Run it
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Research Infosys SDE-1 interview process. What rounds are there and what topics are covered?"}
    ]
})

# Print the final answer
print("\n✅ Final Answer:")
print(result["messages"][-1].content)