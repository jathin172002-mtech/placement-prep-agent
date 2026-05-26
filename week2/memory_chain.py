from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# Prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful placement prep assistant. Remember everything the user tells you."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model | parser

# Manual memory store
conversation_history = []

def chat_with_memory(user_message):
    print(f"\nYou: {user_message}")

    # Run chain with history
    response = chain.invoke({
        "input": user_message,
        "history": conversation_history
    })

    # Save to memory
    conversation_history.append(HumanMessage(content=user_message))
    conversation_history.append(AIMessage(content=response))

    print(f"Assistant: {response}")
    print(f"[Memory: {len(conversation_history)} messages stored]")

    return response

# Test memory
print("=" * 50)
print("Chat with Memory Demo")
print("=" * 50)

chat_with_memory("I am preparing for Amazon SDE-1 interviews")
chat_with_memory("What should I focus on?")
chat_with_memory("What company did I tell you I am preparing for?")
chat_with_memory("What topics did you suggest for that company?")
