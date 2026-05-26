from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Define Tool 1 - Get current date
@tool
def get_current_date() -> str:
    """Returns today's date and day of the week."""
    now = datetime.now()
    return now.strftime("Today is %A, %B %d, %Y")

# Define Tool 2 - Calculator
@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return f"Could not calculate: {expression}"

# Setup model with tools
model = ChatGroq(model="llama-3.3-70b-versatile")
tools = [get_current_date, calculate]
model_with_tools = model.bind_tools(tools)

# The tool loop
def run_with_tools(user_message):
    print(f"\nUser: {user_message}")
    messages = [HumanMessage(content=user_message)]

    # Step 1 - Ask LLM what to do
    response = model_with_tools.invoke(messages)
    messages.append(response)

    # Step 2 - Check if LLM wants to call a tool
    while response.tool_calls:
        print(f"\n🔧 LLM wants to call tools:")

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"   Tool: {tool_name}, Args: {tool_args}")

            # Step 3 - Execute the tool
            if tool_name == "get_current_date":
                tool_result = get_current_date.invoke({})
            elif tool_name == "calculate":
                tool_result = calculate.invoke(tool_args)
            print(f"   Result: {tool_result}")

            # Step 4 - Send result back to LLM
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))

        # Step 5 - Get final response from LLM
        response = model_with_tools.invoke(messages)
        messages.append(response)

    print(f"\n✅ Final Answer: {response.content}")

# Test it
run_with_tools(
    "What is 2847 multiplied by 39, and what day of the week is it today?"
)
