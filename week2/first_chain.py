from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Building Block 1 - The Model
model = ChatGroq(model="llama-3.3-70b-versatile")

# Building Block 2 - The Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Tell me about {company} in 2 sentences."
)

# Building Block 3 - The Output Parser
parser = StrOutputParser()

# Connect all 3 with LCEL pipe operator
chain = prompt | model | parser

# Run the chain
result = chain.invoke({"company": "Google"})
print(result)