from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Define the structure you want
class CompanyProfile(BaseModel):
    company_name: str = Field(description="Name of the company")
    tech_stack: list[str] = Field(description="List of technologies used")
    interview_rounds: list[str] = Field(description="List of interview rounds")
    key_topics: list[str] = Field(description="Key topics to prepare")
    difficulty: Literal["low", "medium", "high"] = Field(
        description="Interview difficulty level"
    )

# Setup
model = ChatGroq(model="llama-3.3-70b-versatile")
parser = JsonOutputParser(pydantic_object=CompanyProfile)

prompt = ChatPromptTemplate.from_template("""
Return interview preparation information about {company} for a {role} role.
{format_instructions}
""")
print(parser.get_format_instructions())
chain = prompt | model | parser

# Run it
company = input("Enter company name: ")
role = input("Enter role: ")

result = chain.invoke({
    "company": company,
    "role": role,
    "format_instructions": parser.get_format_instructions()
})

print("\n--- Interview Prep Profile ---")
print(f"Company:          {result['company_name']}")
print(f"Difficulty:       {result['difficulty']}")
print(f"Tech Stack:       {', '.join(result['tech_stack'])}")
print(f"Interview Rounds: {', '.join(result['interview_rounds'])}")
print(f"Key Topics:       {', '.join(result['key_topics'])}")
