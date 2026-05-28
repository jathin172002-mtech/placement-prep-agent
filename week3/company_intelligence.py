import json
import os
import chromadb
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ── Setup ──────────────────────────────────────────────────

print("🚀 Starting Company Intelligence Store...")

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()
search = TavilySearch(max_results=5)

client = chromadb.PersistentClient(path="week3/chroma_db")

# ── Step 1: Research Company ───────────────────────────────

def research_company(company, role):
    print(f"\n🔍 Researching {company} for {role}...")

    queries = [
        f"{company} {role} interview process 2024",
        f"{company} tech stack engineering",
        f"{company} interview experience tips"
    ]

    all_content = []
    for query in queries:
        print(f"   Searching: {query}")
        results = search.invoke(query)
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    all_content.append(r.get("content", ""))
        else:
            all_content.append(str(results))

    combined_text = "\n\n".join(all_content)
    print(f"   ✅ Found {len(all_content)} results")
    return combined_text

# ── Step 2: Store in ChromaDB ──────────────────────────────

def store_company(company, role, content):
    collection_name = f"{company}_{role}".replace(" ", "_").lower()
    print(f"\n💾 Storing in ChromaDB collection: {collection_name}")

    # Check if already stored
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        print(f"   ✅ Already stored! Loading existing data...")
        return client.get_collection(collection_name)

    # Split content into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([content])
    print(f"   Created {len(chunks)} chunks")

    # Create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "company": company, "role": role}
    )

    # Store chunks
    for i in range(0, len(chunks), 50):
        batch = chunks[i:i+50]
        docs = [c.page_content for c in batch]
        metas = [{"company": company, "role": role, "chunk": i+j}
                 for j, c in enumerate(batch)]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        embs = embeddings_model.embed_documents(docs)
        collection.upsert(
            documents=docs,
            embeddings=embs,
            metadatas=metas,
            ids=ids
        )

    print(f"   ✅ Stored {collection.count()} chunks")
    return collection

# ── Step 3: Conversational RAG ─────────────────────────────

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """Rewrite the follow-up question as a standalone question
using the conversation history. Return ONLY the rewritten question."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Follow-up: {question}")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a placement preparation expert for {company} {role} role.
Answer based ONLY on the context provided.
If answer not in context say "I don't have enough information."
Be specific and actionable."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", """Context:
{context}

Question: {question}""")
])

rewrite_chain = rewrite_prompt | llm | parser
answer_chain = answer_prompt | llm | parser

def ask_question(collection, company, role, question, history):
    # Rewrite if history exists
    if history:
        rewritten = rewrite_chain.invoke({
            "history": history,
            "question": question
        })
        print(f"   🔄 Rewritten: {rewritten}")
    else:
        rewritten = question

    # Retrieve
    query_emb = embeddings_model.embed_query(rewritten)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )
    context = "\n\n".join(results["documents"][0])

    # Generate
    answer = answer_chain.invoke({
        "company": company,
        "role": role,
        "history": history,
        "context": context,
        "question": question
    })

    # Update history
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    return answer

# ── Main Chat Loop ─────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("🎯 COMPANY INTELLIGENCE STORE")
    print("=" * 60)
    print("Commands:")
    print("  'research <company> <role>' → research a new company")
    print("  'quit' → exit")
    print("  anything else → ask a question")
    print("=" * 60)

    current_collection = None
    current_company = None
    current_role = None
    history = []

    while True:
        user_input = input("\nYou: ").strip()

        # Skip empty input
        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        elif user_input.lower().startswith("research "):
            parts = user_input[9:].strip().split(" ", 1)
            if len(parts) < 2:
                print("❌ Usage: research <company> <role>")
                print("   Example: research Google SDE-1")
                continue

            current_company = parts[0]
            current_role = parts[1]
            history = []

            content = research_company(current_company, current_role)
            current_collection = store_company(
                current_company, current_role, content
            )
            print(f"\n✅ Ready! Ask me anything about {current_company} {current_role} interviews.")

        elif current_collection is None:
            print("❌ Please research a company first!")
            print("   Example: research Google SDE-1")

        else:
            answer = ask_question(
                current_collection,
                current_company,
                current_role,
                user_input,
                history
            )
            print(f"\nAssistant: {answer}")
            print(f"[Memory: {len(history)} messages]")

main()