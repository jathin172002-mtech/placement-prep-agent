from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ── Setup ──────────────────────────────────────────────────

print("Setting up Conversational RAG...")

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# ── Load and Store ─────────────────────────────────────────

print("Loading pages...")
urls = [
    "https://www.geeksforgeeks.org/google-interview-preparation/",
    "https://www.geeksforgeeks.org/amazon-interview-preparation/"
]

loader = WebBaseLoader(urls)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)

client = chromadb.PersistentClient(path="week3/chroma_db")

try:
    client.delete_collection("conv_rag")
except:
    pass

collection = client.get_or_create_collection(
    name="conv_rag",
    metadata={"hnsw:space": "cosine"}
)

for i in range(0, len(chunks), 50):
    batch = chunks[i:i+50]
    docs = [c.page_content for c in batch]
    metas = [{"source": c.metadata.get("source", "")} for c in batch]
    ids = [f"chunk_{i+j}" for j in range(len(batch))]
    embs = embeddings_model.embed_documents(docs)
    collection.upsert(documents=docs, embeddings=embs,
                      metadatas=metas, ids=ids)

print(f"✅ Stored {collection.count()} chunks")

# ── Step 1: Rewrite Question Using History ─────────────────

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a conversation history and a follow-up question,
rewrite the follow-up question to be a standalone question that includes
all necessary context from the conversation history.

If the question is already standalone, return it as is.
Return ONLY the rewritten question, nothing else."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Follow-up question: {question}")
])

rewrite_chain = rewrite_prompt | llm | parser

# ── Step 2: Retrieve ───────────────────────────────────────

def retrieve(question, top_k=3):
    query_embedding = embeddings_model.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0], results["metadatas"][0]

# ── Step 3: Generate Answer ────────────────────────────────

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a placement preparation expert.
Answer the question based ONLY on the context provided.
If the answer is not in the context say "I don't have enough information."
Be clear and structured in your answer."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", """Context:
{context}

Question: {question}""")
])

answer_chain = answer_prompt | llm | parser

# ── Full Conversational RAG ────────────────────────────────

conversation_history = []

def chat(question):
    print(f"\n{'='*50}")
    print(f"You: {question}")

    # Step 1 — Rewrite question if needed
    if conversation_history:
        rewritten = rewrite_chain.invoke({
            "history": conversation_history,
            "question": question
        })
        print(f"🔄 Rewritten: {rewritten}")
    else:
        rewritten = question

    # Step 2 — Retrieve relevant chunks
    chunks, metadatas = retrieve(rewritten)
    context = "\n\n".join(chunks)

    # Step 3 — Generate answer
    answer = answer_chain.invoke({
        "history": conversation_history,
        "context": context,
        "question": question
    })

    # Save to history
    conversation_history.append(HumanMessage(content=question))
    conversation_history.append(AIMessage(content=answer))

    print(f"\nAssistant: {answer}")
    print(f"\n[Memory: {len(conversation_history)} messages]")

    return answer

# ── Test Conversation ──────────────────────────────────────

print("\n" + "=" * 50)
print("CONVERSATIONAL RAG — 6 TURN TEST")
print("=" * 50)

chat("Tell me about Google's interview process")
chat("How many rounds does it have?")
chat("What topics should I focus on?")
chat("What about Amazon's process?")
chat("How does it compare to the previous company?")
chat("Which one should I apply to first?")
