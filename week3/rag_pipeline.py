from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ── Setup ─────────────────────────────────────────────────

print("Setting up RAG pipeline...")

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile")

# ── Step 1: Load and Store ─────────────────────────────────

def load_and_store(urls, collection_name):
    print(f"\n📄 Loading {len(urls)} pages...")
    loader = WebBaseLoader(urls)
    pages = loader.load()

    print(f"✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    print(f"✅ Created {len(chunks)} chunks")

    print(f"💾 Storing in ChromaDB...")
    client = chromadb.PersistentClient(path="week3/chroma_db")

    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        documents = [c.page_content for c in batch]
        metadatas = [{"source": c.metadata.get("source", ""), "chunk": i+j}
                     for j, c in enumerate(batch)]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        embeddings = embeddings_model.embed_documents(documents)
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    print(f"✅ Stored {collection.count()} chunks in ChromaDB")
    return collection

# ── Step 2: Retrieve ───────────────────────────────────────

def retrieve(collection, question, top_k=3):
    query_embedding = embeddings_model.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources

# ── Step 3: Generate Answer ────────────────────────────────

def generate_answer(question, chunks, sources):
    context = "\n\n".join(chunks)

    prompt = ChatPromptTemplate.from_template("""
You are a placement preparation expert.
Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information about this."

Context:
{context}

Question: {question}

Provide a clear, structured answer. At the end mention which sources you used.
""")

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, list(set(sources))

# ── Full RAG Function ──────────────────────────────────────

def ask(collection, question):
    print(f"\n🔍 Question: {question}")
    print("Retrieving relevant chunks...")

    chunks, sources = retrieve(collection, question)
    print(f"Found {len(chunks)} relevant chunks")

    print("Generating answer...")
    answer, unique_sources = generate_answer(question, chunks, sources)

    print(f"\n✅ Answer:")
    print(answer)
    print(f"\n📚 Sources:")
    for s in unique_sources:
        print(f"   {s}")

# ── Main ───────────────────────────────────────────────────

urls = [
    "https://www.geeksforgeeks.org/google-interview-preparation/",
    "https://www.geeksforgeeks.org/amazon-interview-preparation/"
]

collection = load_and_store(urls, "rag_demo")

print("\n" + "=" * 60)
print("RAG PIPELINE READY — ASK QUESTIONS!")
print("=" * 60)

ask(collection, "What DSA topics should I focus on for Google interviews?")
ask(collection, "How many rounds does Amazon interview have?")
ask(collection, "What is the difficulty level of Google vs Amazon interviews?")
