from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ── Setup ─────────────────────────────────────────────────

print("Loading embeddings model...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Model loaded!")

# ── Step 1: Load Web Page ─────────────────────────────────

print("\n📄 Step 1: Loading web page...")

urls = [
    "https://www.geeksforgeeks.org/google-interview-preparation/",
    "https://www.geeksforgeeks.org/amazon-interview-preparation/"
]

loader = WebBaseLoader(urls)
pages = loader.load()

print(f"✅ Loaded {len(pages)} pages")
for page in pages:
    print(f"   URL: {page.metadata['source']}")
    print(f"   Length: {len(page.page_content)} characters")

# ── Step 2: Split into Chunks ─────────────────────────────

print("\n✂️  Step 2: Splitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = splitter.split_documents(pages)

print(f"✅ Created {len(chunks)} chunks from {len(pages)} pages")
print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

# Show 2 consecutive chunks to see overlap
print("\n--- Sample: 2 consecutive chunks ---")
print(f"\nChunk 1 (last 100 chars):")
print(f"...{chunks[0].page_content[-100:]}")
print(f"\nChunk 2 (first 100 chars):")
print(f"{chunks[1].page_content[:100]}...")

# ── Step 3: Store in ChromaDB ─────────────────────────────

print("\n💾 Step 3: Storing chunks in ChromaDB...")

client = chromadb.PersistentClient(path="week3/chroma_db")

# Delete old collection if exists
try:
    client.delete_collection("interview_pages")
except:
    pass

collection = client.get_or_create_collection(
    name="interview_pages",
    metadata={"hnsw:space": "cosine"}
)

# Store chunks in batches
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
    print(f"   Stored chunks {i} to {i+len(batch)}")

print(f"\n✅ Total chunks stored: {collection.count()}")

# ── Step 4: Search ────────────────────────────────────────

def search(question, top_k=3):
    print(f"\n🔍 Query: {question}")
    query_embedding = embeddings_model.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    for i, (doc, metadata) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        source = metadata["source"].split("/")[-2]
        print(f"\n{i+1}. [{source}]")
        print(f"   {doc[:200]}...")

print("\n" + "=" * 60)
print("SEARCHING REAL WEB PAGE CONTENT")
print("=" * 60)

search("What is the interview process at Google?")
search("How to prepare for Amazon interviews?")
search("What data structures should I focus on?")