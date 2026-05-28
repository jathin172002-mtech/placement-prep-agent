import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ── Setup ─────────────────────────────────────────────────

print("Loading embeddings model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
print("✅ Model loaded!")

# Create ChromaDB client with persistence
client = chromadb.PersistentClient(path="week3/chroma_db")

# Delete existing collection to start fresh
try:
    client.delete_collection("companies")
except:
    pass

# Create fresh collection
collection = client.get_or_create_collection(
    name="companies",
    metadata={"hnsw:space": "cosine"}
)

print("✅ ChromaDB ready!")

# ── Add Documents ─────────────────────────────────────────

documents = [
    "Google interview is very hard. Focus on DSA and system design. 4 rounds including coding and behavioral.",
    "Google uses Python, Java, C++, JavaScript and Cloud Computing in their tech stack.",
    "Amazon interview focuses on leadership principles and DSA. 5-6 rounds total.",
    "Amazon uses Java, Python, AWS services extensively in their engineering teams.",
    "Infosys interview is entry level. Focus on aptitude, basic coding and HR round.",
    "Infosys uses Java, Python, .NET and SQL for most of their projects.",
    "Zomato interview focuses on backend development, system design and DSA.",
    "Zomato uses Python, Go, Kafka and PostgreSQL in their tech stack.",
    "Flipkart interview focuses on DSA, system design and product thinking.",
    "Flipkart uses Java, Kotlin, Python and MySQL in their tech stack."
]

metadatas = [
    {"company": "Google", "type": "interview"},
    {"company": "Google", "type": "tech_stack"},
    {"company": "Amazon", "type": "interview"},
    {"company": "Amazon", "type": "tech_stack"},
    {"company": "Infosys", "type": "interview"},
    {"company": "Infosys", "type": "tech_stack"},
    {"company": "Zomato", "type": "interview"},
    {"company": "Zomato", "type": "tech_stack"},
    {"company": "Flipkart", "type": "interview"},
    {"company": "Flipkart", "type": "tech_stack"},
]

# Generate embeddings
print("\nGenerating embeddings for documents...")
doc_embeddings = embeddings_model.embed_documents(documents)

# Add to ChromaDB
collection.upsert(
    documents=documents,
    embeddings=doc_embeddings,
    metadatas=metadatas,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"✅ Added {len(documents)} documents to ChromaDB!")
print(f"Total documents in collection: {collection.count()}")

# ── Query ─────────────────────────────────────────────────

def search(question, top_k=3):
    print(f"\n🔍 Query: {question}")

    query_embedding = embeddings_model.embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    print(f"\nTop {top_k} results:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = 1 - distance
        print(f"\n{i+1}. [{metadata['company']} - {metadata['type']}]")
        print(f"   {doc}")
        print(f"   Similarity: {similarity:.3f}")

# ── Test Queries ──────────────────────────────────────────

print("\n" + "=" * 60)
print("SEARCHING THE VECTOR DATABASE")
print("=" * 60)

search("Which company has the hardest interview?")
search("What programming languages does Amazon use?")
search("Which company is good for freshers?")
search("What should I study for backend roles?")
search("What does Flipkart use for their tech stack?")