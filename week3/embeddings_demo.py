from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Load a lightweight local embeddings model
print("Loading embeddings model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
print("✅ Model loaded!")

# Our test sentences
sentences = [
    "I love Python programming",
    "Python is my favorite language",
    "The weather is pleasant today",
    "It is sunny and warm outside",
    "Data structures and algorithms are important",
    "LeetCode helps practice DSA problems"
]

# Convert sentences to embeddings
print("\nGenerating embeddings...")
embeddings = embeddings_model.embed_documents(sentences)
embeddings = np.array(embeddings)
print(f"✅ Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} numbers")

# Calculate cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Compare sentences
print("\n--- Similarity Scores ---")
print("(1.0 = identical meaning, 0.0 = completely different)\n")

pairs = [
    (0, 1, "I love Python  vs  Python is my favorite"),
    (0, 2, "I love Python  vs  Weather is pleasant"),
    (2, 3, "Weather pleasant  vs  Sunny and warm"),
    (4, 5, "DSA important  vs  LeetCode helps DSA"),
    (0, 4, "I love Python  vs  DSA important")
]

for i, j, label in pairs:
    score = cosine_similarity(embeddings[i], embeddings[j])
    bar = "█" * int(score * 20)
    print(f"{label}")
    print(f"Similarity: {score:.3f} {bar}\n")