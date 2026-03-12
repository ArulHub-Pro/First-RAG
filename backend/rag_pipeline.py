from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load healthcare documents
with open("data/healthcare_docs.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n\n")

# Convert documents to embeddings
doc_embeddings = embed_model.encode(documents)

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Load FLAN-T5 model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def retrieve_context(question, k=3):

    query_embedding = embed_model.encode([question])

    distances, indices = index.search(np.array(query_embedding), k)

    retrieved_docs = [documents[i] for i in indices[0]]

    context = " ".join(retrieved_docs)

    return context


def generate_answer(context, question):

    prompt = f"""
You are a helpful healthcare assistant.

Use the medical information below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(prompt, max_length=200)

    return result[0]["generated_text"]


def get_answer(question):

    context = retrieve_context(question)

    answer = generate_answer(context, question)

    return answer