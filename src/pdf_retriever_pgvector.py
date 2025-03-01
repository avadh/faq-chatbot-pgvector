import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

# PostgreSQL connection
conn = psycopg2.connect(
    dbname="rag_db", user="postgres", password="Password@123", host="localhost", port="5433"
)
cursor = conn.cursor()
register_vector(cursor)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_documents(query, top_k=3):
    """Retrieve the most relevant document chunks from PostgreSQL using pgvector."""
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    cursor.execute(
        "SELECT doc_text, 1 - (embedding <=> %s::vector) AS similarity FROM document_vectors ORDER BY similarity DESC LIMIT %s",
        (query_embedding, top_k),
    )
    results = cursor.fetchall()

    # Ensure results are returned as a list of dictionaries
    retrieved_docs = []
    for row in results:
        if len(row) == 2:  # Ensure two values are returned
            doc_text, similarity = row
            retrieved_docs.append({"text": doc_text, "similarity": float(similarity)})
        else:
            print(f"⚠️ Unexpected row format: {row}")

    return retrieved_docs

# Test retrieval
if __name__ == "__main__":
    query = "What is prompt engineering"
    results = retrieve_documents(query)
    for doc in results:
        print(f"🔹 Retrieved Text: {doc}\n")
