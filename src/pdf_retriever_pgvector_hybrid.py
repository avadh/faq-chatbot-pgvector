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


def hybrid_search(query, top_k=5):
    """Perform a hybrid search using BM25 and pgvector and combine results."""
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    cursor.execute(
        """
        WITH bm25_results AS (
            SELECT id, doc_text, ts_rank_cd(to_tsvector(doc_text), plainto_tsquery(%s)) AS bm25_score
            FROM document_vectors
            WHERE to_tsvector(doc_text) @@ plainto_tsquery(%s)
            ORDER BY bm25_score DESC
            LIMIT %s
        ),
        vector_results AS (
            SELECT id, doc_text, 1 - (embedding <=> %s::vector) AS vector_score
            FROM document_vectors
            ORDER BY vector_score DESC
            LIMIT %s
        )
        SELECT id, doc_text, 
               (0.5 * COALESCE(bm25_score, 0)) + (0.5 * COALESCE(vector_score, 0)) AS final_score
        FROM bm25_results
        FULL OUTER JOIN vector_results USING (id, doc_text)
        ORDER BY final_score DESC
        LIMIT %s;
        """,
        (query, query, top_k, query_embedding, top_k, top_k)
    )

    results = cursor.fetchall()

    retrieved_docs = []
    for row in results:
        if len(row) == 3:
            doc_text, final_score = row[1], row[2]
            retrieved_docs.append({"text": doc_text, "similarity": float(final_score)})

    return retrieved_docs


# Test retrieval
if __name__ == "__main__":
    query = "What is prompt engineering"
    results = hybrid_search(query)
    for doc in results:
        print(f"🔹 Retrieved Text: {doc}\n")
