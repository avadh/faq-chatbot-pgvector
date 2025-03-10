import os
import json
import psycopg2
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# PostgreSQL connection
conn = psycopg2.connect(
    dbname="rag_db", user="postgres", password="Password@123", host="localhost", port="5433"
)
cursor = conn.cursor()
register_vector(cursor)

PDF_DIR = "data/"


def extract_text_from_pdfs():
    """Extracts and splits text from PDFs."""
    documents = []
    metadata = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_DIR, pdf_file))
            raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            chunks = text_splitter.split_text(raw_text)

            for chunk in chunks:
                documents.append(chunk)
                metadata.append({"source": pdf_file, "text": chunk})

    return documents, metadata


def remove_null_characters(text):
    return text.replace("\0", "")

def store_embeddings():
    # store documents in postgres db
    documents, metadata = extract_text_from_pdfs()

    for doc_text in documents:
        # Remove NUL characters before embedding and inserting
        cleaned_doc_text = remove_null_characters(doc_text)
        embedding = model.encode(cleaned_doc_text, normalize_embeddings=True).tolist()
        cursor.execute("INSERT INTO document_vectors (doc_text, embedding) VALUES (%s, %s)", (cleaned_doc_text, embedding))

    conn.commit()
    print(f"Stored {len(documents)} documents in PostgreSQL.")


if __name__ == "__main__":
    store_embeddings()