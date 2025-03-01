from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
from pdf_retriever_pgvector import retrieve_documents, model
from pdf_retriever_pgvector_hybrid import hybrid_search

LLM_API_URL = "http://localhost:8000/v1/completions"

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    type: str


@app.post("/query")
async def ask_question(request: QueryRequest):
    if request.type == "1":
        print("Calling hybrid search")
        retrieved_docs = hybrid_search(request.query)
    else:
        retrieved_docs = retrieve_documents(request.query, top_k=3)

    context = "\n\n".join([f"Retrieved Text:\n{doc}" for doc in retrieved_docs])
    print("In api.py: context:", context)
    print("In api.py: similarity_score=", retrieved_docs[0]["similarity"])
    # Check if there are any documents and validate similarity score
    if not retrieved_docs or retrieved_docs[0]["similarity"] < 0.5:
        return {"response": "Out of scope for this FAQ chatbot."}

    payload = {
        "model": "deepseek-r1:8b", # other model is llama3.2:latest
        "prompt": f"User Query: {request.query}\n\nContext:\n{context}\n\nGenerate a detailed answer:",
        "temperature": 0.5,
        "stream": True  # Enable streaming response
    }

    response = requests.post(LLM_API_URL, json=payload, stream=True)

    # Validate JSON response from LLM
    try:
        response_json = response.json()
        if isinstance(response_json, dict) and "choices" in response_json:
            generated_text = response_json["choices"][0].get("text", "No response generated.")
        else:
            generated_text = "Error: Unexpected response format from LLM."
    except (requests.exceptions.JSONDecodeError, json.JSONDecodeError):
        generated_text = "Error: LLM API did not return valid JSON."

    return {"response": generated_text}
