import gradio as gr
import threading
import uvicorn
import requests
from api import app

def start_fastapi():
    """Runs FastAPI backend in a separate thread."""
    uvicorn.run(app, host="0.0.0.0", port=5000)  #  Change FastAPI port to 5000 to avoid conflict with LLM

# Start FastAPI in a separate thread
threading.Thread(target=start_fastapi, daemon=True).start()

# Define chatbot frontend
def chat_with_bot(user_query, type_of_search = 0):
    """Send query to FastAPI backend and return response."""
    try:
        response = requests.post("http://127.0.0.1:5000/query", json={"query": user_query, "type": type_of_search})
        response_json = response.json()  # Ensure response is parsed as JSON

        print("RAW RESPONSE:", response_json)  # Debugging step
        print("Type of response:", type(response_json))

        if isinstance(response_json, dict) and "response" in response_json:
            return response_json["response"]
        else:
            return "Error: Unexpected API response format."
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to connect to backend. {str(e)}"

# Launch Gradio UI
gr.Interface(fn=chat_with_bot, inputs=[
        gr.Textbox(label="User Query"), gr.Textbox(label="Type Of Search")], outputs="text").launch(server_name="0.0.0.0", server_port=7860)
