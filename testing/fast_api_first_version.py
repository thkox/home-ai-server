from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
import os
from dotenv import load_dotenv
import requests
import sqlite3
import uuid
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

ollama_url = os.getenv("OLLAMA_URL")
ollama_key = os.getenv("OLLAMA_KEY")
model_name = os.getenv("MODEL_NAME")

if not ollama_url:
    raise ValueError("Please set the OLLAMA_URL environment variable in your .env file")

# Initialize ChatOllama with the selected model
ollama = ChatOllama(base_url=ollama_url, model=model_name)


class UserMessage(BaseModel):
    content: str


class Session(BaseModel):
    session_id: str
    messages: Optional[List[dict]] = None


# Function to extract model names
def extract_model_names(models_info: list) -> tuple:
    return tuple(model["name"] for model in models_info)


# Function to get Ollama models from the API
def get_ollama_models(ollama_api_base_url: str, token: str = '') -> list:
    url = f"{ollama_api_base_url}/api/tags"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        models = data.get('models', [])
        return [{'id': model.get('model', ''), 'name': model.get('name', model.get('model', ''))} for model in models]
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving Ollama models: {e}")
        return []


# Function to get session history from SQLite database
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, connection="sqlite:///memory.db")


# Function to generate a unique session ID
def generate_session_id():
    return str(uuid.uuid4())


# Function to retrieve all session IDs from the SQLite database
def list_all_sessions():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_id FROM message_store")
    sessions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return sessions


# Route to get available models
@app.get("/models")
def get_models():
    models = get_ollama_models(ollama_url, ollama_key)
    available_model_names = extract_model_names(models)
    return {"models": available_model_names}


# Route to start a new session
@app.post("/session/new")
def start_new_session():
    session_id = generate_session_id()
    memory = ConversationBufferMemory(return_messages=True)
    conversation = RunnableWithMessageHistory(
        runnable=ollama,
        get_session_history=lambda: get_session_history(session_id),
        memory=memory,
        verbose=True
    )
    return {"session_id": session_id}


# Route to resume an existing session
@app.post("/session/resume/{session_id}")
def resume_session(session_id: str):
    sessions = list_all_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    memory = ConversationBufferMemory(return_messages=True)
    conversation = RunnableWithMessageHistory(
        runnable=ollama,
        get_session_history=lambda: get_session_history(session_id),
        memory=memory,
        verbose=True
    )
    return {"session_id": session_id, "message": "Session resumed"}


# Route to list all sessions
@app.get("/sessions")
def get_sessions():
    sessions = list_all_sessions()
    return {"sessions": sessions}


# Route to send a message and get a response from the LLM
@app.post("/chat/{session_id}")
def chat(session_id: str, user_message: UserMessage):
    sessions = list_all_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    memory = ConversationBufferMemory(return_messages=True)
    conversation = RunnableWithMessageHistory(
        runnable=ollama,
        get_session_history=lambda: get_session_history(session_id),
        memory=memory,
        verbose=True
    )

    llm_response = conversation.invoke(input=user_message.content, config={"configurable": {"session_id": session_id}})
    return {"response": llm_response}


# Route to end a session
@app.post("/session/end/{session_id}")
def end_session(session_id: str):
    sessions = list_all_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    # Optionally, you can add code here to handle any cleanup required for ending a session
    return {"message": f"Session {session_id} ended"}


# Uvicorn main
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
