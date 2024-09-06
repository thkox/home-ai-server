from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
import os
from dotenv import load_dotenv
import requests
import sqlite3

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

# Function to convert list of dictionaries to BaseMessage objects
def convert_to_base_messages(message_list):
    converted_messages = []
    for message in message_list:
        if message['role'] == 'user':
            converted_messages.append(HumanMessage(content=message['content']))
        elif message['role'] == 'llm':
            converted_messages.append(AIMessage(content=message['content']))
    return converted_messages

# Function to get session history from SQLite database
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, connection="sqlite:///memory.db")

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file

    ollama_url = os.getenv("OLLAMA_URL")
    ollama_key = os.getenv("OLLAMA_KEY")
    model_name = os.getenv("MODEL_NAME")

    if not ollama_url:
        raise ValueError("Please set the OLLAMA_URL environment variable in your .env file")

    models = get_ollama_models(ollama_url, ollama_key)
    available_model_names = extract_model_names(models)

    print("Available models:", available_model_names)

    if model_name not in available_model_names:
        print(f"Model '{model_name}' not found in the available models.")
    else:
        print(f"Model '{model_name}' found, proceeding with initialization...")

    # Initialize ChatOllama with the selected model
    ollama = ChatOllama(base_url=ollama_url, model=model_name)

    # Initialize conversation memory
    memory = ConversationBufferMemory(return_messages=True)

    # Create a RunnableWithMessageHistory instance
    conversation = RunnableWithMessageHistory(
        runnable=ollama,  # The LLM model
        get_session_history=get_session_history,  # How to retrieve the session history
        memory=memory,
        verbose=True
    )

    # User interaction loop
    while True:
        user_message = input("You: ")

        if user_message.lower() in ["exit", "quit"]:
            print("Ending the conversation.")
            break

        # Generate LLM response with memory
        llm_response = conversation.invoke(input=user_message, config={"configurable": {"session_id": "unique_session_id"}})
        print(f"LLM: {llm_response}")
