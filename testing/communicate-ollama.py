import requests
import json
from requests import JSONDecodeError
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def query_ollama_model(prompt, ollama_url, ollama_key):
    """Sends a query to the Ollama model and returns the desired sentence."""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ollama_key}"
    }

    data = {
        "prompt": prompt,
        "model": "llama3.1:8b-instruct-q4_1"  # Replace with the desired model name
    }

    response = requests.post(f"{ollama_url}/api/generate", headers=headers, json=data)

    if response.status_code == 200:
        print(f"Response Text:\n{response.text}")  # Print raw response
        return response
    else:
        raise Exception(f"Error querying Ollama model: {response.text}")

if __name__ == "__main__":
    ollama_url = os.getenv("OLLAMA_URL")
    ollama_key = os.getenv("OLLAMA_KEY")

    while True:
        prompt = input("Enter your prompt: ")
        response = query_ollama_model(prompt, ollama_url, ollama_key)
        print(response)