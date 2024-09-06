from langchain_community.chat_models import ChatOllama
import os
from dotenv import load_dotenv
import requests
import json


def extract_model_names(models_info: list) -> tuple:
    return tuple(model["name"] for model in models_info)


def get_ollama_models(ollama_api_base_url: str, token: str = '') -> list:
    """
    Retrieves a list of available Ollama models from the given API URL.

    Args:
        ollama_api_base_url (str): The base URL of the Ollama API.
        token (str, optional): An optional authentication token for the API. Defaults to ''.

    Returns:
        list: A list of dictionaries containing model information (id, name).
            If any errors occur, an empty list will be returned.
    """
    url = f"{ollama_api_base_url}/api/tags"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for non-200 status codes

        data = response.json()
        models = data.get('models', [])  # Handle missing 'models' key gracefully

        return [
            {
                'id': model.get('model', ''),
                'name': model.get('name', model.get('model', ''))
            }
            for model in models
        ]

    except requests.exceptions.RequestException as e:
        print(f"Error retrieving Ollama models: {e}")
        return []  # Return an empty list on errors

# TODO: Fix this function
def pull_model(ollama_api_base_url: str, model_name: str, token: str = '') -> bool:
    """
    It does not work!

    Pulls a model from the Ollama API if it does not exist.

    Args:
        ollama_api_base_url (str): The base URL of the Ollama API.
        model_name (str): The name of the model to pull.
        token (str, optional): An optional authentication token for the API. Defaults to ''.

    Returns:
        bool: True if the model was pulled successfully, False otherwise.
    """
    url = f"{ollama_api_base_url}/api/models/pull"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }
    if token:
        headers['Authorization'] = f'Bearer {token}'

    payload = json.dumps({"model": model_name})

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for non-200 status codes

        # Assuming that a successful response indicates that the model is being pulled.
        print(f"Model '{model_name}' is being pulled.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error pulling model '{model_name}': {e}")
        return False

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file

    ollama_url = os.getenv("OLLAMA_URL")
    ollama_key = os.getenv("OLLAMA_KEY")
    model_name = os.getenv("MODEL_NAME")

    if not ollama_url:
        raise ValueError("Please set the OLLAMA_URL environment variable in your .env file")

    models = get_ollama_models(ollama_url, ollama_key)

    # Extract available model names from the API response
    available_model_names = extract_model_names(models)

    print("Available models:", available_model_names)  # Debugging line

    if model_name not in available_model_names:
        print(f"Model '{model_name}' not found in the available models.")

        # # Try to pull the model if it does not exist
        # pull_success = pull_model(ollama_url, model_name, ollama_key)
        # if not pull_success:
        #     raise ValueError(f"Failed to pull model '{model_name}'. Please check the model name or network connection.")
        #
        # print(f"Model '{model_name}' is being pulled successfully. Re-run the script after the model is pulled.")
        # return

    print(f"Model '{model_name}' found, proceeding with initialization...")

    # Initialize ChatOllama with the selected model
    ollama = ChatOllama(
        base_url=ollama_url,
        model=model_name
    )

    print(ollama.invoke("Why is the sky blue?"))