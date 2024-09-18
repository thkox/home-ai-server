import os
from langchain_community.llms.ollama import Ollama
from sqlalchemy.orm import Session
from .models import Conversation, Message
from .schemas import ConversationCreate, ConversationOut
from .auth import get_current_user
from fastapi import Depends, HTTPException, status
from .database import get_db
from langchain.chains import ConversationChain

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

ollama_client = Ollama(base_url=OLLAMA_URL)

# Create a new conversation
def create_new_conversation(db: Session, user_id: str):
    new_conversation = Conversation(user_id=user_id)
    db.add(new_conversation)
    db.commit()
    db.refresh(new_conversation)
    return new_conversation


# Continue an existing conversation
def continue_conversation(db: Session, conversation_id: str, user_id: str, message_content: str):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
        Conversation.status == "active"
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or closed")

    # Ensure Unicode characters are properly logged and encoded
    print(f"Message content: {message_content}")  # For debugging purposes

    # Use Ollama with remote server for generating response
    response = ollama_client.generate(model=MODEL_NAME, prompts=[message_content])

    # Serialize the response from the LLM
    response_text = response.generations[0][0].text if response and response.generations else "No response"

    # Log both user and LLM message to the DB
    new_message = Message(
        conversation_id=conversation_id,
        sender_id=user_id,
        content=message_content,
        llm_model=MODEL_NAME,
        response_time=1.2  # Example response time
    )
    db.add(new_message)

    # LLM response message (serialized as text)
    llm_message = Message(
        conversation_id=conversation_id,
        sender_id=None,  # No sender for LLM, it's system-generated
        content=response_text,  # Store the serialized text response
        llm_model=MODEL_NAME,
        response_time=2.0
    )
    db.add(llm_message)

    db.commit()
    return {"user_message": message_content, "llm_response": response_text}

# Delete a previous conversation
def delete_conversation(db: Session, conversation_id: str, user_id: str):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id, 
        Conversation.user_id == user_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conversation)
    db.commit()
    return {"message": "Conversation deleted successfully"}
