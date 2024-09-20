import os
import logging
from fastapi import HTTPException
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts.chat import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from sqlalchemy.orm import Session

from .models import Conversation, Message
from .utils import ASSISTANT_UUID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

ollama_client = Ollama(
    base_url=OLLAMA_URL,
    model=MODEL_NAME
)

TEMPLATE = """
    You are Home AI. Your job is to assist house members with their daily tasks. Format answers with markdown.

    Current conversation:
    {history}
    Human: {input}
    AI:
"""

PROMPT_TEMPLATE = PromptTemplate(input_variables=["history", "input"], template=TEMPLATE)

def create_ollama_client():
    return Ollama(
        base_url=OLLAMA_URL,
        model=MODEL_NAME
    )

def create_new_conversation(db: Session, user_id: str):
    """
    Creates a new conversation for the given user.
    """
    new_conversation = Conversation(user_id=user_id)
    db.add(new_conversation)
    db.commit()
    db.refresh(new_conversation)
    return new_conversation

def update_conversation_summary(db: Session, conversation: Conversation, memory):
    """
    Updates the conversation summary in the database based on the memory buffer.
    """
    try:
        memory_summary = memory.buffer
        conversation.summary = memory_summary
        print(f"Updating conversation summary: {memory_summary}")
        db.commit()
    except Exception as e:
        logger.error(f"Failed to update conversation summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to update conversation summary.")

def continue_conversation(db: Session, conversation_id: str, user_id: str, message_content: str):
    """
    Continues an active conversation, processes user input, and returns the AI's response.
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
        Conversation.status == "active"
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or closed")

    summary_llm = create_ollama_client()

    # Load the summary into the memory buffer
    memory = ConversationSummaryMemory(
        llm=summary_llm,
        buffer=conversation.summary or "",
        return_messages=False,
        max_token_limit=200  # Adjust token limit based on your model
    )

    conversation_chain = ConversationChain(
        prompt=PROMPT_TEMPLATE,
        llm=ollama_client,
        memory=memory,
        verbose=False
    )

    # Generate response
    try:
        response = conversation_chain.invoke({"history": memory.buffer, "input": message_content})
        response_text = response["response"]
    except Exception as e:
        logger.error(f"Failed to generate AI response: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate AI response.")

    # Save user and AI messages to the database
    try:
        log_message_to_db(db, conversation_id, user_id, message_content, response_text)
    except Exception as e:
        logger.error(f"Failed to log conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to log conversation.")

    # Update conversation summary
    try:
        update_conversation_summary(db, conversation, memory)
    except Exception as e:
        logger.error(f"Failed to update conversation summary: {e}")

    return {"llm_response": response_text}

def log_message_to_db(db: Session, conversation_id: str, user_id: str, user_message: str, ai_response: str):
    """
    Logs user and AI messages to the database.
    """
    new_message = Message(
        conversation_id=conversation_id,
        sender_id=user_id,
        content=user_message,
        llm_model=MODEL_NAME,
        response_time=1.2
    )
    db.add(new_message)

    llm_message = Message(
        conversation_id=conversation_id,
        sender_id=ASSISTANT_UUID,
        content=ai_response,
        llm_model=MODEL_NAME,
        response_time=2.0
    )
    db.add(llm_message)
    db.commit()

def delete_conversation(db: Session, conversation_id: str, user_id: str):
    """
    Deletes a conversation from the database.
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id, 
        Conversation.user_id == user_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conversation)
    db.commit()
    return {"message": "Conversation deleted successfully"}
