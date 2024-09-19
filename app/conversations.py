import os

from fastapi import HTTPException
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts.chat import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from sqlalchemy.orm import Session

from .models import Conversation, Message
from .utils import ASSISTANT_UUID

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


# Create a new conversation
def create_new_conversation(db: Session, user_id: str):
    new_conversation = Conversation(user_id=user_id)
    db.add(new_conversation)
    db.commit()
    db.refresh(new_conversation)
    return new_conversation

# Summarize and save the conversation history to the DB
def update_conversation_summary(db: Session, conversation: Conversation, memory):
    # Save the full memory buffer into the summary field
    memory_summary = memory.load_memory_variables({})['history']
    conversation.summary = memory_summary  # Update summary with chat history
    db.commit()

def continue_conversation(db: Session, conversation_id: str, user_id: str, message_content: str):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
        Conversation.status == "active"
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or closed")

    # Initialize memory and load previous summary if available
    summary_llm = Ollama(
        base_url=OLLAMA_URL,
        temperature=0.0,
        model=MODEL_NAME
    )

    # Log the current conversation summary for debugging
    print(f"Existing Summary: {conversation.summary}")

    # Load the summary into the memory buffer, or start with a fresh memory
    memory = ConversationSummaryMemory(
        llm=summary_llm,
        buffer=conversation.summary if conversation.summary else "",
        return_messages=False,
        max_token_limit=200  # Increased token limit
    )

    conversation_chain = ConversationChain(
        prompt=PROMPT_TEMPLATE,
        llm=ollama_client,
        memory=memory,
        verbose=False
    )

    print(f"Conversation_chain: {conversation_chain}")  # For debugging purposes
    # Now use the conversation_chain to generate the response
    response = conversation_chain.invoke({"history": memory.load_memory_variables({})['history'], "input": message_content})
    print(f"Response: {response}")  # For debugging purposes
    response_text = response["response"]

    # Log the user message to the database
    new_message = Message(
        conversation_id=conversation_id,
        sender_id=user_id,
        content=message_content,
        llm_model=MODEL_NAME,
        response_time=1.2  # Example response time
    )
    db.add(new_message)

    # Log the LLM's response to the database
    llm_message = Message(
        conversation_id=conversation_id,
        sender_id=ASSISTANT_UUID,
        content=response_text,
        llm_model=MODEL_NAME,
        response_time=2.0
    )
    db.add(llm_message)

    # Update the memory with the new dialog
    memory.save_context(
        inputs={"input": message_content},
        outputs={"output": response_text}
    )

    # Update conversation summary in the database
    update_conversation_summary(db, conversation, memory)

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
