import datetime
import os
import logging
from fastapi import HTTPException
from langchain.chains.llm import LLMChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts.chat import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from sqlalchemy.orm import Session
from fastapi import UploadFile
from .rag_processing import process_and_store_documents
from .models import Conversation, Message, Document
from .utils import ASSISTANT_UUID
from typing import List
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")


ollama_client = Ollama(
    base_url=OLLAMA_URL,
    model=MODEL_NAME
)

TEMPLATE = """
You are Home AI. Your job is to assist house members with their daily tasks. You can speak multiple languages, but your native is english.
Use the following documents, if they exist, to answer the question.

Documents:
{context}

Current conversation:
{chat_history}
Human: {question}
AI:
"""

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=TEMPLATE
)

SIMPLE_TEMPLATE = """
You are Home AI. Your job is to assist house members with their daily tasks. You can speak multiple languages, but your native is english.

Current conversation:
{chat_history}
Human: {input}
AI:
"""

SIMPLE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=SIMPLE_TEMPLATE
)



def upload_documents(db: Session, conversation_id: str, user_id: str, files: List[UploadFile]):
    """
    Handles document uploads during a conversation.
    """
    # Verify the conversation exists and belongs to the user
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
        Conversation.status == "active"
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or closed")

    # Process and store documents
    try:
        process_and_store_documents(files, user_id, conversation_id)
    except Exception as e:
        logger.error(f"Failed to process documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to process documents.")

    # Save document metadata in the database
    for upload_file in files:
        new_document = Document(
            user_id=user_id,
            file_name=upload_file.filename,
            file_url="",  # Update if you store the file
            upload_time=datetime.datetime.now(datetime.timezone.utc),
            conversation_id=conversation_id
        )
        db.add(new_document)
    db.commit()

    return {"message": "Documents uploaded and processed successfully."}

def create_ollama_client():
    return Ollama(
        base_url=OLLAMA_URL,
        model=MODEL_NAME
    )

def get_conversation_messages(db: Session, conversation_id: str, user_id: str):
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp.asc()).all()

    history = []
    for msg in messages:
        if str(msg.sender_id) == user_id:
            history.append(HumanMessage(content=msg.content))
        else:
            history.append(AIMessage(content=msg.content))
    return history

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

    # Initialize embeddings and vector store
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBEDDING_MODEL_NAME
    )
    vectorstore = Chroma(
        collection_name=user_id,
        embedding_function=embeddings,
        persist_directory=CHROMADB_PERSIST_DIRECTORY
    )

    # Check if the vector store has any documents
    try:
        if vectorstore._collection.count() == 0:
            retriever = None
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logger.error(f"Error accessing vector store: {e}")
        retriever = None

    # Load previous messages from the conversation
    message_history = get_conversation_messages(db, conversation_id, user_id)
    chat_history = ChatMessageHistory(messages=message_history)

    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=chat_history
    )

    # Set up the chain
    if retriever:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=ollama_client,
            retriever=retriever,
            memory=memory,
            verbose=False,
            combine_docs_chain_kwargs={'prompt': PROMPT_TEMPLATE}
        )
    else:
        conversation_chain = LLMChain(
            llm=ollama_client,
            prompt=SIMPLE_PROMPT_TEMPLATE,
            memory=memory,
            verbose=False
        )

    # Generate response
    try:
        if retriever:
            response_text = conversation_chain.run(question=message_content)
        else:
            response_text = conversation_chain.predict(input=message_content)
    except Exception as e:
        logger.error(f"Failed to generate AI response: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate AI response.")

    # Log messages to the database
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
