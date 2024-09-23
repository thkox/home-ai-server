import datetime
import hashlib
import logging
import os
import time
import uuid
from typing import List, Optional

from fastapi import HTTPException, UploadFile
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts.chat import PromptTemplate
from sqlalchemy.orm import Session

from .models import Conversation, Message, Document
from .rag_processing import process_and_store_documents
from .utils import ASSISTANT_UUID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable ChromaDB telemetry
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./chroma_db")
DOCUMENTS_DIRECTORY = "./documents"

ollama_client = Ollama(
    base_url=OLLAMA_URL,
    model=MODEL_NAME
)

TEMPLATE = """
You are Home AI. Your job is to assist house members with their daily tasks. You can speak multiple languages, but your native is English.
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
You are Home AI. Your job is to assist house members with their daily tasks. You can speak multiple languages, but your native is English.

Current conversation:
{chat_history}
Human: {input}
AI:
"""

SIMPLE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=SIMPLE_TEMPLATE
)


def upload_user_documents(db: Session, user_id: str, files: List[UploadFile]):
    """
    Handles document uploads for a user, without requiring a conversation.
    """
    user_documents_dir = os.path.join(DOCUMENTS_DIRECTORY, user_id)
    os.makedirs(user_documents_dir, exist_ok=True)

    document_instances = []

    # Process each file
    for upload_file in files:
        file_contents = upload_file.file.read()
        # Compute SHA256 checksum
        checksum = hashlib.sha256(file_contents).hexdigest()

        # Check if the document already exists for the user
        existing_document = db.query(Document).filter(
            Document.user_id == user_id,
            Document.checksum == checksum
        ).first()

        if existing_document:
            logger.info(f"Document {upload_file.filename} already exists for user {user_id}. Skipping upload.")
            continue  # Do not reupload

        # Generate the file name in the format {document_id}_{file_name}.{extension}
        file_extension = os.path.splitext(upload_file.filename)[1]
        new_file_name = f"{uuid.uuid4()}_{upload_file.filename}"
        file_path = os.path.join(user_documents_dir, new_file_name)

        # Save the file
        with open(file_path, 'wb') as f:
            f.write(file_contents)

        # Calculate file size in MB
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # Create a new Document instance
        new_document = Document(
            user_id=user_id,
            file_name=upload_file.filename,
            file_path=file_path,
            upload_time=datetime.datetime.now(datetime.timezone.utc),
            size=file_size_mb,
            checksum=checksum
        )
        db.add(new_document)
        db.flush()  # Get the document id

        db.commit()
        db.refresh(new_document)
        document_instances.append(new_document)

    if not document_instances:
        return {"message": "No new documents were uploaded."}

    # Process and store documents
    try:
        process_and_store_documents(document_instances, user_id)
    except Exception as e:
        logger.error(f"Failed to process documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to process documents.")

    return {"message": "Documents uploaded and processed successfully."}


def delete_document(db: Session, document_id: str, user_id: str):
    """
    Deletes a document from storage and ChromaDB, and removes it from conversations.
    """
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == user_id
    ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from storage
    if os.path.exists(document.file_path):
        os.remove(document.file_path)

    # Initialize embeddings and vector store
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBEDDING_MODEL_NAME
    )
    vectorstore = Chroma(
        collection_name=user_id,
        embedding_function=embeddings,
        persist_directory=CHROMADB_PERSIST_DIRECTORY,
    )

    # Get the ids of the embeddings associated with the document
    results = vectorstore._collection.get(
        where={"document_id": document_id}
    )
    ids_to_delete = results['ids']

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)
    else:
        logger.warning(f"No embeddings found for document_id {document_id}")

    # Remove document_id from conversations
    conversations_to_update = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.selected_document_ids.any(uuid.UUID(document_id))
    ).all()

    for conversation in conversations_to_update:
        conversation.selected_document_ids.remove(uuid.UUID(document_id))

    # Delete from database
    db.delete(document)
    db.commit()

    return {"message": "Document deleted successfully."}


def list_user_documents(db: Session, user_id: str):
    documents = db.query(Document).filter(
        Document.user_id == user_id
    ).all()
    return documents


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


def generate_conversation_title(first_user_message: str, first_ai_response: str):
    """
    Generates a conversation title using the LLM
    """
    title_prompt = f"Generate a short 3-4 word title for a conversation based on the following messages. Print ONLY the title.\n" \
                   f"User: {first_user_message}\n" \
                   f"AI: {first_ai_response}\n" \
                   f"Title:"
    llm_chain = LLMChain(
        llm=ollama_client,
        prompt=PromptTemplate(template="{input}", input_variables=["input"])
    )
    title = llm_chain.predict(input=title_prompt)
    # Ensure title is short
    title = ' '.join(title.strip().split()[:4])
    return title


def continue_conversation(db: Session, conversation_id: str, user_id: str, message_content: str,
                          selected_documents: Optional[List[str]] = None):
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

    # Validate selected_documents
    if selected_documents:
        # Ensure the documents belong to the user
        user_documents = db.query(Document.id).filter(
            Document.user_id == user_id,
            Document.id.in_(selected_documents)
        ).all()
        user_document_ids = [str(doc.id) for doc in user_documents]

        invalid_document_ids = set(selected_documents) - set(user_document_ids)
        if invalid_document_ids:
            raise HTTPException(status_code=400, detail=f"Invalid document_ids: {invalid_document_ids}")

        # Update conversation's selected_document_ids
        conversation.selected_document_ids = [uuid.UUID(doc_id) for doc_id in selected_documents]
        db.commit()
    else:
        selected_documents = [str(doc_id) for doc_id in conversation.selected_document_ids]

    # Initialize embeddings and vector store
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBEDDING_MODEL_NAME
    )
    vectorstore = Chroma(
        collection_name=user_id,
        embedding_function=embeddings,
        persist_directory=CHROMADB_PERSIST_DIRECTORY,
    )

    # Check if the vector store has any documents
    try:
        if vectorstore._collection.count() == 0:
            retriever = None
        else:
            # Filter documents based on selected_documents
            if selected_documents:
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3},
                    where={"document_id": {"$in": selected_documents}}
                )
            else:
                retriever = None  # No documents selected
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
        start_time = time.time()
        if retriever:
            response_text = conversation_chain.run(question=message_content)
        else:
            response_text = conversation_chain.predict(input=message_content)
        end_time = time.time()
        response_time = end_time - start_time
        tokens_generated = len(response_text.split())  # Approximate token count
    except Exception as e:
        logger.error(f"Failed to generate AI response: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate AI response.")

    # Log messages to the database
    try:
        log_message_to_db(db, conversation_id, user_id, message_content, response_text, tokens_generated, response_time)
    except Exception as e:
        logger.error(f"Failed to log conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to log conversation.")

    # Generate conversation title if not already set
    if not conversation.title:
        # Assuming the first user message and AI response are the ones just processed
        conversation.title = generate_conversation_title(message_content, response_text)
        db.commit()

    return {"llm_response": response_text}


def log_message_to_db(db: Session, conversation_id: str, user_id: str, user_message: str, ai_response: str,
                      tokens_generated: int, response_time: float):
    """
    Logs user and AI messages to the database.
    """
    new_message = Message(
        conversation_id=conversation_id,
        sender_id=user_id,
        content=user_message,
        llm_model=MODEL_NAME,
        tokens_generated=0,
        response_time=0,
    )
    db.add(new_message)

    llm_message = Message(
        conversation_id=conversation_id,
        sender_id=ASSISTANT_UUID,
        content=ai_response,
        llm_model=MODEL_NAME,
        tokens_generated=tokens_generated,
        response_time=response_time
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

    # Delete related messages
    db.query(Message).filter(Message.conversation_id == conversation_id).delete()

    # Remove conversation's selected documents
    conversation.selected_document_ids = []
    db.commit()

    # Delete conversation
    db.delete(conversation)
    db.commit()
    return {"message": "Conversation and related messages deleted successfully"}
