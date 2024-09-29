import datetime
import hashlib
import logging
import os
import time
from typing import List, Optional
from uuid import uuid4, UUID

from fastapi import HTTPException, UploadFile
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sqlalchemy.orm import Session

from .models import Conversation, Message, Document
from .rag_processing import process_and_store_documents
from .utils import ASSISTANT_UUID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./chroma_db")
DOCUMENTS_DIRECTORY = "./documents"

ollama_client = Ollama(
    base_url=OLLAMA_URL,
    model=MODEL_NAME
)


def upload_user_documents(db: Session, user_id: str, files: List[UploadFile]):
    """
    Handles document uploads for a user, without requiring a conversation.
    """
    user_documents_dir = os.path.join(DOCUMENTS_DIRECTORY, user_id)
    os.makedirs(user_documents_dir, exist_ok=True)

    document_instances = []

    for upload_file in files:
        file_contents = upload_file.file.read()
        checksum = hashlib.sha256(file_contents).hexdigest()

        existing_document = db.query(Document).filter(
            Document.user_id == user_id,
            Document.checksum == checksum
        ).first()

        if existing_document:
            logger.info(f"Document {upload_file.filename} already exists for user {user_id}. Skipping upload.")
            continue

        # Generate the file name in the format {document_id}_{file_name}.{extension}
        new_file_name = f"{uuid4()}_{upload_file.filename}"
        file_path = os.path.join(user_documents_dir, new_file_name)

        with open(file_path, 'wb') as f:
            f.write(file_contents)

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        new_document = Document(
            user_id=user_id,
            file_name=upload_file.filename,
            file_path=file_path,
            upload_time=datetime.datetime.now(datetime.timezone.utc),
            size=file_size_mb,
            checksum=checksum
        )
        db.add(new_document)
        db.flush()

        db.commit()
        db.refresh(new_document)
        document_instances.append(new_document)

    if not document_instances:
        return {"message": "No new documents were uploaded."}

    try:
        process_and_store_documents(document_instances, user_id)
    except Exception as e:
        logger.error(f"Failed to process documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to process documents.")

    return [
        {
            "id": str(doc.id),
            "file_name": doc.file_name,
            "upload_time": doc.upload_time.isoformat(),
            "size": doc.size,
            "checksum": doc.checksum
        }
        for doc in document_instances
    ]


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

    if os.path.exists(document.file_path):
        os.remove(document.file_path)

    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBEDDING_MODEL_NAME
    )
    vectorstore = Chroma(
        collection_name=user_id,
        embedding_function=embeddings,
        persist_directory=CHROMADB_PERSIST_DIRECTORY,
    )

    results = vectorstore._collection.get(
        where={"document_id": document_id}
    )
    ids_to_delete = results['ids']

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)
    else:
        logger.warning(f"No embeddings found for document_id {document_id}")

    conversations_to_update = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.selected_document_ids.any(UUID(document_id))
    ).all()

    for conversation in conversations_to_update:
        conversation.selected_document_ids.remove(UUID(document_id))

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
    title_prompt_template = PromptTemplate(
        template="Generate a short 3-4 word title with an emoji at the start of the title for a conversation based on the following messages. Print ONLY the title.\n"
                 "User: {user_message}\n"
                 "AI: {ai_response}\n"
                 "Title:",
        input_variables=["user_message", "ai_response"]
    )
    chain = title_prompt_template | ollama_client
    title = chain.invoke({"user_message": first_user_message, "ai_response": first_ai_response})
    # Ensure title is short
    title = ' '.join(title.strip().split()[:4])
    return title


def invoke_chain(system_prompt: str, message_history: List[HumanMessage], message_content: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | ollama_client
    messages = message_history + [HumanMessage(content=message_content)]
    try:
        start_time = time.time()
        ai_msg = chain.invoke({"messages": messages})
        end_time = time.time()
        response_text = ai_msg
        response_time = end_time - start_time
        tokens_generated = len(response_text.split())  # Approximate token count
        return response_text, response_time, tokens_generated
    except Exception as e:
        logger.error(f"Failed to generate AI response: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate AI response.")


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

    if selected_documents:
        try:
            selected_document_uuids = [UUID(doc_id) for doc_id in selected_documents]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        user_documents = db.query(Document.id).filter(
            Document.user_id == user_id,
            Document.id.in_(selected_document_uuids)
        ).all()

        user_document_ids = [str(doc.id) for doc in user_documents]

        invalid_document_ids = set(selected_documents) - set(user_document_ids)
        if invalid_document_ids:
            raise HTTPException(status_code=404, detail="One or more documents does not exist")

        conversation.selected_document_ids = selected_document_uuids
        db.commit()
    else:
        selected_document_uuids = [str(doc_id) for doc_id in conversation.selected_document_ids]

    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBEDDING_MODEL_NAME
    )
    vectorstore = Chroma(
        collection_name=user_id,
        embedding_function=embeddings,
        persist_directory=CHROMADB_PERSIST_DIRECTORY,
    )

    try:
        if vectorstore._collection.count() == 0:
            retriever = None
        else:
            if selected_document_uuids:
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3},
                    where={"document_id": {"$in": selected_document_uuids}}
                )
            else:
                retriever = None  # No documents selected
    except Exception as e:
        logger.error(f"Error accessing vector store: {e}")
        retriever = None

    message_history = get_conversation_messages(db, conversation_id, user_id)

    if retriever:
        # Retrieve context
        context_docs = retriever.get_relevant_documents(message_content)
        context_content = "\n\n".join([doc.page_content for doc in context_docs])
        # Prepare system prompt with context
        system_prompt = f"""
        You are Home AI assistant. Your job is to assist house members for question-answering tasks. Your native language is English, but you can speak other languages too.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.
    
        Documents:
        {context_content}
        """
    else:
        # Prepare the system prompt
        system_prompt = """
        You are Home AI assistant. Your job is to assist house members for question-answering tasks. Your native language is English, but you can speak other languages too.
        """

    response_text, response_time, tokens_generated = invoke_chain(system_prompt, message_history, message_content)

    try:
        llm_message = log_message_to_db(db, conversation_id, user_id, message_content, response_text, tokens_generated,
                                        response_time)
    except Exception as e:
        logger.error(f"Failed to log conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to log conversation.")

    if not conversation.title:
        conversation.title = generate_conversation_title(message_content, response_text)
        db.commit()

    return llm_message


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
    return llm_message


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

    db.query(Message).filter(Message.conversation_id == conversation_id).delete()

    conversation.selected_document_ids = []
    db.commit()

    db.delete(conversation)
    db.commit()
    return {"message": "Conversation and related messages deleted successfully"}
