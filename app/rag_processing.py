import os
import tempfile
import logging
from uuid import uuid4
from fastapi import HTTPException, UploadFile
from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def process_and_store_documents(files: List[UploadFile], user_id: str, conversation_id: str):
    # Initialize embedding model
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBEDDING_MODEL_NAME
    )

    # Initialize Chroma vector store
    vectorstore = Chroma(
        collection_name=user_id,  # Separate collections per user
        embedding_function=embeddings,
        persist_directory=CHROMADB_PERSIST_DIRECTORY
    )

    # Process each file
    for upload_file in files:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=upload_file.filename) as tmp_file:
            tmp_file.write(upload_file.file.read())
            tmp_file_path = tmp_file.name

        # Determine file type
        file_extension = os.path.splitext(upload_file.filename)[1].lower()

        # Use appropriate loader
        try:
            if file_extension == ".txt":
                loader = TextLoader(tmp_file_path)
            elif file_extension == ".pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(tmp_file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path=tmp_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # Add documents to vector store with metadata
            vectorstore.add_documents(docs, metadata={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "file_name": upload_file.filename
            })
        except Exception as e:
            logger.error(f"Failed to process document {upload_file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process document {upload_file.filename}: {e}")
        finally:
            # Clean up temporary file
            os.remove(tmp_file_path)
