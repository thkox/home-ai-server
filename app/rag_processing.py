import os
import logging
from fastapi import HTTPException
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
from .models import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DOCUMENTS_DIRECTORY = "./documents"

def process_and_store_documents(documents: List[Document], user_id: str):
    # Initialize embedding model
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBEDDING_MODEL_NAME
    )

    # Initialize Chroma vector store
    vectorstore = Chroma(
        collection_name=user_id,
        embedding_function=embeddings,
        persist_directory=CHROMADB_PERSIST_DIRECTORY,
    )

    # Process each document
    for document in documents:
        file_path = document.file_path
        file_extension = os.path.splitext(document.file_name)[1].lower()

        # Use appropriate loader
        try:
            if file_extension == ".txt":
                loader = TextLoader(file_path)
            elif file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path=file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            loaded_documents = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(loaded_documents)

            # Add documents to vector store with metadata
            vectorstore.add_documents(docs, metadata={
                "user_id": user_id,
                "document_id": str(document.id),
                "file_name": document.file_name
            })
        except Exception as e:
            logger.error(f"Failed to process document {document.file_name}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process document {document.file_name}: {e}")
