"""
policy_store.py
Policy store management using Chroma vector database
"""
import os
import shutil
from pathlib import Path
from typing import List
from django.conf import settings
import logging

# Updated imports (fixing all LangChain deprecation warnings)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger('moderation')

POLICY_STORE_DIR = str(settings.POLICY_STORE_DIR)
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP


# ----------------------------
# Windows-safe folder deletion
# ----------------------------
def safe_delete_folder(path: str):
    """Windows-safe delete to bypass Chroma locking files."""
    if not os.path.exists(path):
        return

    # Walk bottom-up and remove files individually
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            fp = os.path.join(root, f)
            try:
                os.remove(fp)
            except PermissionError:
                # Try a reopen-close to release Windows lock
                try:
                    with open(fp, "rb") as tmp:
                        tmp.read()
                    os.remove(fp)
                except:
                    pass

        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except:
                pass

    try:
        os.rmdir(path)
    except:
        pass


def get_embeddings():
    """Returns embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_or_update_policy_store(file_paths: List[str]) -> Chroma:
    """
    Load policy PDFs, split, embed, and store in Chroma.
    Appends to existing store when present.
    """
    logger.info(f"Building/updating policy store with {len(file_paths)} files")

    # Load all PDF documents
    docs = []
    for file_path in file_paths:
        try:
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
            logger.debug(f"Loaded {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    # Split into text chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()

    # Update existing store OR create a new one
    if os.path.exists(POLICY_STORE_DIR) and os.listdir(POLICY_STORE_DIR):
        logger.info("Updating existing policy store")
        store = Chroma(
            persist_directory=POLICY_STORE_DIR,
            embedding_function=embeddings,
        )
        store.add_documents(chunks)
        store.persist()
    else:
        logger.info("Creating new policy store")
        store = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=POLICY_STORE_DIR,
        )
        store.persist()

    logger.info("Policy store updated successfully")
    return store


def load_policy_store() -> Chroma:
    """Loads existing Chroma database."""
    embeddings = get_embeddings()

    if os.path.exists(POLICY_STORE_DIR) and os.listdir(POLICY_STORE_DIR):
        return Chroma(
            persist_directory=POLICY_STORE_DIR,
            embedding_function=embeddings,
        )
    else:
        logger.error("Policy store is empty")
        raise FileNotFoundError("Policy store is empty. Upload policy PDFs first.")


def clear_policy_store() -> bool:
    """
    Full reset of Chroma policy store.
    Handles Windows file-locking gracefully.
    """
    logger.info("Clearing policy store")

    if os.path.exists(POLICY_STORE_DIR):
        safe_delete_folder(POLICY_STORE_DIR)
        logger.info("Policy store directory deleted safely")

    # Recreate as empty
    os.makedirs(POLICY_STORE_DIR, exist_ok=True)
    logger.info("Policy store cleared successfully")

    return True


def policy_store_exists() -> bool:
    """Checks if policy store exists and contains data."""
    return os.path.exists(POLICY_STORE_DIR) and bool(os.listdir(POLICY_STORE_DIR))
