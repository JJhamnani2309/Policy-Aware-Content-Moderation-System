"""
Core moderation engine for checking files against policies
"""
import tempfile
from pathlib import Path
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from django.conf import settings
from .llm import get_retrieval_qa_chain
import logging

logger = logging.getLogger('moderation')

CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP

def load_pdf_to_chunks(file_path: str, filename: str) -> List[dict]:
    """
    Load a PDF file, split into chunks, and return a list of dicts.
    
    Args:
        file_path: Path to the PDF file
        filename: Original filename
        
    Returns:
        List of dictionaries containing page_content and metadata
    """
    logger.info(f"Loading PDF: {filename}")
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(docs)
    
    chunk_dicts = []
    for i, chunk in enumerate(chunks):
        metadata = chunk.metadata.copy() if hasattr(chunk, "metadata") else {}
        metadata["chunk_id"] = f"{filename}::chunk_{i}"
        chunk_dicts.append({
            "page_content": chunk.page_content,
            "metadata": metadata
        })
    
    logger.info(f"Split {filename} into {len(chunk_dicts)} chunks")
    return chunk_dicts

def moderate_file_against_policy(policy_store: Chroma, file_path: str, filename: str, k: int = 3) -> Dict:
    """
    For each chunk of the uploaded file:
    - Retrieve top-k policy snippets from the policy store
    - Use LLM (RetrievalQA) with custom moderation prompt to judge
    - Parse 'VIOLATION', 'REVIEW', or 'OK' verdict
    
    Args:
        policy_store: Chroma vectorstore with policy documents
        file_path: Path to the file to moderate
        filename: Original filename
        k: Number of policy chunks to retrieve for each file chunk
        
    Returns:
        Dictionary with moderation results
    """
    logger.info(f"Starting moderation for: {filename}")
    
    # Load file into chunks
    chunks = load_pdf_to_chunks(file_path, filename)
    logger.info(f"Processing {len(chunks)} chunks for moderation")
    
    # Initialize the RetrievalQA chain
    chain = get_retrieval_qa_chain(policy_store, k=k, chain_type="stuff")
    
    violations = []
    allowed_count = 0
    review_count = 0
    violation_count = 0
    
    for idx, chunk in enumerate(chunks):
        query_text = chunk["page_content"].strip()
        if not query_text:
            logger.debug(f"Skipping empty chunk {idx}")
            continue
        
        logger.debug(f"Moderating chunk {idx}/{len(chunks)}: len={len(query_text)}")
        
        try:
            # Send chunk to LLM via RetrievalQA chain
            result = chain({"query": query_text})
            answer = result.get("result", "").strip()
            source_docs = result.get("source_documents", [])
            
            # Parse response based on prompt format
            answer_lower = answer.lower()
            
            if answer_lower.startswith("violation"):
                violation_count += 1
                violations.append({
                    "chunk_id": chunk["metadata"].get("chunk_id"),
                    "chunk_text": query_text[:800],  # Truncate for storage
                    "verdict": "violation",
                    "explanation": answer,
                    "sources": [d.metadata.get("source", "") for d in source_docs]
                })
                logger.info(f"Chunk {idx}: VIOLATION detected")
                
            elif answer_lower.startswith("review"):
                review_count += 1
                violations.append({
                    "chunk_id": chunk["metadata"].get("chunk_id"),
                    "chunk_text": query_text[:800],
                    "verdict": "review",
                    "explanation": answer,
                    "sources": [d.metadata.get("source", "") for d in source_docs]
                })
                logger.info(f"Chunk {idx}: REVIEW required")
                
            elif answer_lower.startswith("ok"):
                allowed_count += 1
                logger.debug(f"Chunk {idx}: OK")
                
            else:
                # Treat unclear responses as needing review
                review_count += 1
                violations.append({
                    "chunk_id": chunk["metadata"].get("chunk_id"),
                    "chunk_text": query_text[:800],
                    "verdict": "review",
                    "explanation": f"REVIEW: Unclear moderation result - {answer}",
                    "sources": []
                })
                logger.warning(f"Chunk {idx}: UNCLEAR verdict, marked for REVIEW")
                
        except Exception as e:
            logger.exception(f"Error moderating chunk {idx}")
            violations.append({
                "chunk_id": chunk["metadata"].get("chunk_id"),
                "chunk_text": query_text[:800],
                "verdict": "error",
                "explanation": str(e),
                "sources": []
            })
    
    total_chunks = len(chunks)
    
    # Determine verdict based on violation and review counts
    if violation_count > 0:
        # If there are violations (with or without reviews), it's a violation
        verdict = "violation_found"
    elif review_count > 0:
        # If there are only reviews (no violations), it needs manual review
        verdict = "needs_review"
    else:
        # If no violations and no reviews, it's clean
        verdict = "clean"
    
    result = {
        "verdict": verdict,
        "total_chunks": total_chunks,
        "allowed_chunks": allowed_count,
        "review_chunks": review_count,
        "violation_chunks": violation_count,
        "violations": violations
    }
    
    logger.info(f"Moderation complete for {filename}: verdict={verdict}, "
                f"violations={violation_count}, reviews={review_count}, "
                f"allowed={allowed_count}/{total_chunks}")
    
    return result