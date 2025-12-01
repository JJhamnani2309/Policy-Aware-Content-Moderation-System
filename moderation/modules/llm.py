"""
LLM and RetrievalQA chain management
"""
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from django.conf import settings
import logging

logger = logging.getLogger('moderation')

GROQ_API_KEY = settings.GROQ_API_KEY

MODERATION_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI content moderation system. Your task is to determine "
        "if the given text violates any of the company's policy documents.\n\n"
        "=== POLICY CONTEXT ===\n"
        "{context}\n\n"
        "=== TEXT TO CHECK ===\n"
        "{question}\n\n"
        "Please respond STRICTLY in one of the following three formats:\n"
        "1. 'VIOLATION: <brief explanation of which policy it violates and why>'\n"
        "2. 'REVIEW: <brief explanation of why it needs human review>'\n"
        "3. 'OK: <brief reason why it is compliant>'\n\n"
        "Be concise but explicit in your reasoning. IMPORTANT - DONT JUST CLASSIFY ALL SLIGHTLY VIOLATING FILES INTO VIOLATION, PUT SOME INTO REVIEW AS WELL, ALL WHICH ARENT AN EXTREME VIOLATION MUST GO INTO REVIEW"
    )
)

def get_retrieval_qa_chain(vectorstore, k: int = 3, chain_type: str = "stuff"):
    """
    Return a RetrievalQA chain using Groq LLM and the provided vectorstore retriever.
    
    Args:
        vectorstore: Chroma vectorstore instance
        k: Number of documents to retrieve
        chain_type: Type of chain to use
        
    Returns:
        RetrievalQA chain instance
    """
    logger.info(f"Initializing RetrievalQA chain with k={k}")
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in environment variables")
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=512
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        chain_type_kwargs={"prompt": MODERATION_PROMPT},
        return_source_documents=True
    )
    
    logger.info("RetrievalQA moderation chain initialized successfully")
    return chain

def query_chain(chain, user_input: str) -> dict:
    """
    Run a moderation query against the RetrievalQA chain.
    
    Args:
        chain: RetrievalQA chain instance
        user_input: Text to moderate
        
    Returns:
        Dictionary with response and sources
    """
    try:
        logger.debug(f"Running moderation chain for input: {user_input[:200]}...")
        result = chain({"query": user_input})
        
        response = {
            "response": result["result"],
            "sources": [
                doc.metadata.get("source", "")
                for doc in result.get("source_documents", [])
            ]
        }
        
        logger.debug(f"Chain response: {response['response'][:100]}...")
        return response
    except Exception as e:
        logger.exception("Error in query_chain")
        raise