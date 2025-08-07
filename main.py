# FastAPI Main Application
# 
# This module serves as the entry point for the Multimodal RAG API,
# providing REST endpoints for document processing and question answering.

import asyncio
import logging
import json
from asyncio.exceptions import TimeoutError
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import custom modules
from src.rag_pipeline.core import RAGPipeline
from src.utils.helpers import run_in_threadpool


# Define Pydantic Models
class RequestPayload(BaseModel):
    """
    Request payload model for document processing endpoint.
    
    Attributes:
        document_url (str): URL of the PDF document to process
        questions (list[str]): List of questions to answer based on the document
    """
    documents: str
    questions: list[str]


class ResponsePayload(BaseModel):
    """
    Response payload model for successful document processing.
    
    Attributes:
        answers (list[str]): List of answers corresponding to the input questions
    """
    answers: list[str]


# Configure logging
logging.basicConfig(
    filename="api_requests.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'  # 'a' for append
)

# Instantiate FastAPI application
app = FastAPI(
    title="Multimodal RAG API",
    description="A Retrieval-Augmented Generation API for processing PDF documents and answering questions",
    version="1.0.0"
)

# Create HTTPBearer instance for token authentication
security = HTTPBearer()

# Instantiate the RAG Pipeline
# CRITICAL: Create only ONE instance when the application starts up.
# This ensures models are loaded into memory and GPU only once for optimal performance.
print("Initializing RAG Pipeline...")
rag_pipeline = RAGPipeline()
print("RAG Pipeline ready!")


def verify_token(credentials = Depends(security)):
    """
    Verify the Bearer token for authentication.
    
    Args:
        credentials: HTTPAuthorizationCredentials from HTTPBearer
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    if credentials.credentials != '99d35dc664dee13c02ed1b349bf35ab2f820f5adb1b9a158bdf5aa92fab5efec':
        raise HTTPException(status_code=403, detail="Invalid or missing token")


@app.get("/")
async def root():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        dict: Simple status message
    """
    return {"message": "Multimodal RAG API is running!"}


@app.post("/hackrx/run", response_model=ResponsePayload, dependencies=[Depends(verify_token)])
async def process_document(request: Request, payload: RequestPayload):
    """
    Process a PDF document and answer questions based on its content.
    
    This endpoint handles the complete RAG pipeline:
    1. Downloads and processes the PDF document
    2. Generates embeddings and builds search index
    3. Retrieves relevant context for each question
    4. Uses Gemini LLM to generate answers
    
    Args:
        payload (RequestPayload): Request containing document URL and questions
        
    Returns:
        ResponsePayload: Response containing answers to the questions
        dict: Error response if processing fails
    """
    print(f"Received request to process document: {payload.documents}")
    print(f"Number of questions: {len(payload.questions)}")
    
    # Log the incoming request details to the file
    logging.info(
        f"Received request from IP: {request.client.host} - "
        f"Document URL: {payload.documents} - "
        f"Questions: {json.dumps(payload.questions)}"
    )
    
    # Define request timeout - Extended from 90 to 120 seconds
    REQUEST_TIMEOUT = 120
    
    try:
        # Call the RAG pipeline using threadpool with timeout to avoid blocking the event loop
        # This is crucial for maintaining FastAPI's asynchronous performance
        result = await asyncio.wait_for(
            run_in_threadpool(
                rag_pipeline.run,
                document_url=payload.documents,
                questions=payload.questions,
                timeout=REQUEST_TIMEOUT
            ),
            timeout=REQUEST_TIMEOUT
        )
    except TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": "Request processing timed out after 120 seconds."}
        )
    
    # Check if the result contains an error
    if "error" in result:
        error_message = result['error']
        print(f"Error processing document: {error_message}")
        # Return the error result - FastAPI will handle the response appropriately
        return JSONResponse(
        status_code=500,
        content={"detail": error_message}
    )
    
    print("Successfully processed document and generated answers")
    return result


# Additional endpoint for API information (optional but helpful)
@app.get("/info")
async def api_info():
    """
    Get information about the API and its capabilities.
    
    Returns:
        dict: API information and usage details
    """
    return {
        "api_name": "Multimodal RAG API",
        "version": "1.0.0",
        "description": "Process PDF documents and answer questions using RAG with Gemini LLM",
        "endpoints": {
            "GET /": "Health check",
            "POST /hackrx/run": "Process PDF and answer questions (requires Bearer token)",
            "GET /info": "API information"
        },
        "supported_formats": ["PDF"],
        "features": [
            "Direct text extraction",
            "OCR for scanned pages",
            "Semantic search with embeddings",
            "Question answering with Gemini Pro"
        ]
    }