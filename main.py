# FastAPI Main Application
# 
# This module serves as the entry point for the Multimodal RAG API,
# providing REST endpoints for document processing and question answering.

import asyncio
import logging
import json
import os
from asyncio.exceptions import TimeoutError
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from urllib.parse import urlparse

# Import custom modules
from src.rag_pipeline.core import RAGPipeline
from src.utils.helpers import run_in_threadpool
from src.document_processor.parser import UnsupportedDocumentError


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


# Define supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.xlsx', '.pptx', '.eml', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

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
    description="A Retrieval-Augmented Generation API for processing various document types and answering questions",
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


def check_file_extension(url: str) -> str:
    """
    Extract and validate file extension from URL before processing.
    
    Args:
        url (str): The document URL
        
    Returns:
        str: The file extension
        
    Raises:
        UnsupportedDocumentError: If the file extension is not supported
    """
    try:
        # Parse the URL and get the path
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Remove query parameters if present
        path = path.split('?')[0]
        
        # Extract file extension
        file_extension = os.path.splitext(path)[-1].lower()
        
        # Check if extension is supported
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedDocumentError(f"Unsupported file type: {file_extension}")
            
        return file_extension
        
    except Exception as e:
        if isinstance(e, UnsupportedDocumentError):
            raise e
        # If we can't parse the extension, assume it's unsupported
        raise UnsupportedDocumentError("Unable to determine file type from URL")


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
    Process a document and answer questions based on its content.
    
    This endpoint handles the complete RAG pipeline:
    1. Validates file extension before download
    2. Downloads and processes the document (PDF, DOCX, XLSX, PPTX, EML, Images)
    3. Generates embeddings and builds search index
    4. Retrieves relevant context for each question
    5. Uses Gemini LLM to generate answers
    
    Args:
        payload (RequestPayload): Request containing document URL and questions
        
    Returns:
        ResponsePayload: Response containing answers to the questions
        JSONResponse: Error response if processing fails
    """
    print(f"Received request to process document: {payload.documents}")
    print(f"Number of questions: {len(payload.questions)}")
    
    # Log the incoming request details to the file
    logging.info(
        f"Received request from IP: {request.client.host} - "
        f"Document URL: {payload.documents} - "
        f"Questions: {json.dumps(payload.questions)}"
    )
    
    # FIRST: Check file extension before any processing
    try:
        file_extension = check_file_extension(payload.documents)
        print(f"File extension validated: {file_extension}")
    except UnsupportedDocumentError as e:
        print(f"Unsupported document type detected before download: {e}")
        
        # Get the file extension to provide a more informative message
        try:
            file_extension = os.path.splitext(payload.documents.split('?')[0])[-1].lower()
            if not file_extension:
                file_extension = "unknown"
        except Exception:
            file_extension = "unknown"

        # Craft the user-friendly response message
        error_message = f"This is a {file_extension} file. The system is only built to handle .pdf, .docx, .xlsx, .pptx, .eml, and image files."

        # Create a list of answers with the same message for every question asked
        answers = [error_message] * len(payload.questions)

        # Return a valid 200 OK response with the informative message
        return ResponsePayload(answers=answers)
    
    # Define request timeout - Extended from 90 to 120 seconds
    REQUEST_TIMEOUT = 120
    
    try:
        # Call the RAG pipeline using threadpool with timeout to avoid blocking the event loop
        # This is crucial for maintaining FastAPI's asynchronous performance
        result = await asyncio.wait_for(
            run_in_threadpool(
                rag_pipeline._main_processing_loop,
                document_url=payload.documents,
                questions=payload.questions,
                timeout=REQUEST_TIMEOUT
            ),
            timeout=REQUEST_TIMEOUT
        )
    except UnsupportedDocumentError as e:
        # This is a fallback in case the RAG pipeline also throws UnsupportedDocumentError
        print(f"Unsupported document type received from RAG pipeline: {e}")
        # Get the file extension to provide a more informative message
        try:
            file_extension = os.path.splitext(payload.documents.split('?')[0])[-1].lower()
            if not file_extension:
                file_extension = "unknown"
        except Exception:
            file_extension = "unknown"

        # Craft the user-friendly response message
        error_message = f"This is a {file_extension} file. The system is only built to handle .pdf, .docx, .xlsx, .pptx, .eml, and image files."

        # Create a list of answers with the same message for every question asked
        answers = [error_message] * len(payload.questions)

        # Return a valid 200 OK response with the informative message
        return ResponsePayload(answers=answers)
    except TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": "Request processing timed out after 120 seconds."}
        )
    except Exception as e:
        # Handle any other exceptions that might occur
        print(f"Unexpected error during processing: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"An unexpected error occurred: {str(e)}"}
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
        "description": "Process various document types and answer questions using RAG with Gemini LLM",
        "endpoints": {
            "GET /": "Health check",
            "POST /hackrx/run": "Process documents and answer questions (requires Bearer token)",
            "GET /info": "API information"
        },
        "supported_formats": [
            "PDF", "DOCX", "XLSX", "PPTX", "EML", 
            "JPG", "JPEG", "PNG", "GIF", "BMP", "TIFF", "WEBP"
        ],
        "unsupported_formats": [
            "ZIP", "RAR", "7Z", "TAR", "GZ", "BIN", "EXE", "DLL"
        ],
        "features": [
            "Direct text extraction",
            "OCR for scanned pages and images",
            "Office document processing",
            "Email parsing",
            "Semantic search with embeddings",
            "Question answering with Gemini Pro"
        ]
    }