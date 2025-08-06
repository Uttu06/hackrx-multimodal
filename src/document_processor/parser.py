# Document Parser Module
# 
# This module requires Tesseract OCR to be installed on the system.
# Default Windows installation path:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# 
# For other operating systems, adjust the path accordingly:
# - Linux: usually '/usr/bin/tesseract'
# - macOS: usually '/usr/local/bin/tesseract' or '/opt/homebrew/bin/tesseract'

import re
import time
from dataclasses import dataclass
import requests
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure Tesseract path (uncomment and adjust for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


@dataclass
class Chunk:
    """
    Data class to hold structured information about a text chunk.
    
    Attributes:
        text (str): The actual text content of the chunk
        page_number (int): The page number where this chunk originates
        source_label (str): A descriptive label indicating the source location and context
    """
    text: str
    page_number: int
    source_label: str


def parse_document(document_url: str, timeout: int = 300) -> list[Chunk]:
    """
    Downloads and processes a PDF document from a URL, extracting text through
    direct extraction for digital text and OCR for scanned pages, then chunks the text
    with rich metadata. Includes timeout handling to prevent excessive processing time.
    
    Args:
        document_url (str): URL of the PDF document to process
        timeout (int): Maximum time in seconds to spend on document processing
        
    Returns:
        list[Chunk]: List of Chunk objects with text content and metadata
    """
    # Record start time for timeout tracking
    start_time = time.time()
    processing_timeout = 50  # Hardcoded 50-second limit for parsing
    
    try:
        # Download the PDF document
        print(f"Downloading PDF from: {document_url}")
        response = requests.get(document_url, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading document: {e}")
        return []
    
    try:
        # Load the PDF document from downloaded content
        doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")
        total_pages = doc.page_count
        print(f"Successfully loaded PDF with {total_pages} pages")
        
    except Exception as e:
        print(f"Error loading PDF document: {e}")
        return []
    
    # Initialize list to store text and page numbers from all pages
    page_texts: list[tuple[str, int]] = []
    pages_processed = 0
    timeout_reached = False
    
    # Process each page in the document
    for page_num in range(total_pages):
        # Check timeout before processing each page
        elapsed_time = time.time() - start_time
        if elapsed_time > processing_timeout:
            print(f"Timeout reached after {elapsed_time:.1f} seconds. Processed {pages_processed}/{total_pages} pages.")
            timeout_reached = True
            break
            
        try:
            page = doc[page_num]
            print(f"Processing page {page_num + 1}/{total_pages} (elapsed: {elapsed_time:.1f}s)")
            
            # Attempt direct text extraction first
            extracted_text = page.get_text("text")
            
            # Heuristic: if extracted text is less than 100 characters after stripping,
            # assume it's a scanned page or contains mostly images/diagrams
            if len(extracted_text.strip()) < 100:
                # This appears to be a scanned page - use OCR
                print(f"  Page {page_num + 1}: Using OCR (scanned content detected)")
                
                # Check timeout again before expensive OCR operation
                elapsed_time = time.time() - start_time
                if elapsed_time > processing_timeout:
                    print(f"Timeout reached before OCR processing on page {page_num + 1}")
                    timeout_reached = True
                    break
                
                # Render page as high-resolution image
                pixmap = page.get_pixmap(dpi=300)
                
                # Convert pixmap to PIL Image for pytesseract
                img_data = pixmap.tobytes("ppm")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Perform OCR on the image
                ocr_text = pytesseract.image_to_string(pil_image)
                page_texts.append((ocr_text, page_num + 1))
                
            else:
                # Digital text found - use direct extraction
                print(f"  Page {page_num + 1}: Using direct text extraction")
                page_texts.append((extracted_text, page_num + 1))
            
            pages_processed += 1
                
        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")
            # Continue with next page rather than failing completely
            continue
    
    # Close the document to free memory
    doc.close()
    
    # Report processing results
    total_elapsed = time.time() - start_time
    if timeout_reached:
        print(f"Processing stopped due to timeout. Processed {pages_processed}/{total_pages} pages in {total_elapsed:.1f}s")
    else:
        print(f"Processing completed. Processed {pages_processed}/{total_pages} pages in {total_elapsed:.1f}s")
    
    # If no pages were processed, return empty list
    if not page_texts:
        print("No text was extracted from the document")
        return []
    
    # Join all page texts into a single document and create page map
    full_text = ""
    page_map = {}  # Maps character position to page number
    current_pos = 0
    
    for text, page_num in page_texts:
        page_map[current_pos] = page_num
        full_text += text + "\n\n"
        current_pos = len(full_text)
    
    print(f"Extracted total text length: {len(full_text)} characters")
    
    # Define regex pattern for clause/section headings
    clause_pattern = re.compile(r'^\s*([IVX\d]+\.[\d\.]*|[A-Z]\.[\d\.]*|\(\d+\)|[a-z]\))\s+.*', re.MULTILINE)
    
    # Configure text splitter for chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split the full text into chunks
    text_chunks = splitter.split_text(full_text)
    print(f"Generated {len(text_chunks)} text chunks")
    
    # Create Chunk objects with metadata
    chunks = []
    
    for chunk_text in text_chunks:
        # Find the position of this chunk in the full text
        chunk_start = full_text.find(chunk_text)
        
        # Find the page number by looking up the chunk position in page_map
        page_number = 1  # Default to page 1
        for pos, page_num in sorted(page_map.items(), reverse=True):
            if chunk_start >= pos:
                page_number = page_num
                break
        
        # Find the clause by searching backwards from chunk start
        clause_text = "Unknown"
        if chunk_start >= 0:
            # Search backwards from chunk start to find the last clause match
            text_before_chunk = full_text[:chunk_start]
            matches = list(clause_pattern.finditer(text_before_chunk))
            if matches:
                # Get the last match (closest clause heading before this chunk)
                last_match = matches[-1]
                clause_line = last_match.group(0).strip()
                # Extract just the first line of the match for cleaner display
                clause_text = clause_line.split('\n')[0].strip()
        
        # Construct source label
        source_label = f"Page {page_number}, Clause: {clause_text}"
        
        # Create and append Chunk object
        chunk_obj = Chunk(
            text=chunk_text,
            page_number=page_number,
            source_label=source_label
        )
        chunks.append(chunk_obj)
    
    print(f"Created {len(chunks)} chunks with metadata")
    return chunks