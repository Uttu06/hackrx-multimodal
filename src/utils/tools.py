import requests
from src.document_processor.parser import Chunk

def make_http_get_request(url: str) -> str:
    """Makes a GET request to a URL and returns the text content. Handles errors."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error making request: {str(e)}"

def find_info_in_document(query: str, rag_pipeline: 'RAGPipeline', document_url: str, context_override: str = None) -> str:
    """
    Uses the RAG pipeline to find an answer. If context_override is provided,
    it skips parsing and embedding and searches directly on that text.
    """
    print(f"Agent is using RAG tool to find info for: '{query}'")
    
    if context_override:
        print("--> Using fast-path with context_override.")
        # If we have the context, we can bypass the full RAG pipeline and do a mini-RAG.
        # 1. Create a single chunk from the context
        context_chunk = Chunk(text=context_override, page_number=1, source_label="Instruction Manual")
        text_chunks = [context_chunk]

        # 2. Embed this single chunk
        embeddings = rag_pipeline.embedding_model.encode([context_chunk.text])

        # 3. Create a mini-index
        import faiss
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # The rest of the simplified RAG process can now run
        questions_to_process = [query]
        # NOTE: You could extract the core retrieval/prompting logic from your main pipeline
        # into another helper to avoid code duplication. For now, this direct call is fine.
        result = rag_pipeline._execute_rag_pipeline(
            document_url="", 
            questions=[query], 
            timeout=100,
            text_chunks_override=text_chunks,
            index_override=index
        )
    else:
        # Fallback to the full pipeline if no context is provided
        print("--> Using full RAG pipeline.")
        result = rag_pipeline._execute_rag_pipeline(document_url, [query], timeout=100)
    
    if result and result.get("answers"):
        return result["answers"][0]
    return "No answer found."