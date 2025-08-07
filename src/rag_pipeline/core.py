import json
import time
import numpy as np
import faiss
import torch
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded, ServiceUnavailable
from sentence_transformers import SentenceTransformer

# Import custom modules
import config
from src.document_processor.parser import parse_document


class ApiKeyManager:
    """
    Manages multiple API keys with automatic rotation and retry logic for resilient API calls.
    """
    
    def __init__(self, api_keys):
        """
        Initialize the API Key Manager.
        
        Args:
            api_keys (list): List of API keys to manage
        """
        if not api_keys:
            raise ValueError("At least one API key must be provided")
        
        self.api_keys = api_keys
        self.current_key_index = 0
        self.total_keys = len(api_keys)
        print(f"ApiKeyManager initialized with {self.total_keys} API keys")
        
        # Initialize and configure the genai model for the first key
        genai.configure(api_key=self._get_current_key())
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    def _get_current_key(self):
        """Get the current API key."""
        return self.api_keys[self.current_key_index]
    
    def _rotate_key(self):
        """Rotate to the next API key."""
        self.current_key_index = (self.current_key_index + 1) % self.total_keys
        print(f"Rotated to API key {self.current_key_index + 1}/{self.total_keys}")
        
        # Re-initialize the model with the new key
        genai.configure(api_key=self._get_current_key())
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    def call_llm(self, prompt):
        """
        Make a resilient API call with automatic retry logic for RPD and RPM errors.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            Response object from the LLM
            
        Raises:
            Exception: If all retry attempts fail
        """
        max_retries_per_key = 3
        rpm_delay = 15  # Wait 15 seconds for RPM (Requests Per Minute) limit
        rpd_delay = 3600  # Wait 1 hour for RPD (Requests Per Day) limit
        
        for key_attempt in range(self.total_keys):
            for retry_attempt in range(max_retries_per_key):
                try:
                    print(f"Making API call with key {self.current_key_index + 1}/{self.total_keys}, attempt {retry_attempt + 1}/{max_retries_per_key}")
                    response = self.model.generate_content(prompt)
                    print("API call successful")
                    return response
                    
                except ResourceExhausted as e:
                    error_message = str(e).lower()
                    print(f"ResourceExhausted error: {error_message}")
                    
                    if "per minute" in error_message or "rpm" in error_message:
                        print(f"RPM limit reached. Waiting {rpm_delay} seconds before retry...")
                        time.sleep(rpm_delay)
                        continue  # Retry with the same key after a short delay
                        
                    elif "per day" in error_message or "rpd" in error_message:
                        print(f"RPD limit reached for key {self.current_key_index + 1}. Rotating to next key...")
                        self._rotate_key()
                        break  # Break inner loop to try the next key immediately
                        
                    else:  # Handle other temporary resource issues with backoff
                        wait_time = min(2 ** retry_attempt, 30)
                        print(f"Unknown ResourceExhausted error. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                except (DeadlineExceeded, ServiceUnavailable) as e:
                    print(f"Temporary API error: {type(e).__name__}: {e}")
                    wait_time = min(2 ** retry_attempt, 60)  # Exponential backoff, max 60 seconds
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                    
                except Exception as e:
                    print(f"Unexpected error during API call: {type(e).__name__}: {e}")
                    if retry_attempt < max_retries_per_key - 1:
                        wait_time = min(2 ** retry_attempt, 30)  # Exponential backoff
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("Max retries reached for current key")
                        break
            
            # If we're here, all retries for current key failed, try next key
            if key_attempt < self.total_keys - 1:
                print(f"All retries failed for key {self.current_key_index + 1}. Trying next key...")
                self._rotate_key()
            else:
                print("All API keys exhausted")
        
        # If we reach here, all keys and retries have been exhausted
        raise Exception("All API keys and retry attempts have been exhausted")


class RAGPipeline:
    """
    Main RAG Pipeline class that orchestrates document processing, embedding,
    retrieval, and question answering using Gemini LLM.
    """
    
    def __init__(self):
        """
        Initialize the RAG pipeline by loading models and configuring APIs.
        """
        print("Loading models...")
        
        # Dynamic device detection for CUDA availability
        if torch.cuda.is_available():
            device = 'cuda'
            print("CUDA-enabled GPU detected. Using GPU acceleration.")
        else:
            device = 'cpu'
            print("No CUDA-enabled GPU found. Using CPU.")
        
        # Load embedding model with dynamic device selection
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
        print(f"Loaded embedding model: {config.EMBEDDING_MODEL_NAME} on {device}")
        
        # Initialize API Key Manager (it handles all genai configuration internally)
        if not config.GEMINI_API_KEY_LIST:
            raise ValueError("No Gemini API keys found. Please set GEMINI_API_KEYS environment variable.")
        
        self.api_key_manager = ApiKeyManager(config.GEMINI_API_KEY_LIST)
        print("Configured Gemini API and model with ApiKeyManager")
        
        print("RAG Pipeline initialization complete!")
    
    def _generate_search_queries_for_batch(self, questions: list[str]) -> list[str]:
        """
        Generate alternative search queries for a batch of questions to improve context retrieval.
        
        Args:
            questions (list[str]): List of original questions
            
        Returns:
            list[str]: List of all queries (original + generated alternatives)
        """
        print(f"Generating alternative search queries for {len(questions)} questions...")
        
        # Create numbered list of questions for the prompt
        formatted_questions = "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions)])
        
        multi_query_prompt = f"""You are an expert at generating diverse search queries. For each question below, generate exactly 3 alternative search queries that could help find relevant information to answer the original question.

The alternative queries should:
- Use different keywords and phrasings
- Cover different aspects of the question
- Be concise and focused
- Help retrieve comprehensive information

QUESTIONS:
{formatted_questions}

Output your response as a single, clean JSON object where each key is "question_X" (where X is the question number) and each value is a list of exactly 3 alternative search queries.

Example format:
{{
  "question_1": ["alternative query 1", "alternative query 2", "alternative query 3"],
  "question_2": ["alternative query 1", "alternative query 2", "alternative query 3"]
}}

Your response must be only the JSON object, no additional text."""
        
        try:
            # Use resilient API call
            response = self.api_key_manager.call_llm(multi_query_prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = response_text[start_idx:end_idx + 1]
            else:
                print("Warning: Could not find JSON in multi-query response, using original questions only")
                return questions.copy()
            
            try:
                parsed_queries = json.loads(json_content)
                
                # Combine original questions with generated alternatives
                all_queries = []
                
                for i, original_question in enumerate(questions):
                    # Add original question
                    all_queries.append(original_question)
                    
                    # Add alternative queries if available
                    question_key = f"question_{i+1}"
                    if question_key in parsed_queries and isinstance(parsed_queries[question_key], list):
                        alternatives = parsed_queries[question_key]
                        all_queries.extend(alternatives)
                        print(f"Generated {len(alternatives)} alternative queries for question {i+1}")
                    else:
                        print(f"Warning: No alternatives found for question {i+1}, using original only")
                
                print(f"Multi-query generation successful: {len(questions)} original + {len(all_queries) - len(questions)} generated = {len(all_queries)} total queries")
                return all_queries
                
            except json.JSONDecodeError as e:
                print(f"Error parsing multi-query JSON response: {e}")
                print("Falling back to original questions only")
                return questions.copy()
                
        except Exception as e:
            print(f"Error generating alternative queries: {e}")
            print("Falling back to original questions only")
            return questions.copy()
    
    
    def _is_complex_question(self, question: str) -> bool:
        """
        Determine if a question requires complex reasoning based on keywords.
        
        Args:
            question (str): The question to analyze
            
        Returns:
            bool: True if the question is likely complex, False otherwise
        """
        complex_keywords = ["describe", "explain", "synthesize", "compare", "what are two", "what is the difference"]
        return any(keyword in question.lower() for keyword in complex_keywords)
    
    def _preprocess_and_route_questions(self, questions: list[str]) -> list[dict]:
        """
        Analyzes a list of questions, classifies them, and deconstructs complex ones.
        """
        print(f"Pre-processing {len(questions)} questions with the router...")
        
        prompt = f"""You are an expert query analysis agent. For the following list of user questions, analyze each one and classify it.
        Your tasks are:
        1. Classify each question into one of three types: "VALID_SIMPLE", "VALID_COMPOUND", or "INVALID".
        2. For "VALID_COMPOUND" questions, deconstruct them into a list of simple, single-focus sub-questions.
        3. For "INVALID" questions (e.g., adversarial, out-of-scope, requesting code, unsafe content), provide a brief, polite rejection_reason.
        
        Return your analysis as a single, clean JSON list of objects, with no other text or explanation.

        QUESTIONS: {json.dumps(questions)}

        EXAMPLE OUTPUT:
        [
          {{"original_question": "What is the waiting period for Hydrocele?", "classification": "VALID_SIMPLE", "sub_questions": ["What is the waiting period for Hydrocele?"]}},
          {{"original_question": "Explain the dental claim process and provide the grievance email.", "classification": "VALID_COMPOUND", "sub_questions": ["What is the process for submitting a dental claim?", "What is the grievance redressal email?"]}},
          {{"original_question": "Give me the source code for the test cases.", "classification": "INVALID", "rejection_reason": "The question is out of scope and cannot be answered from the provided document."}}
        ]
        """
        
        try:
            response = self.api_key_manager.call_llm(prompt)
            response_text = response.text.strip()
            json_start = response_text.find('[')
            json_end = response_text.rfind(']')
            if json_start != -1 and json_end != -1:
                json_content = response_text[json_start:json_end + 1]
                return json.loads(json_content)
            else:
                print("Warning: Router failed to return valid JSON. Processing questions as-is.")
                return [{ "original_question": q, "classification": "VALID_SIMPLE", "sub_questions": [q] } for q in questions]
        except Exception as e:
            print(f"Error in router, falling back to simple processing: {e}")
            return [{ "original_question": q, "classification": "VALID_SIMPLE", "sub_questions": [q] } for q in questions]
    
    def run(self, document_url: str, questions: list[str], timeout: int = 55) -> dict:
        """
        Main entry point for processing a document and answering questions.
        
        Args:
            document_url (str): URL of the PDF document to process
            questions (list[str]): List of questions to answer
            timeout (int): Maximum time in seconds for processing (default: 55)
            
        Returns:
            dict: Dictionary containing answers or error information
        """
        
        # Record start time for timeout management
        start_time = time.time()
        
        # Initialize list to store final answers
        final_answers = []
        
        # Step 1: Process Document
        print("Step 1: Processing document...")
        text_chunks = parse_document(document_url, timeout=timeout)
        
        if not text_chunks:
            print("Error: Failed to process document")
            return {"error": "Failed to process document."}
        
        print(f"Successfully extracted {len(text_chunks)} text chunks")
        
        # Step 2: Embed Chunks and Build Index
        print("Step 2: Generating embeddings and building search index...")
        
        # Embed all text chunks (extract text from Chunk objects)
        print("Generating embeddings...")
        chunk_texts = [chunk.text for chunk in text_chunks]
        embeddings = self.embedding_model.encode(chunk_texts)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Create FAISS index for similarity search
        index = faiss.IndexFlatL2(embeddings.shape[1])
        
        # Add embeddings to the index
        index.add(embeddings)
        print(f"Built FAISS index with {index.ntotal} vectors")
        
        # Create a reliable map from each chunk's unique memory ID to its list index.
        # This is faster and more reliable than list.index() for objects.
        chunk_id_to_index_map = {id(chunk): i for i, chunk in enumerate(text_chunks)}
        
        # Check if we have enough time left for question processing
        if time.time() - start_time > timeout - 40:
            print("Error: Document processing took too long, no time left for question answering")
            return {"error": "Document processing exceeded time limit, unable to answer questions."}
        
        # Step 3: Pre-process questions using the router
        tasks = self._preprocess_and_route_questions(questions)

        final_answers_map = {}
        questions_to_process = []
        task_map = {}

        # Separate valid questions from invalid ones
        for task in tasks:
            original_q = task['original_question']
            if task['classification'] == 'INVALID':
                final_answers_map[original_q] = task.get('rejection_reason', "This question cannot be answered.")
            else:
                # Map sub-questions back to their original parent question
                for sub_q in task['sub_questions']:
                    questions_to_process.append(sub_q)
                    task_map[sub_q] = original_q

        # Step 4: Process Valid Questions in Batches
        print(f"Processing {len(questions_to_process)} valid sub-questions in batches...")
        sub_question_answers = {}

        i = 0
        batch_number = 1
        while i < len(questions_to_process):
            # Check for timeout before each batch
            if time.time() - start_time > timeout - 40:  # 40-second buffer
                print("Time limit approaching, ending question processing.")
                break

            # Use the simple batching logic from before, on the clean 'questions_to_process' list
            question_batch = questions_to_process[i:i+3]
            
            # Generate alternative search queries for improved retrieval
            all_search_queries = self._generate_search_queries_for_batch(question_batch)
            
            # b. Gather Context for the Batch using all queries (original + generated)
            batch_context_chunks = []
            
            print(f"Performing vector search with {len(all_search_queries)} queries...")
            for query in all_search_queries:
                # Perform vector search for this query
                query_embedding = self.embedding_model.encode([query])
                distances, indices = index.search(query_embedding, config.TOP_K_RESULTS)
                
                # Extend batch context with results for this query
                for idx in indices[0]:
                    batch_context_chunks.append(text_chunks[idx])
            
            # c. Get Initial Indices and Implement Contextual Straddling
            enriched_indices = set()
            # Get the original indices using our reliable map, which avoids the .index() bug
            retrieved_indices = {chunk_id_to_index_map[id(c)] for c in batch_context_chunks}

            for current_index in retrieved_indices:
                # Add the main chunk's index
                enriched_indices.add(current_index)
                # Add the preceding chunk's index if it's valid
                if current_index - 1 >= 0:
                    enriched_indices.add(current_index - 1)
                # Add the succeeding chunk's index if it's valid
                if current_index + 1 < len(text_chunks):
                    enriched_indices.add(current_index + 1)

            # d. De-duplicate and Build Final Context from sorted indices to preserve document order
            # Sort the final indices to maintain a logical flow of context
            sorted_indices = sorted(list(enriched_indices))
            unique_chunks = [text_chunks[i] for i in sorted_indices]

            print(f"Enriched and straddled context: {len(retrieved_indices)} initial -> {len(unique_chunks)} final chunks for batch {batch_number}")
            
            # Build full context string for this batch
            context_list = []
            for chunk in unique_chunks:
                context_list.append(f"--- START OF SOURCE: {chunk.source_label} ---\n{chunk.text}\n--- END OF SOURCE: {chunk.source_label} ---")
            
            full_context = "\n\n---\n\n".join(context_list)
            
            # Build the Prompt
            formatted_questions_batch = "\n".join([f"{j+1}. {question}" for j, question in enumerate(question_batch)])
            
            prompt = f"""You are a world-class document analysis system acting as a claims auditor. Your task is to precisely answer a list of questions based ONLY on the provided context, and to cite the source for every piece of information.

First, you will think step-by-step for each question in the batch. Enclose your entire reasoning process within <reasoning> tags.

After your reasoning, you will output the final answers. The final output must be a single, valid JSON object enclosed in ```json tags. The JSON must contain a single key "answers" which is a list of strings. The number of answers in the list must exactly match the number of questions.

**CRITICAL INSTRUCTION:** Every sentence in every answer must end with a citation pointing to the exact source. For example: `[Page 12, Clause: VI. A. WAITING PERIODS]`.

**Expert Behavior Mandate:**
- Pay close attention to definitions, exclusions, and conditions specified in the clauses.
- Synthesize information from multiple sources if a question requires it, but explicitly state how the sources connect in your reasoning.
- Do not make assumptions beyond what is explicitly stated in the context provided.

CONTEXT:
---
{full_context}
---

QUESTIONS:
---
{formatted_questions_batch}
---

EXAMPLE OUTPUT:
<reasoning>
Question 1: What is the waiting period? I found the answer in the context labeled "Page 12, Clause: VI. A. WAITING PERIODS". The text says 48 months.
Question 2: What is the policy number? The context does not contain the policy number. The answer is not found.
</reasoning>
```json
{{
  "answers": [
    "The waiting period for Pre-Existing Diseases is 48 months. [Page 12, Clause: VI. A. WAITING PERIODS]",
    "Answer not found in the provided context."
  ]
}}
```

**YOUR TASK:** Now, generate the reasoning and the final JSON output for the provided context and questions."""
            
            # Make ONE API Call
            try:
                print(f"Calling LLM for batch {batch_number}...")
                response = self.api_key_manager.call_llm(prompt)
                response_text = response.text
                print(f"Received response from LLM for batch {batch_number}")
                
                # Parse and Store Results in sub_question_answers
                # Extract only the JSON block from the response, ignoring the <reasoning> section
                json_start = response_text.find('```json')
                json_end = response_text.find('```', json_start + 7)
                
                if json_start != -1 and json_end != -1:
                    # Extract JSON content between ```json and ```
                    json_content = response_text[json_start + 7:json_end].strip()
                else:
                    # Fallback: try to find JSON object directly
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        json_content = response_text[start_idx:end_idx + 1]
                    else:
                        json_content = response_text
                
                try:
                    parsed_response = json.loads(json_content)
                    
                    # Extract answers and store in sub_question_answers dictionary
                    if "answers" in parsed_response and isinstance(parsed_response["answers"], list):
                        batch_answers = parsed_response["answers"]
                        for idx, answer in enumerate(batch_answers):
                            if idx < len(question_batch):
                                sub_question_answers[question_batch[idx]] = answer
                        print(f"Successfully processed {len(batch_answers)} answers for batch {batch_number}")
                    else:
                        # Add error messages for each question in the batch
                        for question in question_batch:
                            sub_question_answers[question] = "Error: No answers found in LLM response."
                        print(f"Warning: No answers found in response for batch {batch_number}")
                        
                except json.JSONDecodeError as e:
                    error_msg = f"Failed to parse LLM response as JSON: {str(e)}"
                    # Add error message for each question in the batch
                    for question in question_batch:
                        sub_question_answers[question] = f"Error: {error_msg}"
                    print(f"Error parsing JSON for batch {batch_number}: {e}")
                    print(f"Attempted to parse: {json_content[:200]}...")  # Show first 200 chars for debugging
            
            except Exception as e:
                error_msg = f"Failed to generate response: {str(e)}"
                # Add error message for each question in the batch
                for question in question_batch:
                    sub_question_answers[question] = f"Error: {error_msg}"
                print(f"Error calling LLM for batch {batch_number}: {e}")
            
            # Move to the next batch
            i += len(question_batch)
            batch_number += 1

        # Step 5: Re-assemble Final Answers
        print("Re-assembling final answers from processed tasks...")
        final_answers_list = []
        for original_q in questions:  # Iterate through original questions to preserve order
            # Find the task for this original question
            task = next((t for t in tasks if t['original_question'] == original_q), None)
            if task:
                if task['classification'] == 'INVALID':
                    final_answers_list.append(final_answers_map[original_q])
                else:
                    # Collect and join answers for all sub-questions
                    answers_for_this_task = [sub_question_answers.get(sq, "Processing timed out before this question could be answered.") for sq in task['sub_questions']]
                    final_answers_list.append(" ".join(answers_for_this_task))

        print("Successfully processed all tasks.")
        return {"answers": final_answers_list}