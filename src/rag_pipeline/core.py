import json
import time
import numpy as np
import faiss
import torch
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded, ServiceUnavailable
from sentence_transformers import SentenceTransformer
import os
from urllib.parse import urlparse
import requests
import re

# Import custom modules
import config
from src.document_processor.parser import parse_document, UnsupportedDocumentError, DocumentProcessingError
from src.utils import tools


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
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    def _get_current_key(self):
        """Get the current API key."""
        return self.api_keys[self.current_key_index]
    
    def _rotate_key(self):
        """Rotate to the next API key."""
        self.current_key_index = (self.current_key_index + 1) % self.total_keys
        print(f"Rotated to API key {self.current_key_index + 1}/{self.total_keys}")
        
        # Re-initialize the model with the new key
        genai.configure(api_key=self._get_current_key())
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def call_llm(self, prompt, image_url=None, timeout=30):
        """
        Make a resilient API call with automatic retry logic for RPD and RPM errors.
        
        Args:
            prompt (str): The prompt to send to the LLM
            image_url (str, optional): URL of image for multimodal inputs
            timeout (int): Timeout for individual API call
            
        Returns:
            Response object from the LLM
            
        Raises:
            Exception: If all retry attempts fail
        """
        max_retries_per_key = 2  # Reduced retries for faster failover
        rpm_delay = 10  # Reduced wait time for RPM limit
        rpd_delay = 5  # Reduced wait time for RPD limit (will rotate keys instead)
        
        for key_attempt in range(self.total_keys):
            for retry_attempt in range(max_retries_per_key):
                try:
                    print(f"Making API call with key {self.current_key_index + 1}/{self.total_keys}, attempt {retry_attempt + 1}/{max_retries_per_key}")
                    
                    if image_url:
                        # Multimodal call with image
                        import PIL.Image
                        import requests
                        from io import BytesIO
                        
                        # Download and prepare the image with timeout
                        response = requests.get(image_url, timeout=timeout//2)
                        image = PIL.Image.open(BytesIO(response.content))
                        
                        # Make multimodal API call
                        response = self.model.generate_content([prompt, image], request_options={"timeout": timeout})
                    else:
                        # Text-only call
                        response = self.model.generate_content(prompt, request_options={"timeout": timeout})
                    
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
                        wait_time = min(2 ** retry_attempt, 15)  # Reduced max wait
                        print(f"Unknown ResourceExhausted error. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                except (DeadlineExceeded, ServiceUnavailable) as e:
                    print(f"Temporary API error: {type(e).__name__}: {e}")
                    wait_time = min(2 ** retry_attempt, 30)  # Reduced max wait
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                    
                except Exception as e:
                    print(f"Unexpected error during API call: {type(e).__name__}: {e}")
                    if retry_attempt < max_retries_per_key - 1:
                        wait_time = min(2 ** retry_attempt, 15)  # Reduced max wait
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
    Unified Web-Enabled RAG Agent that intelligently combines traditional RAG with 
    tool-using agentic workflows for comprehensive document understanding.
    """
    
    def __init__(self):
        """
        Initialize the unified RAG agent by loading models and configuring APIs.
        """
        print("Loading unified web-enabled RAG agent...")
        
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
        
        # Initialize secure toolbox for agentic workflow
        self.tools = {
            "make_http_get_request": self._secure_http_request,
            "find_info_in_document": tools.find_info_in_document
        }
        
        print("Unified RAG Agent initialization complete!")
    
    def _secure_http_request(self, url: str) -> str:
        """
        Secure HTTP GET request with relaxed restrictions for puzzle/hackathon APIs.
        
        Args:
            url (str): URL to request
            
        Returns:
            str: Response content or error message
        """
        print(f"Making secure HTTP request to: {url}")
        
        try:
            # Add User-Agent to appear more like a regular browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Make HTTP request with timeout and headers
            response = requests.get(url, timeout=15, allow_redirects=True, headers=headers)
            
            # Allow more HTTP status codes for APIs (not just 200)
            if response.status_code in [200, 201, 202]:
                print(f"Successfully retrieved content from URL ({len(response.text)} chars)")
                return response.text
            else:
                error_msg = f"API returned status {response.status_code}: {response.reason}"
                print(f"API Status Error: {error_msg}")
                
                # Still return the response text in case it contains useful error info
                if response.text:
                    return f"Status {response.status_code}: {response.text}"
                else:
                    return error_msg
                
        except requests.RequestException as e:
            error_msg = f"Failed to access URL: {str(e)}"
            print(f"Request error: {error_msg}")
            return error_msg
    
    def _is_image_file(self, document_url: str) -> bool:
        """
        Check if the document URL points to an image file.
        
        Args:
            document_url (str): URL of the document
            
        Returns:
            bool: True if the URL points to an image file, False otherwise
        """
        parsed_url = urlparse(document_url)
        file_path = parsed_url.path.lower()
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff']
        return any(file_path.endswith(ext) for ext in image_extensions)
    
    def _classify_image_content(self, document_url: str) -> str:
        """
        Classify image content to determine if it's document-like or abstract.
        
        Args:
            document_url (str): URL of the image to classify
            
        Returns:
            str: 'Document' if image contains significant text/tables, 'Abstract' otherwise
        """
        print("Classifying image content...")
        
        classification_prompt = ("Analyze this image and classify its content. "
                               "Respond with a single word: 'Document' if it contains significant text, "
                               "tables, or looks like a document, or 'Abstract' otherwise.")
        
        try:
            response = self.api_key_manager.call_llm(classification_prompt, image_url=document_url, timeout=20)
            classification = response.text.strip().lower()
            
            # Normalize the response
            if 'document' in classification:
                result = 'Document'
            elif 'abstract' in classification:
                result = 'Abstract'
            else:
                # Default to 'Abstract' if unclear
                result = 'Abstract'
                
            print(f"Image classified as: {result}")
            return result
            
        except Exception as e:
            print(f"Error classifying image content: {e}")
            # Default to 'Abstract' on error to be safe
            return 'Abstract'
    
    def _generate_image_description(self, document_url: str, questions: list[str]) -> dict:
        """
        Generate a simple description of an abstract image.
        
        Args:
            document_url (str): URL of the image to describe
            questions (list[str]): List of questions (for consistency in return format)
            
        Returns:
            dict: Dictionary with answers list containing the same description for all questions
        """
        print("Generating description for abstract image...")
        
        description_prompt = "Describe this image in one or two sentences."
        
        try:
            response = self.api_key_manager.call_llm(description_prompt, image_url=document_url, timeout=20)
            description = response.text.strip()
            print(f"Generated image description: {description[:100]}...")
            
            # Return the same description as answer to all questions
            answers = [description] * len(questions)
            return {"answers": answers}
            
        except Exception as e:
            error_msg = f"Error generating image description: {str(e)}"
            print(error_msg)
            # Return error message for all questions
            answers = [f"Error: {error_msg}"] * len(questions)
            return {"answers": answers}
    
    def _translate_to_english(self, text: str) -> str:
        """Translates a given text to English using a fast LLM call."""
        if not text.strip():
            return ""
        try:
            # Simple check to avoid translating English
            text.encode(encoding='utf-8').decode('ascii')
            return text
        except UnicodeDecodeError:
            # Text is not pure ASCII, likely needs translation
            print(f"Translating non-English text: {text[:50]}...")
            prompt = f"Translate the following text to English. Respond with ONLY the translated text and nothing else:\n\n{text}"
            try:
                response = self.api_key_manager.call_llm(prompt, timeout=10)
                translated_text = response.text.strip()
                print(f"Translation successful: {translated_text[:50]}...")
                return translated_text
            except Exception as e:
                print(f"Translation failed: {e}")
                return text  # Return original text on failure
    
    def _generate_search_queries_for_batch(self, questions: list[str], remaining_time: float) -> list[str]:
        """
        Generate alternative search queries for a batch of questions to improve context retrieval.
        
        Args:
            questions (list[str]): List of original questions
            remaining_time (float): Remaining processing time
            
        Returns:
            list[str]: List of all queries (original + generated alternatives)
        """
        # Skip query generation if we're running low on time
        if remaining_time < 15:
            print("Low on time, skipping query generation")
            return questions.copy()
        
        print(f"Generating alternative search queries for {len(questions)} questions...")
        
        # Create numbered list of questions for the prompt
        formatted_questions = "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions)])
        
        multi_query_prompt = f"""You are an expert at generating diverse search queries. For each question below, generate exactly 2 alternative search queries that could help find relevant information to answer the original question.

The alternative queries should:
- Use different keywords and phrasings
- Cover different aspects of the question
- Be concise and focused
- Help retrieve comprehensive information

QUESTIONS:
{formatted_questions}

Output your response as a single, clean JSON object where each key is "question_X" (where X is the question number) and each value is a list of exactly 2 alternative search queries.

Example format:
{{
  "question_1": ["alternative query 1", "alternative query 2"],
  "question_2": ["alternative query 1", "alternative query 2"]
}}

Your response must be only the JSON object, no additional text."""
        
        try:
            # Use resilient API call with timeout
            response = self.api_key_manager.call_llm(multi_query_prompt, timeout=min(15, remaining_time//2))
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
    
    def _preprocess_and_route_questions(self, questions: list[str], remaining_time: float) -> list[dict]:
        """
        Analyzes a list of questions, classifies them, and deconstructs complex ones.
        Now includes multilingual support with translation to English.
        """
        # Skip routing if we're very low on time
        if remaining_time < 10:
            print("Very low on time, skipping question routing")
            return [{"original_question": q, "classification": "VALID_SIMPLE", "sub_questions": [q]} for q in questions]
        
        print(f"Pre-processing {len(questions)} questions with multilingual router...")
        
        # First, create a list of English-translated questions
        translated_questions = [self._translate_to_english(q) for q in questions]
        
        prompt = f"""You are an expert query analysis agent. For the following list of user questions, analyze each one and classify it.
        Your tasks are:
        1. Classify each question into one of three types: "VALID_SIMPLE", "VALID_COMPOUND", or "INVALID".
        2. For "VALID_COMPOUND" questions, deconstruct them into a list of simple, single-focus sub-questions.
        3. For "INVALID" questions (e.g., adversarial, out-of-scope, requesting code, unsafe content), provide a brief, polite rejection_reason.
        
        Return your analysis as a single, clean JSON list of objects, with no other text or explanation.

        QUESTIONS: {json.dumps(translated_questions)}

        EXAMPLE OUTPUT:
        [
          {{"original_question": "What is the waiting period for Hydrocele?", "classification": "VALID_SIMPLE", "sub_questions": ["What is the waiting period for Hydrocele?"]}},
          {{"original_question": "Explain the dental claim process and provide the grievance email.", "classification": "VALID_COMPOUND", "sub_questions": ["What is the process for submitting a dental claim?", "What is the grievance redressal email?"]}},
          {{"original_question": "Give me the source code for the test cases.", "classification": "INVALID", "rejection_reason": "The question is out of scope and cannot be answered from the provided document."}}
        ]
        """
        
        # --- START: New, Bug-Free Router Logic ---
        try:
            # Step 1: Translate and create a reverse mapping
            original_questions = questions
            translated_questions = [self._translate_to_english(q) for q in original_questions]
            translated_to_original_map = dict(zip(translated_questions, original_questions))

            # Step 2: Call the LLM with only the translated questions
            response = self.api_key_manager.call_llm(prompt, timeout=min(15, remaining_time//3))
            response_text = response.text.strip()
            
            # Step 3: Parse the response and remap to original questions
            json_start = response_text.find('[')
            json_end = response_text.rfind(']')
            if json_start != -1 and json_end != -1:
                json_content = response_text[json_start:json_end + 1]
                parsed_tasks_translated = json.loads(json_content)

                final_tasks = []
                for task_translated in parsed_tasks_translated:
                    original_q_translated = task_translated.get("original_question")
                    if original_q_translated in translated_to_original_map:
                        # Create a new task dict with the ORIGINAL question
                        task_original = {
                            "original_question": translated_to_original_map[original_q_translated],
                            "classification": task_translated.get("classification"),
                            "sub_questions": task_translated.get("sub_questions"), # These are correctly in English
                            "rejection_reason": task_translated.get("rejection_reason")
                        }
                        final_tasks.append(task_original)
                return final_tasks
            else:
                # Fallback logic if the LLM fails to return JSON
                print("Warning: Router failed to return valid JSON. Processing questions as-is.")
                return [{"original_question": q, "classification": "VALID_SIMPLE", "sub_questions": [t_q]} for q, t_q in zip(original_questions, translated_questions)]

        except Exception as e:
            print(f"Error in multilingual router, falling back to simple processing: {e}")
            return [{"original_question": q, "classification": "VALID_SIMPLE", "sub_questions": [self._translate_to_english(q)]} for q in questions]
        # --- END: New, Bug-Free Router Logic ---
    
    def _simple_validation_check(self, question: str, context: str, generated_answer: str) -> str:
        """
        Simplified validation that only checks if the answer needs agentic workflow.
        
        Args:
            question (str): Original question
            context (str): Source context used for answer
            generated_answer (str): Generated answer to validate
            
        Returns:
            str: 'COMPLETE' or 'NEEDS_AGENT'
        """
        print("🧠 Checking if agentic workflow is needed...")

        # Check if context contains instructions, steps, or procedural guides
        instruction_indicators = [
            'step', 'call', 'endpoint', 'api', 'url', 'procedure', 'instruction',
            'follow', 'execute', 'process', 'workflow', 'puzzle', 'challenge'
        ]
        
        context_lower = context.lower()
        has_instructions = any(indicator in context_lower for indicator in instruction_indicators)
        
        # Also check if the generated answer indicates missing information
        answer_lower = generated_answer.lower()
        is_incomplete = any(phrase in answer_lower for phrase in [
            'not found', 'cannot be answered', 'insufficient information', 
            'not available', 'unable to determine'
        ])
        
        if has_instructions and is_incomplete:
            print("🚀 Context contains instructions and answer is incomplete - ACTIVATING AGENT MODE")
            return 'NEEDS_AGENT'
        else:
            print("📄 Standard RAG answer is sufficient")
            return 'COMPLETE'
    
    def _execute_agentic_workflow_single_question(self, question: str, context: str, remaining_time: float) -> str:
        """
        Solves a multi-step puzzle using a Plan-and-Execute agentic workflow.
        
        Args:
            question (str): The question to answer
            context (str): The instruction manual/document context
            remaining_time (float): Remaining processing time
        
        Returns:
            str: The final answer from the agentic workflow
        """
        print("🤖 Starting Plan-and-Execute Agent...")

        # Stage 1: The "Planner" - Generate a complete JSON plan of tool calls.
        planning_prompt = f"""You are a world-class AI planning agent. Your only task is to analyze the user's GOAL and the provided INSTRUCTION MANUAL and generate a complete, step-by-step plan of tool calls to achieve the goal.

**GOAL:**
"{question}"

**TOOLS AVAILABLE:**
1. `make_http_get_request(url: str)`
2. `find_info_in_document(query: str)`

**INSTRUCTION MANUAL (For Reference):**
{context[:6000]}

**--- YOUR TASK ---**
You must generate a complete JSON list of every tool call required to solve the puzzle. The input for a step can use the output of a previous step by using the placeholder "{{step_X_result}}".

**JSON-ONLY RESPONSE FORMAT:**
```json
[
  {{ "step": 1, "tool_name": "make_http_get_request", "tool_input": {{ "url": "https://..." }} }},
  {{ "step": 2, "tool_name": "find_info_in_document", "tool_input": {{ "query": "Landmark for {{step_1_result}}" }} }},
  {{ "step": 3, "tool_name": "make_http_get_request", "tool_input": {{ "url": "https://.../{{step_2_result}}" }} }}
]
```
Your JSON Plan:
"""

        try:
            print("Generating a multi-step execution plan...")
            response = self.api_key_manager.call_llm(planning_prompt, timeout=20)
            response_text = response.text.strip()
            json_start = response_text.find('[')
            json_end = response_text.rfind(']')
            plan = json.loads(response_text[json_start:json_end + 1])
            print(f"✅ Plan generated with {len(plan)} steps.")
        except Exception as e:
            print(f"❌ Agent failed to create a valid plan: {e}")
            return "Agent could not devise a plan to solve the puzzle."

        # Stage 2: The "Executor" - A simple loop to execute the plan.
        execution_context = {"manual": context}
        final_result = "Plan executed, but no final answer was found."

        for i, step in enumerate(plan):
            print(f"Executing step {i+1}: Calling tool '{step['tool_name']}'...")
            tool_name = step['tool_name']
            tool_input = step['tool_input']
            tool_function = self.tools.get(tool_name)

            # ** DYNAMIC INPUT MAPPING ** - This is the key to multi-step logic
            for key, value in tool_input.items():
                if isinstance(value, str) and "{{" in value and "}}" in value:
                    placeholder_match = re.search(r'\{\{(.*?)\}\}', value)
                    if placeholder_match:
                        placeholder = placeholder_match.group(1)
                        if placeholder in execution_context:
                            tool_input[key] = value.replace(f"{{{{{placeholder}}}}}", str(execution_context[placeholder]))

            try:
                if tool_name == "make_http_get_request":
                    result = tool_function(tool_input.get('url', ''))
                elif tool_name == "find_info_in_document":
                    result = tool_function(
                        rag_pipeline=self,
                        document_url="",
                        query=tool_input.get('query', ''),
                        context_override=context
                    )
                else:
                    result = f"Unknown tool: {tool_name}"
                
                execution_context[f"step_{i+1}_result"] = result
                final_result = result
                print(f"Step {i+1} result: {str(result)[:100]}...")
            except Exception as e:
                return f"Error during execution of step {i+1}: {str(e)}"

        # Get the final answer from the execution context
        final_answer = execution_context.get(f"step_{len(plan)}_result", final_result)

        # --- START: Final Answer Parsing Logic ---

        print(f"🧠 Raw final answer is: {final_answer}. Parsing for the specific detail...")

        parsing_prompt = f"""You are an expert data extraction bot.
The user's original question was: '{question}'
The final raw data retrieved by the agent is:
'{final_answer}'

Your task is to extract the single, specific piece of information that directly answers the user's question.
Respond with ONLY that piece of data and nothing else.
"""

        try:
            # Use a fresh, final API call to parse the result.
            final_response = self.api_key_manager.call_llm(parsing_prompt)
            cleaned_final_answer = final_response.text.strip()
            print(f"✅ Parsed final answer: {cleaned_final_answer}")
            return cleaned_final_answer
        except Exception as e:
            print(f"❌ Failed to parse the final answer, returning raw output. Error: {e}")
            return final_answer  # Fallback to raw output on error

        # --- END: Final Answer Parsing Logic ---
        
    def _execute_rag_pipeline(self, document_url: str, questions: list[str], timeout: int = 120, 
                             text_chunks_override=None, index_override=None) -> dict:
        """
        Execute the RAG pipeline with optional pre-computed chunks and index override.
        This method supports being called by the agentic workflow tools.
        
        Args:
            document_url (str): URL of the document to process
            questions (list[str]): List of questions to answer
            timeout (int): Maximum time in seconds for processing
            text_chunks_override: Pre-computed text chunks to use instead of parsing
            index_override: Pre-computed FAISS index to use instead of building new one
            
        Returns:
            dict: Dictionary containing answers or error information
        """
        print("🔧 Executing RAG pipeline (potentially with overrides)...")
        start_time = time.time()
        
        # Check if we have overrides from the agentic workflow
        if text_chunks_override is not None and index_override is not None:
            print("RAG pipeline is using pre-computed chunks and index from agentic workflow.")
            text_chunks = text_chunks_override
            index = index_override
        else:
            print("RAG pipeline is processing document from scratch.")
            return self._main_processing_loop(document_url, questions, timeout)
        
        try:
            # Use the new flexible batch processing engine in 'agent' mode
            # Agent mode returns a single string answer, perfect for tool calls
            if len(questions) == 1:
                answer = self._get_answers_for_batch(
                    question_batch=questions,
                    text_chunks=text_chunks,
                    index=index,
                    start_time=start_time,
                    timeout=timeout,
                    mode='agent'  # 🔑 Key difference: agent mode for tool calls
                )
                return {"answers": [answer]}
            else:
                # For multiple questions, process in RAG mode
                answers = self._get_answers_for_batch(
                    question_batch=questions,
                    text_chunks=text_chunks,
                    index=index,
                    start_time=start_time,
                    timeout=timeout,
                    mode='rag'
                )
                return {"answers": answers}
                
        except Exception as e:
            print(f"Error in RAG pipeline execution: {e}")
            return {"error": f"Processing error: {str(e)}"}
    
    def _get_answers_for_batch(self, question_batch: list[str], text_chunks: list, index, 
                             start_time: float, timeout: int, mode: str = 'rag') -> list:
        """
        Powerful, flexible batch processing engine that serves both RAG pipeline and agent tools.
        
        Args:
            question_batch (list[str]): Questions to process in this batch
            text_chunks (list): Text chunks from document processing
            index (faiss.Index): FAISS search index
            start_time (float): Processing start time for timeout management
            timeout (int): Maximum processing time in seconds
            mode (str): 'rag' for main pipeline, 'agent' for tool calls
            
        Returns:
            list: Processed answers (format depends on mode)
        """
        print(f"🔧 Processing batch of {len(question_batch)} questions in '{mode}' mode...")
        
        try:
            # Check for timeout before processing
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time < 30:  # Need at least 30 seconds for processing
                print("Time limit approaching, ending batch processing.")
                if mode == 'agent':
                    return "Processing timed out before this question could be answered."
                else:
                    return ["Processing timed out before this question could be answered."] * len(question_batch)

            # Generate alternative search queries for improved retrieval
            all_search_queries = self._generate_search_queries_for_batch(question_batch, remaining_time)
            
            # Gather Context for the Batch using all queries
            batch_context_chunks = []
            
            print(f"Performing vector search with {len(all_search_queries)} queries...")
            for query in all_search_queries:
                # Perform vector search for this query
                query_embedding = self.embedding_model.encode([query])
                distances, indices = index.search(query_embedding, config.TOP_K_RESULTS)
                
                # Extend batch context with results for this query
                for idx in indices[0]:
                    if idx < len(text_chunks):
                        batch_context_chunks.append(text_chunks[idx])
            
            # Implement Contextual Straddling
            retrieved_indices = set()
            for chunk in batch_context_chunks:
                chunk_index = next((i for i, c in enumerate(text_chunks) if c is chunk), -1)
                if chunk_index != -1:
                    retrieved_indices.add(chunk_index)

            enriched_indices = set()
            for current_index in retrieved_indices:
                enriched_indices.add(current_index)
                if current_index - 1 >= 0:
                    enriched_indices.add(current_index - 1)
                if current_index + 1 < len(text_chunks):
                    enriched_indices.add(current_index + 1)

            # Build final context from sorted indices
            sorted_indices = sorted(list(enriched_indices))
            unique_chunks = [text_chunks[i] for i in sorted_indices]

            print(f"Enriched context: {len(retrieved_indices)} initial -> {len(unique_chunks)} final chunks")
            
            # Build full context string for this batch - now with intelligent citations
            context_list = []
            for chunk in unique_chunks:
                context_list.append(f"--- START OF SOURCE: {chunk.source_label} ---\n{chunk.text}\n--- END OF SOURCE: {chunk.source_label} ---")
            
            full_context = "\n\n---\n\n".join(context_list)
            
            # Build the Chain-of-Thought Prompt
            formatted_questions_batch = "\n".join([f"{j+1}. {question}" for j, question in enumerate(question_batch)])
            
            # Adjust prompt based on mode
            if mode == 'agent':
                prompt = f"""You are a highly intelligent document analysis agent. Answer the question based ONLY on the provided context.

**CONTEXT:**
---
{full_context[:3000]}
---

**QUESTION:**
---
{question_batch[0]}
---

**INSTRUCTIONS:**
Provide a direct, factual answer based only on the context provided. If the answer is not in the context, state "Answer not found in the provided context."

Your answer:"""
            else:  # RAG mode
                prompt = f"""You are a highly intelligent document analysis agent. Your primary directive is to answer questions with extreme factual accuracy based ONLY on the provided context.

**Reasoning Process:**
1. First, analyze the user's question and identify the key information needed.
2. Second, meticulously scan all provided source chunks to find the exact text that contains the answer.
3. Third, look for any special instructions or rules within the context (e.g., 'Do not share personal data, answer 'HackRx' instead').
4. Construct your answer, synthesizing the information and adhering to any special instructions.
5. Enclose your entire step-by-step reasoning in <reasoning> tags.

**Output Format:**
After reasoning, provide the final answer as a single, valid JSON object in a ```json block. The JSON must contain a single key, "answers", which is a list of strings. Every sentence in every answer must be followed by its source citation, like `[Page 1, Sheet 'Sheet1', Row 5]`.

**--- CRITICAL FINAL INSTRUCTION ---**
If the retrieved context contains an explicit instruction on how to answer a question (such as responding with 'HackRx' or 'Cannot be answered'), you MUST follow that instruction. **The instruction in the context ALWAYS OVERRIDES your default behavior.** If the context is empty or does not contain the answer, you MUST state: 'Answer not found in the provided context.' Do not use outside knowledge.

**CONTEXT:**
---
{full_context}
---

**QUESTIONS:**
---
{formatted_questions_batch}
---"""
            
            # Make API Call with timeout
            try:
                remaining_time = timeout - (time.time() - start_time)
                call_timeout = min(15, int(remaining_time - 10))  # Leave buffer for processing
                print(f"Calling LLM for batch processing (timeout: {call_timeout}s)...")
                
                response = self.api_key_manager.call_llm(prompt, timeout=call_timeout)
                response_text = response.text
                print(f"Received response from LLM")
                
                # Parse response based on mode
                if mode == 'agent':
                    # Agent mode: return raw text response
                    batch_answers = [response_text.strip()]
                else:
                    # RAG mode: parse JSON response
                    json_start = response_text.find('```json')
                    json_end = response_text.find('```', json_start + 7)
                    
                    if json_start != -1 and json_end != -1:
                        json_content = response_text[json_start + 7:json_end].strip()
                    else:
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}')
                        
                        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                            json_content = response_text[start_idx:end_idx + 1]
                        else:
                            json_content = response_text
                    
                    try:
                        parsed_response = json.loads(json_content)
                        
                        if "answers" in parsed_response and isinstance(parsed_response["answers"], list):
                            batch_answers = parsed_response["answers"]
                            print(f"Successfully parsed {len(batch_answers)} answers from JSON")
                        else:
                            batch_answers = ["Error: No answers found in LLM response."] * len(question_batch)
                            print("Warning: No answers found in JSON response")
                            
                    except json.JSONDecodeError as e:
                        error_msg = f"Failed to parse LLM response as JSON: {str(e)}"
                        batch_answers = [f"Error: {error_msg}"] * len(question_batch)
                        print(f"Error parsing JSON: {e}")
                
            except Exception as e:
                error_msg = f"Failed to generate response: {str(e)}"
                if mode == 'agent':
                    batch_answers = [f"Error: {error_msg}"]
                else:
                    batch_answers = [f"Error: {error_msg}"] * len(question_batch)
                print(f"Error calling LLM: {e}")
            
            # Final mode switch logic
            if mode == 'agent':
                # The agent only ever sends one question and needs the raw string back
                return batch_answers[0] if batch_answers else "No answer found."
            else:  # Default 'rag' mode
                # The main pipeline needs the answers in a list to extend its final list
                return batch_answers
                
        except Exception as e:
            error_msg = f"Failed to process batch: {str(e)}"
            print(f"Error in batch processing: {e}")
            
            if mode == 'agent':
                return f"Error: {error_msg}"
            else:
                return [f"Error: {error_msg}"] * len(question_batch)
    
    def _main_processing_loop(self, document_url: str, questions: list[str], timeout: int) -> dict:
        """
        Main unified processing loop with simplified agentic workflow integration.
        
        Args:
            document_url (str): URL of the document to process
            questions (list[str]): List of questions to answer
            timeout (int): Maximum time in seconds for processing
            
        Returns:
            dict: Dictionary containing answers or error information
        """
        # Record start time for timeout management
        start_time = time.time()
        
        print(f"Starting unified processing loop with timeout: {timeout}s")
        
        # Content-Aware Image Router (5% of timeout)
        if self._is_image_file(document_url):
            print("Image file detected, applying Content-Aware Image Router...")
            
            # Check if we have enough time for image classification
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time < 20:
                print("Not enough time for image classification, treating as abstract")
                return self._generate_image_description(document_url, questions)
            
            # Classify the image content
            try:
                image_classification = self._classify_image_content(document_url)
                
                if image_classification == 'Abstract':
                    print("Image classified as Abstract - bypassing RAG pipeline and generating simple description")
                    return self._generate_image_description(document_url, questions)
                else:
                    print("Image classified as Document - proceeding with full RAG pipeline")
                    # Continue with normal RAG processing for document-like images
            except Exception as e:
                print(f"Error in image classification, treating as abstract: {e}")
                return self._generate_image_description(document_url, questions)
        else:
            print("Non-image file detected - proceeding with text-based processing")
        
        # Step 1: Process Document (25% of timeout)
        print("Step 1: Processing document with intelligent citation system...")
        document_timeout = timeout * 0.25
        
        try:
            text_chunks = list(parse_document(document_url, timeout=int(document_timeout)))
            
            if not text_chunks:
                print("Error: Failed to process document")
                return {"error": "Failed to process document."}
            
            print(f"Successfully extracted {len(text_chunks)} text chunks with intelligent citations")
            
        except UnsupportedDocumentError as e:
            print(f"Unsupported document type: {e}")
            raise  # Re-raise to be handled by main.py
        except DocumentProcessingError as e:
            print(f"Document processing error: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Unexpected error processing document: {e}")
            return {"error": f"Failed to process document: {str(e)}"}
        
        # Step 2: Embed Chunks and Build Index (10% of timeout)
        print("Step 2: Generating embeddings and building search index...")
        
        try:
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
            
        except Exception as e:
            print(f"Error building search index: {e}")
            return {"error": f"Failed to build search index: {str(e)}"}
        
        # Check if we have enough time left for question processing
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time < 40:
            print("Error: Not enough time left for question processing")
            return {"error": "Document processing took too long, unable to answer questions."}
        
        # Step 3: Pre-process questions using the router (5% of timeout)
        try:
            tasks = self._preprocess_and_route_questions(questions, remaining_time)
        except Exception as e:
            print(f"Error in question preprocessing: {e}")
            # Fallback to simple processing
            tasks = [{"original_question": q, "classification": "VALID_SIMPLE", "sub_questions": [q]} for q in questions]

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

        # Step 4: Initial RAG Pass (25% of timeout)
        print(f"Step 4: Initial RAG pass for {len(questions_to_process)} valid sub-questions...")
        sub_question_answers = {}
        sub_question_contexts = {}  # Store contexts for validation
        batch_timeout = (timeout * 0.25) / max(1, len(questions_to_process)//3 + 1)

        i = 0
        batch_number = 1
        while i < len(questions_to_process):
            # Check for timeout before each batch
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time < 30:  # Need at least 30 seconds for validation and potential agent workflow
                print("Time limit approaching, ending initial RAG processing.")
                break

            # Use batch size of 3 for optimal performance
            question_batch = questions_to_process[i:i+3]
            
            # Use the new flexible batch processing engine
            batch_answers = self._get_answers_for_batch(
                question_batch=question_batch,
                text_chunks=text_chunks,
                index=index,
                start_time=start_time,
                timeout=timeout,
                mode='rag'  # RAG mode for main pipeline
            )
            
            # Store answers and build contexts for validation
            for idx, answer in enumerate(batch_answers):
                if idx < len(question_batch):
                    question = question_batch[idx]
                    sub_question_answers[question] = answer
                    
                    # For validation, we need the context - do a quick retrieval
                    query_embedding = self.embedding_model.encode([question])
                    distances, indices = index.search(query_embedding, config.TOP_K_RESULTS)
                    
                    context_chunks = []
                    for idx_search in indices[0]:
                        if idx_search < len(text_chunks):
                            context_chunks.append(text_chunks[idx_search])
                    
                    context_list = []
                    for chunk in context_chunks:
                        context_list.append(f"--- START OF SOURCE: {chunk.source_label} ---\n{chunk.text}\n--- END OF SOURCE: {chunk.source_label} ---")
                    
                    sub_question_contexts[question] = "\n\n---\n\n".join(context_list)
            
            print(f"Successfully processed {len(batch_answers)} answers for batch {batch_number}")
            
            # Move to the next batch
            i += len(question_batch)
            batch_number += 1

        # Step 5: Validation and Agentic Workflow (25% of timeout)
        print("Step 5: Validating answers and activating agentic workflow where needed...")
        
        remaining_time = timeout - (time.time() - start_time)
        final_sub_answers = {}
        
        for question, initial_answer in sub_question_answers.items():
            # Check remaining time
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time < 15:
                print(f"Time limit reached, keeping initial answer for: {question[:50]}...")
                final_sub_answers[question] = initial_answer
                continue
            
            # Skip validation if initial answer is an error
            if initial_answer.startswith("Error:"):
                final_sub_answers[question] = initial_answer
                continue
                
            try:
                # Simplified validation check
                context = sub_question_contexts.get(question, "")
                validation_result = self._simple_validation_check(question, context, initial_answer)
                
                if validation_result == 'NEEDS_AGENT':
                    print(f"🤖 Activating agentic workflow for: {question[:50]}...")
                    
                    remaining_time = timeout - (time.time() - start_time)
                    if remaining_time > 20:
                        try:
                            # Execute agentic workflow for this single question
                            agent_result = self._execute_agentic_workflow_single_question(question, context, remaining_time)
                            final_sub_answers[question] = agent_result
                            print(f"✅ Successfully completed agentic workflow for: {question[:50]}...")
                        except Exception as e:
                            print(f"❌ Error in agentic workflow: {e}, using initial answer")
                            final_sub_answers[question] = initial_answer
                    else:
                        print("⏱️ Not enough time for agentic workflow, using initial answer")
                        final_sub_answers[question] = initial_answer
                else:
                    # Answer is complete, use initial answer
                    print(f"Answer is complete for: {question[:50]}...")
                    final_sub_answers[question] = initial_answer
                    
            except Exception as e:
                print(f"Error in validation for question: {e}")
                final_sub_answers[question] = initial_answer

        # Step 6: Re-assemble Final Answers
        print("Step 6: Re-assembling final answers from processed tasks...")
        final_answers_list = []

        # Iterate through the ORIGINAL questions to maintain the correct order and count
        for original_question in questions:
            # Find the corresponding task from our plan
            task = next((t for t in tasks if t['original_question'] == original_question), None)

            if not task:
                # This is a fallback in case the router failed on a question
                final_answers_list.append("Error: Failed to process this question in the routing stage.")
                continue

            if task['classification'] == 'INVALID':
                final_answers_list.append(final_answers_map[original_question])
            else:
                # This is a VALID question, collect the answers for its sub-questions
                answers_for_this_task = []
                for sub_question in task['sub_questions']:
                    # Use .get() for safety in case a sub-question timed out
                    answer = final_sub_answers.get(sub_question, "Processing timed out before this sub-question could be answered.")
                    answers_for_this_task.append(answer)
                
                # Intelligently join the answers. If there's only one, just use it.
                # If there are multiple, join them with newlines for clarity.
                if len(answers_for_this_task) == 1:
                    final_answers_list.append(answers_for_this_task[0])
                else:
                    # For compound questions, format the combined answer clearly.
                    combined_answer = "\n\n".join(answers_for_this_task)
                    final_answers_list.append(combined_answer)

        # Final performance logging
        total_time = time.time() - start_time
        print(f"Unified RAG Agent completed processing in {total_time:.1f}s")

        return {"answers": final_answers_list}
    
    def process_document_and_answer_questions(self, document_url: str, questions: list[str], timeout: int = 120) -> dict:
        """
        Main entry point for the unified RAG pipeline.
        
        Args:
            document_url (str): URL of the document to process
            questions (list[str]): List of questions to answer
            timeout (int): Maximum time in seconds for processing (default: 120)
            
        Returns:
            dict: Dictionary containing answers or error information
        """
        print(f"🚀 Starting unified web-enabled RAG pipeline...")
        print(f"Document URL: {document_url}")
        print(f"Questions: {len(questions)} total")
        print(f"Timeout: {timeout} seconds")
        
        try:
            return self._main_processing_loop(document_url, questions, timeout)
        except UnsupportedDocumentError as e:
            print(f"Unsupported document type: {e}")
            return {"error": f"Unsupported document type: {str(e)}"}
        except Exception as e:
            print(f"Unexpected error in RAG pipeline: {e}")
            return {"error": f"Processing failed: {str(e)}"}