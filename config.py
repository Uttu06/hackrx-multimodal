import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Keep for backward compatibility

# Load comma-separated API keys and convert to list
GEMINI_API_KEYS_STRING = os.getenv('GEMINI_API_KEYS', '')
GEMINI_API_KEY_LIST = [key.strip() for key in GEMINI_API_KEYS_STRING.split(',') if key.strip()]

# If no list is provided, fall back to single key for backward compatibility
if not GEMINI_API_KEY_LIST and GEMINI_API_KEY:
    GEMINI_API_KEY_LIST = [GEMINI_API_KEY]

# Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Search Configuration
TOP_K_RESULTS = 5