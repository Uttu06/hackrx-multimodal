Multimodal RAG API Project
This project is a powerful, web-enabled Retrieval-Augmented Generation (RAG) pipeline capable of processing various document types, including PDFs, Office documents, and images with text. The entire application is containerized using Docker, allowing for a simple and consistent setup on any machine.
Key Features
Multi-Format Document Processing: Handles .pdf, .docx, .xlsx, .pptx, .eml, and common image formats (.png, .jpg, etc.).
Integrated OCR: Uses Google's Tesseract to automatically extract text from scanned documents and images.
Intelligent RAG Pipeline: Employs advanced techniques like query expansion, an agentic workflow, and a multi-key management system to provide accurate and resilient answers.
Secure and Scalable: Built on a FastAPI backend featuring token-based authentication and asynchronous processing.
Why Docker? The Tesseract OCR Dependency
A critical component of this project is its ability to perform Optical Character Recognition (OCR). This functionality is provided by Google's Tesseract OCR engine, which allows the application to "read" text from images and scanned PDFs.
Tesseract is a system-level dependency, meaning it cannot be installed simply with pip install. Running this project locally without Docker would require you to manually install Tesseract on your operating system (Windows, macOS, or Linux) and ensure it's correctly configured in your system's PATH. This can be a complex and error-prone process.
This is where Docker simplifies everything. The Dockerfile for this project includes commands to automatically install the Tesseract OCR engine directly into the container's isolated Linux environment. When you run the docker build command, Docker handles this entire setup process for you.
Therefore, you do not need to install Tesseract on your own computer. The Docker container is a self-contained package with all necessary software and dependencies included, guaranteeing it will run the same way everywhere.
How to Run with Docker
This is the recommended method. It guarantees that all system dependencies, like Tesseract, are handled correctly.
Prerequisites
Docker must be installed and running on your system.
Steps
1. Build the Docker Image
Open a terminal in the project's root directory and run the following command. This will download the base Python image, install Tesseract, set up all Python packages from requirements.txt, and package the application.
code
Bash
docker build -t multimodal-rag-api .
2. Run the Docker Container
After the build is complete, run the container. You must provide your Gemini API keys as an environment variable using the -e flag. This is the secure way to handle secrets.
code
Bash
docker run -p 8000:8000 -e GEMINI_API_KEYS="YOUR_API_KEY_1,YOUR_API_KEY_2" --name rag-api-container multimodal-rag-api
Command Breakdown:
-p 8000:8000: Maps your local port 8000 to the container's port 8000, making the API accessible.
-e GEMINI_API_KEYS="...": Securely passes your secret API keys to the application running inside the container. Replace the placeholder with your actual comma-separated keys.
--name rag-api-container: Gives your running container a memorable name for easy management (e.g., docker stop rag-api-container).
multimodal-rag-api: The name of the image you built in the previous step.
3. Check API Health
The API will now be running and accessible at http://localhost:8000. You can visit this URL in your browser or use curl to see the health check message:
code
Bash
curl http://localhost:8000/
# Expected output: {"message":"Multimodal RAG API is running!"}
API Usage
The primary functionality is exposed via the /hackrx/run endpoint.
Endpoint: /hackrx/run
Method: POST
Authentication: Bearer Token
Token: 99d35dc664dee13c02ed1b349bf35ab2f820f5adb1b9a158bdf5aa92fab5efec
Example Request (curl)
Here is an example of how to send a request to the running API using curl. This example asks two questions about a public PDF document.
code
Bash
curl -X 'POST' \
  'http://localhost:8000/hackrx/run' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer 99d35dc664dee13c02ed1b349bf35ab2f820f5adb1b9a158bdf5aa92fab5efec' \
  -H 'Content-Type: application/json' \
  -d '{
  "documents": "https://www.africau.edu/images/default/sample.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What does the last paragraph say?"
  ]
}'
You should receive a JSON response containing a list of answers corresponding to your questions.