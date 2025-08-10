# Multimodal RAG API Project

This project is a powerful, web-enabled Retrieval-Augmented Generation (RAG) pipeline capable of processing various document types, including PDFs, Office documents, and images with text. The entire application is containerized using Docker, allowing for a simple and consistent setup on any machine.

### Key Features

-   **Multi-Format Document Processing**: Handles `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.eml`, and common image formats (`.png`, `.jpg`, etc.).
-   **Integrated OCR**: Uses Google's Tesseract to automatically extract text from scanned documents and images.
-   **Intelligent RAG Pipeline**: Employs advanced techniques like query expansion, an agentic workflow, and a multi-key management system to provide accurate and resilient answers.
-   **Secure and Scalable**: Built on a FastAPI backend featuring token-based authentication and asynchronous processing.

---

## Why Docker? The Tesseract OCR Dependency

A critical component of this project is its ability to perform Optical Character Recognition (OCR). This functionality is provided by **Google's Tesseract OCR engine**, which allows the application to "read" text from images and scanned PDFs.

Tesseract is a system-level dependency, meaning it cannot be installed simply with `pip install`. Running this project locally without Docker would require you to manually install Tesseract on your operating system (Windows, macOS, or Linux) and ensure it's correctly configured in your system's PATH. This can be a complex and error-prone process.

**This is where Docker simplifies everything.** The `Dockerfile` for this project includes commands to automatically install the Tesseract OCR engine directly into the container's isolated Linux environment. When you run the `docker build` command, Docker handles this entire setup process for you.

Therefore, **you do not need to install Tesseract on your own computer.** The Docker container is a self-contained package with all necessary software and dependencies included, guaranteeing it will run the same way everywhere.

---

## How to Run with Docker

This is the recommended method. It guarantees that all system dependencies, like Tesseract, are handled correctly.

### Prerequisites

-   [Docker](https://www.docker.com/get-started) must be installed and running on your system.

### Steps

**1. Build the Docker Image**

Open a terminal in the project's root directory and run the following command. This will download the base Python image, install Tesseract, set up all Python packages from `requirements.txt`, and package the application.


docker build -t multimodal-rag-api .
2. Run the Docker Container
After the build is complete, run the container. You must provide your Gemini API keys as an environment variable using the -e flag. This is the secure way to handle secrets.
code
Bash
docker run -p 8000:8000 -e GEMINI_API_KEYS="YOUR_API_KEY_1,YOUR_API_KEY_2" --name rag-api-container multimodal-rag-api
3. Check API Health
The API will now be running and accessible at http://localhost:8000. You can visit this URL in your browser or use curl to see the health check message:
code
Bash
curl http://localhost:8000/
# Expected output: {"message":"Multimodal RAG API is running!"}