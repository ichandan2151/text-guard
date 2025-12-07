# TextGuard Processor

TextGuard Processor is a FastAPI-based microservice designed to analyze text documents for potential risks. It downloads PDF files from a specified storage, processes them by breaking them into manageable chunks, and uses a pre-trained transformer model to classify each chunk as either "risk" or "no_risk". The results are then stored in a database for further analysis.

This service is ideal for integrating automated document compliance and risk assessment into your workflows.

## Features

*   **PDF Document Processing**: Extracts text directly from PDF files.
*   **Smart Text Chunking**: Splits documents into sections and then into smaller, overlapping text chunks suitable for model processing.
*   **ML-Powered Risk Classification**: Uses a Hugging Face `transformers` model for sequence classification on each text chunk.
*   **Supabase Integration**: Seamlessly connects with Supabase for both file storage (Storage) and data persistence (PostgreSQL database).
*   **Asynchronous Processing**: Document processing is handled as a background task to ensure the API remains responsive.
*   **RESTful API**: Provides simple and clear endpoints for classifying text and processing entire documents.

## Prerequisites

*   Python 3.8+
*   A pre-trained Hugging Face transformer model for sequence classification.
*   A Supabase project for database and file storage.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd textguard-processor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`.

    ```text
    fastapi
    uvicorn[standard]
    pydantic
    torch
    transformers
    httpx
    requests
    PyPDF2
    python-dotenv
    ```

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Your Model:**
    The service loads a model from a local directory. Place your pre-trained model files (`config.json`, `pytorch_model.bin`, `tokenizer_config.json`, etc.) into a directory. By default, the application looks for a folder named `model`.

5.  **Configure Environment Variables:**
    Create a `.env` file in the root of your project and add the following variables. The application will load these automatically.

    ```env
    # --- Service Configuration ---
    MODEL_DIR=model
    PORT=8080

    # --- Supabase Configuration ---
    SUPABASE_URL="https://your-project-ref.supabase.co"
    SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key"
    SUPABASE_BUCKET="documents"
    ```

    *   `MODEL_DIR`: The local directory where your classification model is stored.
    *   `SUPABASE_URL`: The URL of your Supabase project.
    *   `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key, which provides admin-level access.
    *   `SUPABASE_BUCKET`: The name of the public Supabase Storage bucket where your PDFs are stored.

## Running the Service

To start the FastAPI application, run the following command in your terminal:

```bash
uvicorn main:app --host "0.0.0.0" --port 8080 --reload
```

The `--reload` flag is useful for development, as it automatically restarts the server when you make changes to the code.

## API Endpoints

### Health Checks

*   **`GET /`**
    *   **Description**: Returns the status of the service.
    *   **Response**: `{"status": "ok"}`

*   **`GET /healthz`**
    *   **Description**: A standard health check endpoint.
    *   **Response**: `{"status": "healthy"}`

### Text Classification

*   **`POST /classify`**
    *   **Description**: Classifies a single string of text.
    *   **Request Body**:
        ```json
        {
          "text": "This is a sample text to be classified for risk."
        }
        ```
    *   **Response**:
        ```json
        {
          "label": "no_risk",
          "score": 0.98,
          "probs": [0.98, 0.02]
        }
        ```

### Document Processing

*   **`POST /process_document`**
    *   **Description**: Initiates the background processing of a PDF document stored in Supabase Storage. The API returns a `202 Accepted` response immediately.
    *   **Request Body**:
        ```json
        {
          "document_id": "doc-uuid-12345",
          "storage_path": "path/to/your/document.pdf",
          "file_name": "document.pdf",
          "project_name": "Project Alpha"
        }
        ```
    *   **Response**:
        ```json
        {
            "message": "Document processing started in the background."
        }
        ```
    *   **Background Workflow**:
        1.  Downloads the PDF from `SUPABASE_URL/storage/v1/object/public/SUPABASE_BUCKET/storage_path`.
        2.  Extracts text and splits it into chunks.
        3.  Classifies each chunk using the model.
        4.  Inserts the classified chunks into the `chunks` table in your Supabase database.
        5.  Updates the corresponding row in the `documents` table with a summary of the analysis (risk score, chunk counts, etc.).

