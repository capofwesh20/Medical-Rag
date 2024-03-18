# Medical-Rag
![image](https://github.com/capofwesh20/Medical-Rag/assets/35642413/1a56974f-ba34-4748-9be4-7288bf2d7865)



## Overview

This project leverages advanced NLP techniques and models to extract, index, and retrieve medical information from unstructured text documents. It uses the `langchain` and `qdrant` libraries for processing and storing embeddings for efficient similarity searches. The system includes a FastAPI backend to serve retrieval-based answers, which are informed by document embeddings and reordering for context relevance.

## Features

- **Document Ingestion**: Process and ingest documents from directories, handling PDFs and other unstructured text formats.
- **Embeddings Generation**: Generate sentence embeddings using `SentenceTransformerEmbeddings` with a pre-trained model (`NeuML/pubmedbert-base-embeddings`).
- **Vector Database**: Store and index document embeddings in a `Qdrant` vector database for efficient similarity searches.
- **Retrieval-Based QA**: Implement a retrieval-based question answering system with reordering for context relevance.
- **API Server**: A FastAPI server that provides a web interface for query processing and displaying responses.

## Installation

Before running the project, ensure you have Python 3.8 or later installed. Clone the repository, then install the dependencies:

```bash
git clone https://github.com/capofwesh20/Medical-Rag.git
cd Medical-Rag
pip install -r requirements.txt
```

## Usage

To start the project, you need to run the FastAPI server:

```bash
uvicorn rag:app --reload
```

This command starts the API server, making it accessible at `http://localhost:8000`. You can interact with the server using the provided web interface or directly through API endpoints.

## Files and Directories

- `ingest.py`: Script for document ingestion and embeddings generation.
- `retriever.py`: Module for similarity search and document retrieval.
- `rag.py`: FastAPI application server script for handling queries and returning responses.
- `requirements.txt`: Contains all necessary Python packages.
- `data/`: Directory to place your PDFs and other documents for ingestion.
- `template/`: Contains Jinja2 templates for the web interface.
- `static/`: Static files directory for the FastAPI app.

## How It Works

1. **Document Ingestion**: The system ingests documents from the `data/` directory, generating embeddings for each document segment.
2. **Vector Database**: Embeddings are indexed in a Qdrant vector database, facilitating efficient similarity searches.
3. **Query Processing**: Users submit queries through the web interface. The system retrieves relevant documents and uses a retrieval-based QA model to generate responses.
4. **Response Generation**: The system reorders the context for relevance and presents the information to the user.

## Contributing

Contributions to the Medical Rag project are welcome. Please follow the standard fork and pull request workflow. Ensure you write clean code and document any new functions or modules you add.

## Contact

For any questions or contributions, please contact [Ikenna Odezuligbo] at `<Odezuligboe@gmail.com>`.


