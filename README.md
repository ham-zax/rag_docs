# Intelligent Document Oracle - RAG System Implementation

## Overview
A Python implementation of a Retrieval-Augmented Generation (RAG) system that provides intelligent document querying capabilities.

## Key Features

| Feature | Description |
|---------|-------------|
| Document Processing | Extract and process text from PDF documents |
| Text Chunking | Split documents into manageable chunks with configurable size and overlap |
| Semantic Search | Find relevant content using hybrid search (semantic + keyword) |
| Answer Generation | Generate natural language answers using OpenAI GPT models |
| Conversation History | Maintain context across multiple queries |

## System Architecture

```
[Document (PDF)] --> [Text Extraction] --> [Text Chunking] --> [Embedding Generation]
    ^                                                                 |
    |                                                                 v
[User Query] <-- [Answer Generation] <-- [Semantic Search] <-- [Vector Database]
```

## Installation

### Requirements
- Python 3.8+
- OpenAI API key

### Steps
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure environment variables in `.env`:
```env
OPENAI_API_KEY=your_api_key_here
```

## Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| chunk_size | 500 | Maximum size of text chunks |
| chunk_overlap | 100 | Overlap between consecutive chunks |
| embedding_model | text-embedding-3-small | OpenAI embedding model |
| completion_model | gpt-4o-mini | OpenAI completion model |
| max_context_chunks | 3 | Maximum chunks used for context |

## Usage

1. Place PDF documents in the project directory
2. Run the system:
```bash
python rag_system.py
```
3. Enter queries at the prompt

Example:
```
> What is the main topic of this document?
The document primarily discusses...
```

## Dependencies

| Package | Purpose |
|---------|---------|
| openai | OpenAI API interaction |
| pymupdf | PDF text extraction |
| qdrant-client | Vector database operations |
| langchain | Text processing and chunking |
| python-dotenv | Environment variable management |

## Documentation
For detailed technical explanations, see [rag_learning_doc.md](rag_learning_doc.md)

## License
MIT License
# rag_docs
