# RAG System Implementation

A Python implementation of a Retrieval-Augmented Generation (RAG) system that answers questions based on document content.

## Features
- PDF document processing
- Text chunking and embedding
- Semantic search using FAISS
- Answer generation with OpenAI GPT

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your OpenAI API key in `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Place your PDF document in the project directory
2. Run the system:
```python
python rag_system.py
```
3. Enter questions at the prompt

## Dependencies
- openai
- pymupdf
- faiss-cpu
- numpy
- langchain
- python-dotenv

## Documentation
For more detailed explanations, see [rag_learning_doc.md](rag_learning_doc.md)

## License
MIT
