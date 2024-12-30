# Simplified Guide to Building a RAG System

This guide explains how to build a Retrieval-Augmented Generation (RAG) system in Python. A RAG system helps answer questions based on document content.

## How It Works
1. **Load** - Read text from a PDF
2. **Chunk** - Break text into smaller pieces
3. **Embed** - Convert text into number vectors
4. **Index** - Store vectors for quick searching
5. **Query** - Convert questions into vectors
6. **Retrieve** - Find most relevant text chunks
7. **Generate** - Create answers using AI

## Key Components

### 1. Importing Tools
```python
import openai  # For AI capabilities
import pymupdf  # To read PDFs
import faiss  # For fast searching
import numpy as np  # For number handling
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split text
```

### 2. Setting Up
```python
# Load secret API key
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
```

### 3. Main System Class
```python
class RAGSystem:
    def __init__(self):
        # Set up text splitter and search index
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        self.index = faiss.IndexFlatL2(1536)
        self.chunks = []
```

### 4. Core Functions

#### Load Document
```python
def load_document(self, file_path):
    # Open PDF and extract text
    doc = pymupdf.open(file_path)
    return "".join(page.get_text() for page in doc)
```

#### Process Text
```python
def process_text(self, text):
    # Split text and create embeddings
    self.chunks = self.text_splitter.split_text(text)
    embeddings = [self.get_embedding(chunk) for chunk in self.chunks]
    self.index.add(np.array(embeddings))
```

#### Query System
```python
def query(self, question):
    # Find relevant chunks and generate answer
    query_embedding = self.get_embedding(question)
    _, indices = self.index.search(np.array([query_embedding]), 3)
    context = "\n\n".join(self.chunks[i] for i in indices[0])
    return self.generate_answer(question, context)
```

### 5. Using the System
```python
if __name__ == "__main__":
    rag = RAGSystem()
    text = rag.load_document("document.pdf")
    rag.process_text(text)
    
    while True:
        question = input("Ask a question (or 'quit'): ")
        if question.lower() == 'quit': break
        print(f"Answer: {rag.query(question)}")
```

## Key Concepts

- **Embeddings**: Numbers that represent text meaning
- **FAISS**: Fast library for finding similar vectors
- **Chunking**: Breaking text into smaller pieces
- **Context**: Relevant information for answering questions

## Tips for Improvement
- Try different text chunk sizes
- Experiment with different AI models
- Add support for more file types
- Test with different questions

