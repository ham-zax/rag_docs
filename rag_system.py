import openai
import pymupdf
import faiss
import numpy as np
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import lru_cache

# Custom exceptions
class RAGError(Exception):
    """Base exception for RAG system"""
    pass

class DocumentLoadError(RAGError):
    """Raised when document loading fails"""
    pass

class EmbeddingError(RAGError):
    """Raised when embedding generation fails"""
    pass

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model: str = "text-embedding-3-small"
    completion_model: str = "gpt-4"
    embedding_dimension: int = 1536
    max_context_chunks: int = 3
    cache_dir: Path = Path("./cache")

class RAGSystem:
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG system with optional configuration"""
        self.config = config or RAGConfig()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.index = faiss.IndexFlatL2(self.config.embedding_dimension)
        self.chunks: List[str] = []
        self._setup_cache()
        
    def _setup_cache(self) -> None:
        """Setup cache directory"""
        self.config.cache_dir.mkdir(exist_ok=True)
        
    def load_document(self, file_path: str) -> str:
        """
        Load and extract text from PDF document
        
        Args:
            file_path: Path to PDF document
            
        Returns:
            Extracted text from document
            
        Raises:
            DocumentLoadError: If document loading fails
        """
        try:
            doc = pymupdf.open(file_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            return full_text
        except Exception as e:
            raise DocumentLoadError(f"Failed to load document: {e}")
            
    def process_text(self, text: str) -> None:
        """
        Split text into chunks and generate embeddings
        
        Args:
            text: Text to process
        """
        self.chunks = self.text_splitter.split_text(text)
        embeddings = self._batch_get_embeddings(self.chunks)
        self.index.add(np.array(embeddings, dtype='float32'))
        self._save_state()
        
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate OpenAI embedding for text with caching
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            response = openai.embeddings.create(
                input=[text],
                model=self.config.embedding_model
            )
            return np.array(response.data[0].embedding, dtype='float32')
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}")
            
    def _batch_get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings in batches"""
        return [self.get_embedding(text) for text in texts]
            
    def query(self, question: str, k: Optional[int] = None) -> Optional[str]:
        """
        Query the RAG system
        
        Args:
            question: Query question
            k: Number of relevant chunks to retrieve
            
        Returns:
            Generated answer
        """
        try:
            k = k or self.config.max_context_chunks
            query_embedding = self.get_embedding(question)
            
            distances, indices = self.index.search(
                np.array([query_embedding], dtype='float32'), k
            )
            
            relevant_chunks = [self.chunks[i] for i in indices[0]]
            return self.generate_answer(question, relevant_chunks)
            
        except RAGError as e:
            print(f"Error during query: {e}")
            return None
            
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Generate answer using OpenAI GPT model
        
        Args:
            question: Query question
            context_chunks: Relevant context chunks
            
        Returns:
            Generated answer
        """
        try:
            context = "\n\n---\n\n".join(context_chunks)
            prompt = self._build_prompt(question, context)
            
            response = openai.chat.completions.create(
                model=self.config.completion_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
            
        except Exception as e:
            raise RAGError(f"Failed to generate answer: {e}")
            
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for answer generation"""
        return f"""Please answer the following question based on the provided context. 
        If the answer cannot be derived from the context, say so.

        Context:
        {context}

        Question: {question}
        
        Answer:"""
        
    def _save_state(self) -> None:
        """Save system state to cache"""
        state = {
            'chunks': self.chunks,
            'index': faiss.serialize_index(self.index)
        }
        with open(self.config.cache_dir / 'state.pkl', 'wb') as f:
            pickle.dump(state, f)
            
    def _load_state(self) -> None:
        """Load system state from cache"""
        try:
            with open(self.config.cache_dir / 'state.pkl', 'rb') as f:
                state = pickle.load(f)
            self.chunks = state['chunks']
            self.index = faiss.deserialize_index(state['index'])
        except FileNotFoundError:
            pass

def main():
    """Example usage"""
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        cache_dir=Path("./rag_cache")
    )
    
    rag = RAGSystem(config)
    
    try:
        text = rag.load_document("document.pdf")
        rag.process_text(text)
        
        while True:
            question = input("Enter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            answer = rag.query(question)
            if answer:
                print(f"Answer: {answer}")
                
    except RAGError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
