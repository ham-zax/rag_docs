from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from datetime import datetime

import openai
import pymupdf
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    collection_name: str = "documents"
    chunk_size: int = 600
    chunk_overlap: int = 150
    embedding_model: str = "text-embedding-3-small"
    completion_model: str = "gpt-4o-mini"
    embedding_dimension: int = 1536
    max_context_chunks: int = 12
    qdrant_path: str = "./qdrant_data"
    
class RAGSystem:
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG system with optional configuration"""
        self.config = config or RAGConfig()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(path=self.config.qdrant_path)
        self._init_collection()
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Add to existing init
        self.usage_stats = {
            "embedding_tokens": 0,
            "completion_input_tokens": 0,
            "completion_output_tokens": 0
        }

    def _init_collection(self) -> None:
        """Initialize Qdrant collection with hybrid search capabilities"""
        try:
            self.qdrant.get_collection(self.config.collection_name)
        except:
            self.qdrant.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=Distance.COSINE
                ),
                # Enable payload indexing for hybrid search
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,  # Index all vectors
                ),
                # Enable full-text search on content field
                on_disk_payload=True
            )
            
    def _get_document_hash(self, content: str) -> str:
        """Generate hash for document content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def estimate_processing_costs(self, content: str) -> Dict[str, float]:
        """Estimate processing costs for both embedding and completion"""
        try:
            encoding = tiktoken.encoding_for_model(self.config.embedding_model)
            num_tokens = len(encoding.encode(content))
            
            # Calculate costs
            embedding_cost = (num_tokens / 1_000_000) * 0.02
            estimated_completion_tokens = num_tokens * 0.2
            completion_cost = (estimated_completion_tokens / 1_000_000) * (0.15 + 0.60)
            
            return {
                "num_tokens": num_tokens,
                "embedding_cost": embedding_cost,
                "completion_cost": completion_cost,
                "total_cost": embedding_cost + completion_cost
            }
        except Exception as e:
            raise Exception(f"Error estimating costs: {str(e)}")

    def load_document(self, file_path: str) -> Optional[str]:
        """Enhanced load_document with cost estimation and user confirmation"""
        try:
            doc = pymupdf.open(file_path)
            num_pages = doc.page_count
            content = ""
            for page in doc:
                content += page.get_text()
            doc.close()

            if num_pages > 25:
                # Get cost estimates
                costs = self.estimate_processing_costs(content)
                
                print(f"""
Large Document Warning
---------------------
Document: {file_path}
Pages: {num_pages}
Estimated tokens: {costs['num_tokens']:,}
Cost Breakdown:
- Embedding: ${costs['embedding_cost']:.4f}
- Completion: ${costs['completion_cost']:.4f}
Total estimated cost: ${costs['total_cost']:.4f}

Would you like to proceed? (yes/no)""")
                
                response = input().lower()
                if response != 'yes':
                    print("Processing cancelled")
                    return None

            # Continue with existing processing...
            doc_hash = self._get_document_hash(content)
            existing = self.qdrant.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="doc_hash",
                        match=models.MatchValue(value=doc_hash)
                    )]
                )
            )

            if not existing[0]:
                self.process_text(content, doc_hash)

            return content

        except Exception as e:
            print(f"Error loading document: {e}")
            return None

    def process_text(self, text: str, doc_hash: str) -> None:
        """Process text and store in Qdrant"""
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Prepare points for Qdrant
        points = []
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            
            points.append(models.PointStruct(
                id=len(points),
                vector=embedding,
                payload={
                    "content": chunk,
                    "doc_hash": doc_hash,
                    "chunk_index": i,
                    "timestamp": datetime.now().isoformat()
                }
            ))
            
        # Upload to Qdrant in batches
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            self.qdrant.upsert(
                collection_name=self.config.collection_name,
                points=batch
            )
            
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not text or not text.strip():
            raise ValueError("Empty or invalid input text")
        try:
            # Ensure text is properly formatted
            cleaned_text = text.strip()
            response = openai.embeddings.create(
                input=cleaned_text,
                model=self.config.embedding_model
                )
        
            if not response.data or not response.data[0].embedding:
                raise ValueError("No embedding generated")
            embedding = response.data[0].embedding
            if len(embedding) != self.config.embedding_dimension:
                raise ValueError(f"Invalid embedding dimension: {len(embedding)}")    
            # Track embedding tokens
            self.usage_stats["embedding_tokens"] += response.usage.total_tokens
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
            
    def query(self, question: str) -> Optional[str]:
        """Query the RAG system with hybrid search"""
        if not question or not question.strip():
            return "Please provide a valid question."
        
        try:
            question_embedding = self.get_embedding(question)
            if not question_embedding:
                return "Unable to process question. Please try again."
            # Generate question embedding
            question_embedding = self.get_embedding(question)
            
            # Hybrid search using both semantic and keyword matching
            search_result = self.qdrant.search(
                collection_name=self.config.collection_name,
                query_vector=question_embedding,
                query_filter=None,  # Can add filters if needed
                limit=self.config.max_context_chunks,
                search_params=models.SearchParams(
                    hnsw_ef=128,  # Increase for better recall
                    exact=False   # Set to True for exact search
                ),
                with_payload=True,
                score_threshold=0.0  # Adjust based on needs
            )
            
            # Extract relevant chunks
            relevant_chunks = [hit.payload["content"] for hit in search_result]
            
            # Generate answer
            answer = self.generate_answer(question, relevant_chunks)
            
            # Update conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            return answer
            
        except Exception as e:
            print(f"Error during query: {e}")
            return None
            
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate answer using conversation history and context"""
        try:
            # Build context from chunks and recent conversation
            context = "\n\n---\n\n".join(context_chunks)
            
            # Include recent conversation history
            conv_context = ""
            if self.conversation_history:
                recent_conv = self.conversation_history[-3:]  # Last 3 exchanges
                conv_context = "\n".join([
                    f"Q: {conv['question']}\nA: {conv['answer']}"
                    for conv in recent_conv
                ])
            
            # Build prompt
            prompt = f"""Please answer the following question based on the provided context.
            If the answer cannot be derived from the context, say so.

            Previous conversation:
            {conv_context}

            Context:
            {context}

            Question: {question}
            
            Answer:"""
            
            # Generate response
            response = openai.chat.completions.create(
                model=self.config.completion_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Track completion tokens
            self.usage_stats["completion_input_tokens"] += response.usage.prompt_tokens
            self.usage_stats["completion_output_tokens"] += response.usage.completion_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return ""

    def calculate_final_costs(self) -> Dict[str, float]:
        """Calculate final costs based on usage"""
        embedding_cost = (self.usage_stats["embedding_tokens"] / 1_000_000) * 0.02
        completion_input_cost = (self.usage_stats["completion_input_tokens"] / 1_000_000) * 0.15
        completion_output_cost = (self.usage_stats["completion_output_tokens"] / 1_000_000) * 0.60
        
        total_cost = embedding_cost + completion_input_cost + completion_output_cost
        
        return {
            "usage": self.usage_stats,
            "costs": {
                "embedding_cost": embedding_cost,
                "completion_input_cost": completion_input_cost,
                "completion_output_cost": completion_output_cost,
                "total_cost": total_cost
            }
        }

def main():
    """Modified main function with cost reporting"""
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        qdrant_path="./qdrant_data"
    )
    
    rag = RAGSystem(config)
    
    try:
        # Load document
        text = rag.load_document("report.pdf")
        
        # Interactive query loop
        while True:
            question = input("Enter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            answer = rag.query(question)
            if answer:
                print(f"Answer: {answer}")
        
        # Display final costs
        final_costs = rag.calculate_final_costs()
        print("\nFinal Usage and Cost Report")
        print("==========================")
        print("\nToken Usage:")
        print(f"Embedding tokens: {final_costs['usage']['embedding_tokens']:,}")
        print(f"Completion input tokens: {final_costs['usage']['completion_input_tokens']:,}")
        print(f"Completion output tokens: {final_costs['usage']['completion_output_tokens']:,}")
        print("\nCosts:")
        print(f"Embedding cost: ${final_costs['costs']['embedding_cost']:.4f}")
        print(f"Completion input cost: ${final_costs['costs']['completion_input_cost']:.4f}")
        print(f"Completion output cost: ${final_costs['costs']['completion_output_cost']:.4f}")
        print(f"Total cost: ${final_costs['costs']['total_cost']:.4f}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
