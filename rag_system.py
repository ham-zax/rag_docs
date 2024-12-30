from typing import List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import hashlib
from datetime import datetime

import openai
import pymupdf
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

from dotenv import load_dotenv
load_dotenv()

@dataclass
class RAGConfig:
    """Streamlined configuration for RAG system"""
    collection_name: str = "documents"
    chunk_size: int = 600
    chunk_overlap: int = 150
    embedding_model: str = "text-embedding-3-small"
    completion_model: str = "gpt-4o-mini"
    embedding_dimension: int = 1536
    max_context_chunks: int = 12
    qdrant_path: str = "./qdrant_data"
    embedding_price_per_million: float = 0.02
    completion_input_price_per_million: float = 0.015
    completion_output_price_per_million: float = 0.60
    completion_token_estimation_multiplier: float = 0.2

class RAGSystem:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.qdrant = QdrantClient(path=self.config.qdrant_path)
        self._init_collection()
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.usage_stats = {"embedding_tokens": 0, "completion_tokens": 0}

    def _init_collection(self) -> None:
        """Initialize Qdrant collection if it doesn't exist"""
        try:
            self.qdrant.get_collection(self.config.collection_name)
        except:
            self.qdrant.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=Distance.COSINE
                )
            )

    def load_document(self, file_path: str) -> Optional[str]:
        """Simplified document loading"""
        try:
            doc = pymupdf.open(file_path)
            content = " ".join(page.get_text() for page in doc)
            doc.close()

            doc_hash = hashlib.md5(content.encode()).hexdigest()
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
        """Simplified text processing"""
        chunks = self.text_splitter.split_text(text)
        points = []
        
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            if embedding:
                points.append(models.PointStruct(
                    id=i,
                    vector=embedding,
                    payload={"content": chunk, "doc_hash": doc_hash}
                ))

        if points:
            self.qdrant.upsert(
                collection_name=self.config.collection_name,
                points=points
            )

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Simplified embedding generation"""
        try:
            response = openai.embeddings.create(
                input=text.strip(),
                model=self.config.embedding_model
            )
            self.usage_stats["embedding_tokens"] += response.usage.total_tokens
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def query(self, question: str) -> Optional[str]:
        """Simplified query processing"""
        try:
            question_embedding = self.get_embedding(question)
            if not question_embedding:
                return "Unable to process question."

            search_result = self.qdrant.search(
                collection_name=self.config.collection_name,
                query_vector=question_embedding,
                limit=self.config.max_context_chunks,
                with_payload=True
            )

            context = " ".join(hit.payload["content"] for hit in search_result)
            history = self.memory.load_memory_variables({})
            history_str = "\n".join([
                f"Human: {msg.content}" if isinstance(msg, HumanMessage) 
                else f"Assistant: {msg.content}"
                for msg in history.get("chat_history", [])
            ])

            prompt = f"""Previous conversation:
            {history_str}

            Context:
            {context}

            Question: {question}

            Answer:"""

            response = openai.chat.completions.create(
                model=self.config.completion_model,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content
            self.memory.save_context({"input": question}, {"output": answer})
            self.usage_stats["completion_tokens"] += (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
            return answer

        except Exception as e:
            print(f"Query error: {e}")
            return None

    def get_final_costs(self) -> Dict[str, float]:
        """Calculate final costs"""
        embedding_cost = (self.usage_stats["embedding_tokens"] / 1_000_000) * self.config.embedding_price_per_million
        completion_cost = (self.usage_stats["completion_tokens"] / 1_000_000) * (
            self.config.completion_input_price_per_million + self.config.completion_output_price_per_million
        )
        return {
            "embedding_cost": embedding_cost,
            "completion_cost": completion_cost,
            "total_cost": embedding_cost + completion_cost
        }

def main():
    rag = RAGSystem()
    
    try:
        if rag.load_document("report.pdf"):
            while True:
                question = input("\nEnter question (or 'quit'): ")
                if question.lower() == 'quit':
                    break
                    
                if answer := rag.query(question):
                    print(f"\nAnswer: {answer}")
            
            costs = rag.get_final_costs()
            print(f"\nTotal cost: ${costs['total_cost']:.4f}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
