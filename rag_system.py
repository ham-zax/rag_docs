from typing import List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import hashlib
from datetime import datetime
import os

import openai
import pymupdf
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

@dataclass
class RAGConfig:
    """Streamlined configuration for RAG system"""
    collection_name: str = "documents"
    chunk_size: int = 800
    chunk_overlap: int = 150
    embedding_model: str = "text-embedding-3-small"
    completion_model: str = "gpt-4o-mini"
    embedding_dimension: int = 1536
    max_context_chunks: int = 10
    qdrant_path: str = "./qdrant_data"
    message_store_path: str = "./message_store"
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
        
        # Ensure message store directory exists
        os.makedirs(self.config.message_store_path, exist_ok=True)
        
        # Create LCEL chain components
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant answering questions based on the provided context."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        
        self.llm = ChatOpenAI(
            model=self.config.completion_model,
            temperature=0
        )
        
        # Construct LCEL chain
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # Wrap with message history
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: FileChatMessageHistory(
                os.path.join(self.config.message_store_path, f"{session_id}.json")
            ),
            input_messages_key="question",
            history_messages_key="history"
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
        """Load and process document"""
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
        """Process and store text chunks"""
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
        """Generate embeddings for text"""
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

    def query(self, question: str, session_id: str = "default") -> Optional[str]:
        """Query using LCEL chain with message history"""
        try:
            question_embedding = self.get_embedding(question)
            if not question_embedding:
                return "Unable to process question."

            # Debug: Print question embedding (truncated)
            # print("\nDEBUG: Question Embedding (first 5 dimensions):")
            # print(question_embedding[:5])

            search_result = self.qdrant.search(
                collection_name=self.config.collection_name,
                query_vector=question_embedding,
                limit=self.config.max_context_chunks,
                with_payload=True
            )

            # Debug: Print search results with scores
            # print("\nDEBUG: Top matched chunks with scores:")
            # for i, hit in enumerate(search_result):
                # print(f"\nMatch {i+1} (Score: {hit.score:.4f}):")
                # # print(f"Content preview: {hit.payload['content'][:200]}...")

            context = " ".join(hit.payload["content"] for hit in search_result)

            # Debug: Print context length and preview
            # print(f"\nDEBUG: Total context length: {len(context)} characters")
            # print(f"Context preview: {context[:200]}...")

            response = self.chain_with_history.invoke(
                {"question": question, "context": context},
                config={"configurable": {"session_id": session_id}}
            )

            estimated_tokens = len(response.split()) * self.config.completion_token_estimation_multiplier
            self.usage_stats["completion_tokens"] += int(estimated_tokens)

            return response

        except Exception as e:
            print(f"Query error: {e}")
            return None


    def get_embedding_cost(self) -> float:
        """Calculate embedding cost"""
        return (self.usage_stats["embedding_tokens"] / 1_000_000) * self.config.embedding_price_per_million

    def get_final_costs(self) -> Dict[str, float]:
        """Calculate final costs"""
        embedding_cost = self.get_embedding_cost()
        completion_cost = (self.usage_stats["completion_tokens"] / 1_000_000) * (
            self.config.completion_input_price_per_million + 
            self.config.completion_output_price_per_million
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
            session_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            while True:
                question = input("\nEnter question (or 'quit'): ")
                if question.lower() == 'quit':
                    costs = rag.get_final_costs()
                    print(f"\nSession costs:")
                    print(f"Embedding cost: ${costs['embedding_cost']:.4f}")
                    print(f"Completion cost: ${costs['completion_cost']:.4f}")
                    print(f"Total cost: ${costs['total_cost']:.4f}")
                    break

                if answer := rag.query(question, session_id=session_id):
                    print(f"\nAnswer: {answer}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
