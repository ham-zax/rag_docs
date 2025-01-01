from typing import List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import hashlib
from datetime import datetime
import os

from openai import OpenAI

client = OpenAI()
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
from langchain_community.document_loaders.csv_loader import CSVLoader

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

@dataclass
class RAGConfig:
    """Streamlined configuration for RAG system"""
    collection_name: str = "documents"
    chunk_size: int = 800
    chunk_overlap: int = 150
    embedding_model: str = "text-embedding-ada-002"
    completion_model: str = "gpt-4"
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

    def load_csv(self, file_path: str) -> Optional[str]:
        """Load and process CSV document"""
        try:
            # Configure CSV loader with correct columns
            loader = CSVLoader(
                file_path=file_path,
                csv_args={
                    "delimiter": ",",
                    "quotechar": '"',
                }
            )

            # Load CSV data
            documents = loader.load()

            # Process each document with metadata
            processed_content = []
            for doc in documents:
                content = doc.page_content
                metadata = {
                    "offense": doc.metadata.get("Offense", ""),
                    "punishment": doc.metadata.get("Punishment", ""),
                    "section": doc.metadata.get("Section", "")
                }

                # Combine content with metadata
                processed_text = f"""
                Section: {metadata['section']}
                Offense: {metadata['offense']}
                Punishment: {metadata['punishment']}
                Description: {content}
                """
                processed_content.append(processed_text)

            # Join all processed content
            full_content = "\n\n".join(processed_content)

            # Generate document hash
            doc_hash = hashlib.md5(full_content.encode()).hexdigest()

            # Check if already processed
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
                self.process_text(full_content, doc_hash)

            return full_content

        except Exception as e:
            st.error(f"Error loading CSV document: {e}")
            st.error(f"File path attempted: {os.path.abspath(file_path)}")
            return None


    def load_document(self, file_path: str) -> Optional[str]:
        """Load document based on file type"""
        try:
            # Detect file type
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.pdf':
                return self._load_pdf(file_path)
            elif file_ext == '.csv':
                return self.load_csv(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

        except Exception as e:
            st.error(f"Error loading document: {e}")
            return None

    def _load_pdf(self, file_path: str) -> Optional[str]:
        """Load PDF document (existing implementation)"""
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
            st.error(f"Error loading PDF document: {e}")
            return None

    def process_text(self, text: str, doc_hash: str) -> None:
        """Process and store text chunks"""
        chunks = self.text_splitter.split_text(text)
        points = []

        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            if embedding:
                points.append(models.PointStruct(
                    id=int(hashlib.md5((doc_hash + str(i)).encode()).hexdigest(), 16) % (10 ** 12),  # Unique ID
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
            response = client.embeddings.create(input=text.strip(),
            model=self.config.embedding_model)
            self.usage_stats["embedding_tokens"] += response.usage.total_tokens
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return None

    def query(self, question: str, session_id: str = "default") -> Optional[str]:
        """Query using LCEL chain with message history"""
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

            response = self.chain_with_history.invoke(
                {"question": question, "context": context},
                config={"configurable": {"session_id": session_id}}
            )

            estimated_tokens = len(response.split()) * self.config.completion_token_estimation_multiplier
            self.usage_stats["completion_tokens"] += int(estimated_tokens)

            return response

        except Exception as e:
            st.error(f"Query error: {e}")
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

def initialize_rag_system():
    if 'rag' not in st.session_state:
        st.session_state.rag = RAGSystem()
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'answers' not in st.session_state:
        st.session_state.answers = []

def main():
    st.set_page_config(page_title="RAG System with Streamlit", layout="wide")
    st.title("Retrieval-Augmented Generation (RAG) System")

    initialize_rag_system()
    rag: RAGSystem = st.session_state.rag
    session_id: str = st.session_state.session_id

    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing document..."):
            content = rag.load_document(file_path)
            if content:
                st.success("Document loaded and processed successfully!")
                st.session_state.questions = []
                st.session_state.answers = []
            else:
                st.error("Failed to process the document.")

    st.header("Ask a Question")
    question = st.text_input("Enter your question:", key="input_question")
    if st.button("Submit", key="submit_button") and question:
        with st.spinner("Generating answer..."):
            answer = rag.query(question, session_id=session_id)
            if answer:
                st.session_state.questions.append(question)
                st.session_state.answers.append(answer)
                st.success("Answer generated!")

    if st.session_state.questions:
        st.header("Conversation History")
        for q, a in zip(st.session_state.questions, st.session_state.answers):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.write("---")

    st.sidebar.header("Usage Statistics")
    embedding_cost = rag.get_embedding_cost()
    final_costs = rag.get_final_costs()
    st.sidebar.text(f"Embedding Tokens: {rag.usage_stats['embedding_tokens']}")
    st.sidebar.text(f"Completion Tokens: {rag.usage_stats['completion_tokens']}")
    st.sidebar.text(f"Embedding Cost: ${final_costs['embedding_cost']:.6f}")
    st.sidebar.text(f"Completion Cost: ${final_costs['completion_cost']:.6f}")
    st.sidebar.text(f"Total Cost: ${final_costs['total_cost']:.6f}")

    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Conversation"):
        st.session_state.questions = []
        st.session_state.answers = []
        st.sidebar.success("Conversation history reset.")

if __name__ == "__main__":
    main()
