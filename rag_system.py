import openai
import pymupdf
import faiss
import numpy as np
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

class RAGSystem:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        self.embedding_dimension = 1536
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.chunks = []
        
    def load_document(self, file_path):
        """Load and extract text from PDF document"""
        try:
            doc = pymupdf.open(file_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
            return full_text
        except Exception as e:
            print(f"Error loading document: {e}")
            return None
            
    def process_text(self, text):
        """Split text into chunks and generate embeddings"""
        self.chunks = self.text_splitter.split_text(text)
        embeddings = [self.get_embedding(chunk) for chunk in self.chunks]
        self.index.add(np.array(embeddings, dtype='float32'))
        
    def get_embedding(self, text):
        """Generate OpenAI embedding for text"""
        try:
            response = openai.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
            
    def query(self, question, k=3):
        """Query the RAG system"""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(question)
            if query_embedding is None:
                return None
                
            # Search for relevant chunks
            distances, indices = self.index.search(
                np.array([query_embedding], dtype='float32'), k
            )
            relevant_chunks = [self.chunks[i] for i in indices[0]]
            
            # Generate answer
            context = "\n\n".join(relevant_chunks)
            return self.generate_answer(question, context)
        except Exception as e:
            print(f"Error during query: {e}")
            return None
            
    def generate_answer(self, question, context):
        """Generate answer using OpenAI GPT model"""
        try:
            prompt = f"Answer the following question based on the context provided:\n\nContext:\n{context}\n\nQuestion: {question}"
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

# Example usage
if __name__ == "__main__":
    rag = RAGSystem()
    text = rag.load_document("document.pdf")
    if text:
        rag.process_text(text)
        while True:
            question = input("Enter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            answer = rag.query(question)
            if answer:
                print(f"Answer: {answer}")
