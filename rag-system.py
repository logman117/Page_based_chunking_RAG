import os
import json
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import logging
from embedding_generator import EmbeddingGenerator, SupabaseVectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates responses using Azure OpenAI GPT-4o."""
    
    def __init__(self):
        """Initialize with API credentials from environment variables."""
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2024-08-01-preview"
        self.model_name = "gpt-4o"
        
        if not self.api_base or not self.api_key:
            raise ValueError("Missing Azure OpenAI API credentials in environment variables")
        
        self.chat_endpoint = f"{self.api_base}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}"
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate_response(self, 
                              system_prompt: str,
                              user_query: str, 
                              context: str) -> str:
        """
        Generate a response using GPT-4o.
        
        Args:
            system_prompt: System prompt for the model
            user_query: User's query
            context: Retrieved context with page numbers
            
        Returns:
            Generated response
        """
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {user_query}\n\nContext:\n{context}"}
        ]
        
        payload = {
            "messages": messages,
            "temperature": 0.1,  # Lower temperature for more consistent responses
            "max_tokens": 1000
        }
        
        try:
            response = await self.client.post(
                self.chat_endpoint,
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Error generating response: {response.status_code} {response.text}")
                raise Exception(f"Error generating response: {response.status_code} {response.text}")
            
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            return response_text
            
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI API: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

class RAGSystem:
    """Retrieval-Augmented Generation system for Nilfisk service manual."""
    
    def __init__(self):
        """Initialize RAG system components."""
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = SupabaseVectorStore()
        self.response_generator = ResponseGenerator()
    
    async def process_query(self, 
                          query: str, 
                          top_k: int = 5) -> str:
        """
        Process a user query and generate a response with page citations.
        
        Args:
            query: User's query
            top_k: Number of chunks to retrieve
            
        Returns:
            Generated response with page citations
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Retrieve relevant chunks
            chunks = await self._retrieve_chunks(query_embedding, top_k)
            
            # Format context with page numbers
            context = self._format_context(chunks)
            
            # Generate system prompt
            system_prompt = self._create_system_prompt()
            
            # Generate response
            response = await self.response_generator.generate_response(
                system_prompt=system_prompt,
                user_query=query,
                context=context
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"An error occurred while processing your query: {e}"
    
    async def _retrieve_chunks(self, 
                             query_embedding: List[float],
                             top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from the vector store.
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of chunks with metadata and similarity scores
        """
        chunks = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            limit=top_k,
            similarity_threshold=0.7
        )
        
        # Sort by page number, then by similarity
        chunks.sort(key=lambda x: (x["page_number"], -x["similarity"]))
        
        return chunks
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks as context with page numbers.
        
        Args:
            chunks: List of chunks with metadata
            
        Returns:
            Formatted context string with page numbers
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            page_number = chunk["page_number"]
            context_parts.append(f"[Page {page_number}]\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for GPT-4o.
        
        Returns:
            System prompt instructing the model to cite page numbers
        """
        return """
You are an assistant for the Nilfisk service manual. Your task is to provide accurate and helpful information based on the manual content.

Important guidelines:
1. Base your responses ONLY on the provided context. Do not use any prior knowledge.
2. Always cite the specific page number where you found the information using the format "According to page X" or "[Page X]".
3. If multiple pages contain relevant information, cite all of them.
4. If the information isn't in the provided context, say "I couldn't find information about this in the service manual" rather than making up an answer.
5. If different pages contain conflicting information, present both and note the discrepancy.
6. Use clear, concise language appropriate for a technical manual.
7. When providing instructions, present them in step-by-step format for clarity.
8. When mentioning specific parts or components, include their part numbers if available in the context.

Remember: Always cite page numbers and only use information from the provided context.
"""
    
    async def close(self):
        """Close all clients."""
        await self.embedding_generator.close()
        await self.response_generator.close()

async def initialize_rag_system():
    """Initialize and return a RAG system instance."""
    return RAGSystem()

if __name__ == "__main__":
    # Example usage
    
    async def test_rag():
        rag_system = await initialize_rag_system()
        
        try:
            # Test with a sample query
            query = "How do I replace the motor brushes in the Nilfisk vacuum?"
            response = await rag_system.process_query(query)
            
            print(f"Query: {query}")
            print("-" * 50)
            print(f"Response:\n{response}")
            
        finally:
            await rag_system.close()
    
    asyncio.run(test_rag())
