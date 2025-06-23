"""
RAG (Retrieval Augmented Generation) implementation for the fact-checking system.
"""
import os
import torch
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dataclasses import dataclass
import json
from src.utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    def __init__(
        self,
        knowledge_base_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        rerank_top_n: int = 10,
        rerank_threshold: float = 0.6
    ):
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.rerank_top_n = rerank_top_n
        self.rerank_threshold = rerank_threshold

class RAGHandler:
    """Handler for RAG operations."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.index = None
        self.documents = []
        
        # Load or initialize components
        self._load_components()
    
    def _load_components(self):
        """Load or initialize RAG components."""
        # Load embedding model
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.model = SentenceTransformer(self.config.embedding_model)
        
        # Load or create FAISS index
        index_path = os.path.join(self.config.knowledge_base_path, "faiss_index.bin")
        documents_path = os.path.join(self.config.knowledge_base_path, "documents.json")
        
        if os.path.exists(index_path) and os.path.exists(documents_path):
            logger.info("Loading existing FAISS index and documents")
            self.index = faiss.read_index(index_path)
            with open(documents_path, 'r') as f:
                self.documents = json.load(f)
        else:
            logger.info("Creating new FAISS index")
            self.index = None
            self.documents = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the knowledge base."""
        if not documents:
            return
        
        # Get document texts
        texts = [doc["text"] for doc in documents]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Initialize or update FAISS index
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Add documents to list
        self.documents.extend(documents)
        
        # Save index and documents
        index_path = os.path.join(self.config.knowledge_base_path, "faiss_index.bin")
        documents_path = os.path.join(self.config.knowledge_base_path, "documents.json")
        
        os.makedirs(self.config.knowledge_base_path, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(documents_path, 'w') as f:
            json.dump(self.documents, f, indent=2)
    
    def query(self, query_text: str) -> List[Dict[str, Any]]:
        """Query the knowledge base."""
        if not self.index or not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query_text])[0]
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            self.config.top_k
        )
        
        # Get relevant documents
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["similarity_score"] = float(1 / (1 + distance))  # Convert distance to similarity
                if doc["similarity_score"] >= self.config.similarity_threshold:
                    results.append(doc)
        
        # Sort by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return results
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        # Get query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, self.config.top_k)
        
        # Filter and format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents) and dist < self.config.similarity_threshold:
                doc = self.documents[idx]
                results.append({
                    "document": doc,
                    "similarity_score": float(1 / (1 + dist))  # Convert distance to similarity score
                })
        
        return results
    
    def format_rag_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a prompt."""
        if not retrieved_docs:
            return query
        
        prompt = f"""Based on the following relevant information and the query, provide a detailed analysis:

Relevant Information:
"""
        
        for i, result in enumerate(retrieved_docs, 1):
            doc = result["document"]
            score = result["similarity_score"]
            prompt += f"\n{i}. {doc['text']} (Relevance: {score:.2f})"
        
        prompt += f"\n\nQuery: {query}\nAnalysis:"
        
        return prompt 