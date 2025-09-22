"""
Vector Database Module for AI-Powered Delivery Failure Analysis

This module implements a vector database using ChromaDB for semantic search
and retrieval of delivery failure data. It provides document vectorization,
storage, and retrieval capabilities for the RAG system.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Vector database implementation using ChromaDB for semantic search.
    
    This class handles:
    - Document vectorization using sentence transformers
    - Vector storage and retrieval
    - Semantic search across multiple data sources
    - Metadata management for contextual retrieval
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 model_name: Optional[str] = None,
                 collection_name: str = "delivery_failures"):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to store the ChromaDB database (if None, will get from env)
            model_name: Name of the sentence transformer model (if None, will get from env)
            collection_name: Name of the ChromaDB collection
        """
        self.db_path = db_path or os.getenv("VECTOR_DB_PATH", "data/vector_db")
        self.model_name = model_name or os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize sentence transformer model
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Delivery failure analysis data"}
        )
        
        logger.info(f"Vector database initialized at {db_path}")
    
    def _create_document_text(self, row: pd.Series, source_name: str) -> str:
        """
        Convert a data row into a searchable text document.
        
        Args:
            row: Pandas Series representing a data row
            source_name: Name of the data source (e.g., 'orders', 'fleet_logs')
            
        Returns:
            Formatted text document for vectorization
        """
        # Create a comprehensive text representation of the data
        doc_parts = [f"Data Source: {source_name}"]
        
        # Add all non-null fields as key-value pairs
        for field, value in row.items():
            if pd.notna(value) and str(value).strip():
                doc_parts.append(f"{field}: {value}")
        
        return " | ".join(doc_parts)
    
    def _create_metadata(self, row: pd.Series, source_name: str, chunk_id: str) -> Dict[str, Any]:
        """
        Create metadata for a document.
        
        Args:
            row: Pandas Series representing a data row
            source_name: Name of the data source
            chunk_id: Unique identifier for the chunk
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "source": source_name,
            "chunk_id": chunk_id,
            "timestamp": datetime.now().isoformat(),
            "record_count": 1
        }
        
        # Add specific metadata based on source type
        if source_name == "orders":
            metadata.update({
                "order_id": str(row.get("order_id", "")),
                "client_id": str(row.get("client_id", "")),
                "status": str(row.get("status", "")),
                "city": str(row.get("city", "")),
                "state": str(row.get("state", ""))
            })
        elif source_name == "fleet_logs":
            metadata.update({
                "driver_id": str(row.get("driver_id", "")),
                "vehicle_id": str(row.get("vehicle_id", "")),
                "status": str(row.get("status", ""))
            })
        elif source_name == "warehouse_logs":
            metadata.update({
                "warehouse_id": str(row.get("warehouse_id", "")),
                "operation": str(row.get("operation", ""))
            })
        elif source_name == "external_factors":
            metadata.update({
                "factor_type": str(row.get("factor_type", "")),
                "severity": str(row.get("severity", ""))
            })
        elif source_name == "feedback":
            metadata.update({
                "client_id": str(row.get("client_id", "")),
                "rating": str(row.get("rating", "")),
                "sentiment": str(row.get("sentiment", ""))
            })
        
        return metadata
    
    def add_documents(self, 
                     df: pd.DataFrame, 
                     source_name: str, 
                     chunk_id: str = None) -> Dict[str, Any]:
        """
        Add documents to the vector database.
        
        Args:
            df: DataFrame containing the data to vectorize
            source_name: Name of the data source
            chunk_id: Optional chunk identifier
            
        Returns:
            Dictionary with operation results
        """
        if df.empty:
            logger.warning(f"Empty DataFrame provided for source {source_name}")
            return {"status": "skipped", "reason": "empty_dataframe"}
        
        if chunk_id is None:
            chunk_id = f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Adding {len(df)} documents from {source_name} (chunk: {chunk_id})")
        
        try:
            # Prepare documents and metadata
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in df.iterrows():
                # Create document text
                doc_text = self._create_document_text(row, source_name)
                documents.append(doc_text)
                
                # Create metadata
                metadata = self._create_metadata(row, source_name, chunk_id)
                metadatas.append(metadata)
                
                # Create unique ID
                doc_id = f"{source_name}_{chunk_id}_{idx}"
                ids.append(doc_id)
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            logger.info(f"Successfully added {len(documents)} documents to vector database")
            
            return {
                "status": "success",
                "documents_added": len(documents),
                "chunk_id": chunk_id,
                "source": source_name
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "source": source_name
            }
    
    def search(self, 
               query: str, 
               n_results: int = 10,
               source_filter: Optional[str] = None,
               metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            source_filter: Optional source name filter
            metadata_filter: Optional metadata filter
            
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Searching for: '{query}' (n_results={n_results})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Prepare where clause for filtering
            where_clause = {}
            if source_filter:
                where_clause["source"] = source_filter
            if metadata_filter:
                where_clause.update(metadata_filter)
            
            # Search the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            
            return {
                "status": "success",
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze sources
            sample_results = self.collection.get(limit=min(1000, count))
            sources = {}
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    source = metadata.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
            
            return {
                "status": "success",
                "total_documents": count,
                "sources": sources,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all documents from the collection.
        
        Returns:
            Dictionary with operation results
        """
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                logger.info(f"Cleared {len(all_docs['ids'])} documents from collection")
                return {
                    "status": "success",
                    "documents_cleared": len(all_docs['ids'])
                }
            else:
                return {
                    "status": "success",
                    "documents_cleared": 0,
                    "message": "Collection was already empty"
                }
                
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def close(self):
        """Close the database connection."""
        # ChromaDB persistent client doesn't need explicit closing
        logger.info("Vector database connection closed")


def main():
    """Test the vector database functionality."""
    # Initialize vector database
    vector_db = VectorDatabase()
    
    try:
        # Test with sample data
        sample_data = pd.DataFrame({
            'order_id': [1, 2, 3],
            'client_id': [101, 102, 103],
            'status': ['delivered', 'failed', 'delayed'],
            'city': ['New York', 'Los Angeles', 'Chicago'],
            'failure_reason': ['', 'Weather delay', 'Traffic congestion']
        })
        
        # Add sample documents
        result = vector_db.add_documents(sample_data, "orders", "test_chunk")
        print(f"Add documents result: {result}")
        
        # Get collection stats
        stats = vector_db.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Test search
        search_result = vector_db.search("delivery failure", n_results=5)
        print(f"Search result: {json.dumps(search_result, indent=2)}")
        
    finally:
        vector_db.close()


if __name__ == "__main__":
    main()
