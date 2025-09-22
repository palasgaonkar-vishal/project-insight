"""
RAG Engine Module for AI-Powered Delivery Failure Analysis

This module implements the core RAG (Retrieval-Augmented Generation) functionality
that combines vector search with LLM to answer questions about delivery failures.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from dataclasses import dataclass

import openai
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a RAG query."""
    query: str
    response: str
    context_documents: List[Dict[str, Any]]
    sources: List[str]
    confidence: float
    processing_time: float
    tokens_used: int


class RAGEngine:
    """
    RAG Engine for delivery failure analysis.
    
    This class handles:
    - Query parsing and intent classification
    - Context retrieval from vector database
    - LLM integration for response generation
    - Prompt engineering for delivery analysis
    """
    
    def __init__(self, 
                 vector_db,
                 data_foundation,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 provider: str = "openai"):
        """
        Initialize the RAG engine.
        
        Args:
            vector_db: VectorDatabase instance
            data_foundation: StreamingDataFoundation instance
            api_key: API key for LLM provider (if None, will try to get from env)
            model_name: Model to use (if None, will get from env)
            temperature: Temperature for response generation (if None, will get from env)
            provider: LLM provider ("openai" or "openrouter")
        """
        self.vector_db = vector_db
        self.data_foundation = data_foundation
        self.provider = provider.lower()
        
        # Get configuration from environment variables or parameters
        if self.provider == "openrouter":
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free")
            self.temperature = temperature or float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
        else:  # openai
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4")
            self.temperature = temperature or float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        
        # Set up LLM based on provider
        if self.api_key:
            if self.provider == "openrouter":
                os.environ["OPENROUTER_API_KEY"] = self.api_key
                # Use ChatOpenAI with OpenRouter's API endpoint
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/palasgaonkar-vishal/project-insight",
                        "X-Title": "AI-Powered Delivery Failure Analysis"
                    }
                )
            else:  # openai
                os.environ["OPENAI_API_KEY"] = self.api_key
                openai.api_key = self.api_key
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=self.api_key
                )
        else:
            logger.warning(f"No {self.provider.upper()} API key found. RAG functionality will be limited.")
            self.llm = None
        
        # Query intent patterns
        self.intent_patterns = {
            "count": [
                r"how many", r"count", r"number of", r"total", r"quantity"
            ],
            "failure_analysis": [
                r"why.*fail", r"failure.*reason", r"what.*caused", r"root cause",
                r"delivery.*problem", r"issue.*with", r"trouble.*with"
            ],
            "performance": [
                r"performance", r"efficiency", r"speed", r"time", r"duration",
                r"how.*fast", r"how.*slow", r"delays"
            ],
            "geographic": [
                r"city", r"state", r"location", r"region", r"area", r"where",
                r"geographic", r"geographical"
            ],
            "temporal": [
                r"when", r"date", r"time", r"period", r"day", r"week", r"month",
                r"recent", r"last", r"during", r"between"
            ],
            "comparison": [
                r"compare", r"versus", r"vs", r"better", r"worse", r"difference",
                r"ranking", r"top", r"bottom", r"best", r"worst"
            ],
            "prediction": [
                r"predict", r"forecast", r"future", r"likely", r"expected",
                r"trend", r"pattern", r"will.*happen"
            ]
        }
        
        # Initialize prompt templates
        self._initialize_prompt_templates()
        
        logger.info(f"RAG Engine initialized with model: {model_name}")
    
    def _initialize_prompt_templates(self):
        """Initialize prompt templates for different query types."""
        
        self.system_prompt = """You are an expert delivery operations analyst with deep knowledge of logistics, supply chain management, and delivery failure analysis. 

Your role is to analyze delivery data and provide insightful, actionable responses to help operations managers understand and improve delivery performance.

Key capabilities:
- Analyze delivery failure patterns and root causes
- Identify performance trends and bottlenecks
- Provide data-driven insights and recommendations
- Explain complex logistics scenarios in simple terms
- Suggest actionable improvements for delivery operations

Always base your responses on the provided context data and be specific about findings, patterns, and recommendations."""

        self.prompt_templates = {
            "count": """Based on the following context data, answer the query about counts or quantities:

Context Data:
{context}

Query: {query}

Please provide:
1. The specific count/quantity requested
2. Brief explanation of what this number represents
3. Any relevant context or trends

Answer:""",

            "failure_analysis": """Analyze the following delivery failure data to answer the query about failure reasons or root causes:

Context Data:
{context}

Query: {query}

Please provide:
1. Root cause analysis of the failures
2. Key patterns or trends identified
3. Specific examples from the data
4. Actionable recommendations to prevent similar failures

Answer:""",

            "performance": """Analyze the following delivery performance data to answer the query:

Context Data:
{context}

Query: {query}

Please provide:
1. Performance metrics and analysis
2. Key performance indicators
3. Trends and patterns identified
4. Recommendations for improvement

Answer:""",

            "geographic": """Analyze the following geographic delivery data to answer the query:

Context Data:
{context}

Query: {query}

Please provide:
1. Geographic patterns and trends
2. Location-specific insights
3. Regional performance differences
4. Geographic recommendations

Answer:""",

            "temporal": """Analyze the following time-based delivery data to answer the query:

Context Data:
{context}

Query: {query}

Please provide:
1. Time-based patterns and trends
2. Temporal analysis of the data
3. Time-specific insights
4. Recommendations based on timing

Answer:""",

            "comparison": """Compare the following delivery data to answer the query:

Context Data:
{context}

Query: {query}

Please provide:
1. Comparative analysis
2. Key differences and similarities
3. Rankings or performance comparisons
4. Recommendations based on comparisons

Answer:""",

            "prediction": """Based on the following historical delivery data, provide predictive analysis:

Context Data:
{context}

Query: {query}

Please provide:
1. Predictive insights based on patterns
2. Likely future scenarios
3. Risk factors and opportunities
4. Recommendations for future planning

Answer:""",

            "default": """Analyze the following delivery data to answer the query:

Context Data:
{context}

Query: {query}

Please provide a comprehensive analysis including:
1. Key findings from the data
2. Relevant patterns or trends
3. Specific insights and observations
4. Actionable recommendations

Answer:"""
        }
    
    def _classify_query_intent(self, query: str) -> str:
        """
        Classify the intent of a query based on patterns.
        
        Args:
            query: User query string
            
        Returns:
            Intent classification string
        """
        query_lower = query.lower()
        
        # Score each intent based on pattern matches
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent] = score
        
        # Return the intent with highest score, or default
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        return "default"
    
    def _retrieve_context(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from the vector database.
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            # Search vector database
            search_result = self.vector_db.search(query, n_results=n_results)
            
            if search_result["status"] == "success":
                return search_result["results"]
            else:
                logger.error(f"Vector search failed: {search_result.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get('metadata', {}).get('source', 'unknown')
            document_text = doc.get('document', '')
            distance = doc.get('distance', 0)
            
            context_parts.append(f"Source {i} ({source}, relevance: {1-distance:.2f}):")
            context_parts.append(document_text)
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, intent: str) -> Dict[str, Any]:
        """
        Generate response using LLM.
        
        Args:
            query: User query
            context: Formatted context
            intent: Query intent
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.llm:
            return {
                "response": "LLM not available. Please check OpenAI API key configuration.",
                "tokens_used": 0,
                "error": "No LLM configured"
            }
        
        try:
            # Get appropriate prompt template
            template = self.prompt_templates.get(intent, self.prompt_templates["default"])
            
            # Format the prompt
            formatted_prompt = template.format(
                context=context,
                query=query
            )
            
            # Create messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=formatted_prompt)
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            
            # Extract token usage if available
            tokens_used = 0
            if hasattr(response, 'response_metadata'):
                tokens_used = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            
            return {
                "response": response.content,
                "tokens_used": tokens_used,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": f"Error generating response: {str(e)}",
                "tokens_used": 0,
                "error": str(e)
            }
    
    def query(self, query: str, n_context: int = 10) -> QueryResult:
        """
        Process a query and return a comprehensive result.
        
        Args:
            query: User query string
            n_context: Number of context documents to retrieve
            
        Returns:
            QueryResult object with response and metadata
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Classify query intent
            intent = self._classify_query_intent(query)
            logger.info(f"Query intent classified as: {intent}")
            
            # Retrieve context
            context_documents = self._retrieve_context(query, n_context)
            logger.info(f"Retrieved {len(context_documents)} context documents")
            
            # Format context
            context = self._format_context(context_documents)
            
            # Generate response
            llm_result = self._generate_response(query, context, intent)
            
            # Extract sources
            sources = list(set([
                doc.get('metadata', {}).get('source', 'unknown') 
                for doc in context_documents
            ]))
            
            # Calculate confidence (simple heuristic based on context relevance)
            confidence = 0.8 if context_documents else 0.3
            if context_documents:
                avg_distance = sum(doc.get('distance', 1) for doc in context_documents) / len(context_documents)
                confidence = max(0.1, 1 - avg_distance)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=query,
                response=llm_result["response"],
                context_documents=context_documents,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time,
                tokens_used=llm_result["tokens_used"]
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=query,
                response=f"Error processing query: {str(e)}",
                context_documents=[],
                sources=[],
                confidence=0.0,
                processing_time=processing_time,
                tokens_used=0
            )
    
    def get_query_suggestions(self) -> List[str]:
        """
        Get sample query suggestions for users.
        
        Returns:
            List of sample queries
        """
        return [
            "How many orders failed in the last month?",
            "What are the main reasons for delivery failures?",
            "Which cities have the highest failure rates?",
            "How does weather affect delivery performance?",
            "What is the average delivery time for successful orders?",
            "Which drivers have the best performance?",
            "What are the most common customer complaints?",
            "How has delivery performance changed over time?",
            "Which warehouses have the most dispatch delays?",
            "What external factors cause the most delays?"
        ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG system.
        
        Returns:
            Dictionary with system status information
        """
        try:
            # Check vector database status
            vector_stats = self.vector_db.get_collection_stats()
            
            # Check data foundation status
            data_stats = self.data_foundation.get_data_statistics()
            
            return {
                "status": "operational",
                "llm_available": self.llm is not None,
                "model_name": self.model_name,
                "vector_db_status": vector_stats["status"],
                "total_documents": vector_stats.get("total_documents", 0),
                "data_sources": data_stats.get("sources", {}),
                "intent_patterns": list(self.intent_patterns.keys()),
                "prompt_templates": list(self.prompt_templates.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def close(self):
        """Close the RAG engine and clean up resources."""
        if hasattr(self, 'vector_db') and self.vector_db:
            self.vector_db.close()
        if hasattr(self, 'data_foundation') and self.data_foundation:
            self.data_foundation.close()
        logger.info("RAG Engine closed.")


def main():
    """Test the RAG engine functionality."""
    import sys
    sys.path.append('src')
    
    from vector_database import VectorDatabase
    from data_foundation import StreamingDataFoundation
    
    # Initialize components
    vector_db = VectorDatabase()
    data_foundation = StreamingDataFoundation()
    
    # Initialize RAG engine
    rag_engine = RAGEngine(vector_db, data_foundation)
    
    try:
        # Test queries
        test_queries = [
            "How many orders failed?",
            "What are the main reasons for delivery failures?",
            "Which cities have delivery problems?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            result = rag_engine.query(query)
            
            print(f"Response: {result.response}")
            print(f"Sources: {result.sources}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Tokens Used: {result.tokens_used}")
        
        # Get system status
        print(f"\n{'='*60}")
        print("System Status:")
        print('='*60)
        status = rag_engine.get_system_status()
        print(json.dumps(status, indent=2))
        
    finally:
        vector_db.close()
        data_foundation.close()


if __name__ == "__main__":
    main()
