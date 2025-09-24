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
                 provider: Optional[str] = None):
        """
        Initialize the RAG engine.
        
        Args:
            vector_db: VectorDatabase instance
            data_foundation: StreamingDataFoundation instance
            api_key: API key for LLM provider (if None, will try to get from env)
            model_name: Model to use (if None, will get from env)
            temperature: Temperature for response generation (if None, will get from env)
            provider: LLM provider ("openai" or "openrouter") - if None, will get from env
        """
        self.vector_db = vector_db
        self.data_foundation = data_foundation
        
        # Get provider from environment if not specified
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        
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
                r"recent", r"last", r"during"
            ],
            "comparison": [
                r"compare", r"versus", r"vs", r"better", r"worse", r"difference",
                r"ranking", r"top", r"bottom", r"best", r"worst", r"between.*and",
                r"analyze.*differences", r"contrast", r"relative"
            ],
            "prediction": [
                r"predict", r"forecast", r"future", r"likely", r"expected",
                r"trend", r"pattern", r"will.*happen", r"should.*expect",
                r"what.*risks", r"mitigate", r"onboard", r"scaling",
                r"capacity", r"risk.*assessment", r"what.*happen"
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
1. Comparative analysis between all mentioned cities/locations
2. Key differences and similarities with specific data points
3. Rankings or performance comparisons with metrics
4. City-specific insights and patterns
5. Recommendations based on comparisons

Important: Make sure to analyze data for ALL cities mentioned in the query. If the context contains data for multiple cities, provide a comprehensive comparison across all of them.

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
        
        # Prioritize comparison over temporal when both match
        if intent_scores.get("comparison", 0) > 0 and intent_scores.get("temporal", 0) > 0:
            if intent_scores["comparison"] >= intent_scores["temporal"]:
                return "comparison"
        
        # Return the intent with highest score, or default
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        return "default"
    
    def _retrieve_context(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from the vector database and direct database queries.
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            # First, try direct database queries for specific date/location combinations
            direct_results = self._get_direct_database_context(query)
            
            # Then, search vector database
            search_result = self.vector_db.search(query, n_results=n_results)
            
            if search_result["status"] == "success":
                vector_results = search_result["results"]
            else:
                logger.error(f"Vector search failed: {search_result.get('error', 'Unknown error')}")
                vector_results = []
            
            # Check if this is a comparison query with multiple cities
            analysis = self._analyze_query_with_llm(query)
            cities = analysis.get('cities', [])
            is_comparison_query = len(cities) > 1 and any(word in query.lower() for word in ['compare', 'versus', 'vs', 'between'])
            
            if is_comparison_query and direct_results:
                # For comparison queries, ensure balanced representation of all cities
                balanced_results = self._balance_city_representation(direct_results, cities, n_results)
                combined_results = balanced_results + vector_results
            else:
                # Combine results, prioritizing direct database results
                combined_results = direct_results + vector_results
            
            # Remove duplicates based on order_id
            seen_order_ids = set()
            unique_results = []
            for result in combined_results:
                order_id = result.get('metadata', {}).get('order_id', '')
                if order_id and order_id not in seen_order_ids:
                    seen_order_ids.add(order_id)
                    unique_results.append(result)
                elif not order_id:  # Include results without order_id (from other sources)
                    unique_results.append(result)
            
            # Limit to requested number of results
            return unique_results[:n_results]
                
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def _balance_city_representation(self, direct_results: List[Dict[str, Any]], cities: List[str], n_results: int) -> List[Dict[str, Any]]:
        """
        Balance city representation in results for comparison queries.
        
        Args:
            direct_results: List of direct database results
            cities: List of cities mentioned in the query
            n_results: Total number of results to return
            
        Returns:
            Balanced list of results with equal representation from each city
        """
        try:
            # Group results by city
            city_groups = {}
            for result in direct_results:
                city = result.get('metadata', {}).get('city', '')
                if city in cities:
                    if city not in city_groups:
                        city_groups[city] = []
                    city_groups[city].append(result)
            
            # Calculate how many results to take from each city
            num_cities = len(city_groups)
            if num_cities == 0:
                return direct_results[:n_results]
            
            # Reserve some slots for vector results (about 30%)
            vector_slots = max(2, n_results // 3)
            direct_slots = n_results - vector_slots
            results_per_city = max(1, direct_slots // num_cities)
            
            balanced_results = []
            
            # Take equal number of results from each city
            for city in cities:
                if city in city_groups:
                    city_results = city_groups[city][:results_per_city]
                    balanced_results.extend(city_results)
                    logger.info(f"Added {len(city_results)} results for {city}")
            
            # If we have space, add more results from cities with more data
            remaining_slots = direct_slots - len(balanced_results)
            if remaining_slots > 0:
                # Sort cities by number of available results
                sorted_cities = sorted(city_groups.items(), key=lambda x: len(x[1]), reverse=True)
                
                for city, city_results in sorted_cities:
                    if remaining_slots <= 0:
                        break
                    
                    # Add one more result from this city if available
                    if len(city_results) > results_per_city:
                        additional_result = city_results[results_per_city]
                        balanced_results.append(additional_result)
                        remaining_slots -= 1
                        logger.info(f"Added additional result for {city}")
            
            logger.info(f"Balanced results: {len(balanced_results)} total, {len(city_groups)} cities represented")
            return balanced_results
            
        except Exception as e:
            logger.error(f"Error balancing city representation: {str(e)}")
            return direct_results[:n_results]
    
    def _analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze the query and extract relevant entities and intent.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with extracted entities and query analysis
        """
        try:
            analysis_prompt = f"""
            Analyze this delivery/logistics query and extract relevant information:
            
            Query: "{query}"
            
            Please extract and return a JSON response with the following structure:
            {{
                "cities": ["list of cities mentioned"],
                "dates": ["list of dates mentioned in YYYY-MM-DD format"],
                "date_ranges": ["list of date ranges like 'last week', 'yesterday'"],
                "query_intent": "delays|failures|performance|status|other",
                "temporal_scope": "specific_date|date_range|recent|all_time",
                "geographic_scope": "specific_city|region|all_locations",
                "status_terms": ["delayed", "failed", "successful", "pending"],
                "time_terms": ["yesterday", "today", "last week", "this month"],
                "keywords": ["list of important keywords for search"]
            }}
            
            Rules:
            - Convert all dates to YYYY-MM-DD format
            - Extract city names as they appear in the query
            - Identify the main intent of the query
            - Be specific about temporal and geographic scope
            - Include relevant keywords for better search
            """
            
            response = self.llm.invoke(analysis_prompt)
            
            if response and hasattr(response, 'content') and response.content:
                import json
                try:
                    # Try to extract JSON from the response
                    content = response.content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    
                    analysis = json.loads(content)
                    logger.info(f"Query analysis: {analysis}")
                    return analysis
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response as JSON: {content}")
                    return self._fallback_query_analysis(query)
            else:
                logger.warning("Empty LLM response for query analysis")
                return self._fallback_query_analysis(query)
                
        except Exception as e:
            logger.error(f"Error in LLM query analysis: {str(e)}")
            return self._fallback_query_analysis(query)
    
    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """
        Fallback query analysis using simple patterns when LLM fails.
        
        Args:
            query: User query
            
        Returns:
            Basic analysis dictionary
        """
        import re
        from datetime import datetime, timedelta
        
        # Simple city extraction
        cities = []
        city_patterns = [
            r'\b(?:in|at|to|from)\s+([A-Za-z\s]+?)(?:\s+on|\s+for|\s+in|\s+at|\s+to|\s+from|\s+deliveries|\s+orders|\s+$)',
            r'\b(?:deliveries|orders|delivery|order)\s+(?:in|at|to|from)\s+([A-Za-z\s]+?)(?:\s+on|\s+for|\s+in|\s+at|\s+to|\s+from|\s+$)',
        ]
        
        for pattern in city_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                city = match.strip()
                if city.lower() not in ['why', 'what', 'how', 'when', 'where', 'deliveries', 'orders', 'delivery', 'order', 'delayed', 'failed']:
                    cities.append(city)
        
        # Simple date extraction
        dates = []
        date_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})\b',
            r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 3:
                    if match[0] in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
                        month, day, year = match
                    else:
                        day, month, year = match
                    
                    month_num = {
                        'january': '01', 'february': '02', 'march': '03', 'april': '04',
                        'may': '05', 'june': '06', 'july': '07', 'august': '08',
                        'september': '09', 'october': '10', 'november': '11', 'december': '12'
                    }.get(month.lower(), '01')
                    
                    date_str = f"{year}-{month_num}-{day.zfill(2)}"
                    dates.append(date_str)
                else:
                    dates.append(match)
        
        return {
            "cities": cities,
            "dates": dates,
            "date_ranges": [],
            "query_intent": "delays" if any(word in query.lower() for word in ['delay', 'delayed', 'late', 'slow']) else "other",
            "temporal_scope": "specific_date" if dates else "all_time",
            "geographic_scope": "specific_city" if cities else "all_locations",
            "status_terms": [word for word in ['delayed', 'failed', 'successful', 'pending'] if word in query.lower()],
            "time_terms": [word for word in ['yesterday', 'today', 'last week', 'this month'] if word in query.lower()],
            "keywords": query.lower().split()
        }
    
    def _get_direct_database_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Get context directly from database using LLM-analyzed query entities.
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents from direct database queries
        """
        try:
            import pandas as pd
            from datetime import datetime, timedelta
            
            # Analyze query with LLM
            analysis = self._analyze_query_with_llm(query)
            logger.info(f"Query analysis result: {analysis}")
            
            cities = analysis.get('cities', [])
            dates = analysis.get('dates', [])
            date_ranges = analysis.get('date_ranges', [])
            
            # Handle date ranges
            if date_ranges and not dates:
                for date_range in date_ranges:
                    if 'yesterday' in date_range.lower():
                        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                        dates.append(yesterday)
                    elif 'today' in date_range.lower():
                        today = datetime.now().strftime('%Y-%m-%d')
                        dates.append(today)
                    elif 'last week' in date_range.lower():
                        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                        dates.append(week_ago)
            
            # If no specific cities or dates, return empty
            if not cities and not dates:
                logger.info("No specific cities or dates found in query")
                return []
            
            # Query database for combinations of cities and dates found
            all_documents = []
            
            # If we have specific cities and dates
            if cities and dates:
                for city in cities:
                    for date_str in dates:
                        try:
                            # Validate date format
                            datetime.strptime(date_str, '%Y-%m-%d')
                            
                            # Query database for specific city and date
                            sql_query = f"""
                                SELECT * FROM orders 
                                WHERE city = '{city}' AND DATE(order_date) = '{date_str}'
                                ORDER BY order_date
                            """
                            
                            df = pd.read_sql_query(sql_query, self.data_foundation.db)
                            
                            if not df.empty:
                                logger.info(f"Found {len(df)} direct database results for {city} on {date_str}")
                                
                                # Convert to document format
                                for _, row in df.iterrows():
                                    doc_text = self.vector_db._create_document_text(row, "orders")
                                    metadata = self.vector_db._create_metadata(row, "orders", f"direct_{row['order_id']}")
                                    
                                    all_documents.append({
                                        "id": f"direct_{row['order_id']}",
                                        "document": doc_text,
                                        "metadata": metadata,
                                        "distance": 0.0  # Direct database results get highest priority
                                    })
                        
                        except ValueError:
                            logger.warning(f"Invalid date format: {date_str}")
                            continue
            
            # If we have cities but no specific dates, get recent data
            elif cities and not dates:
                for city in cities:
                    # Get recent orders for this city (last 30 days)
                    sql_query = f"""
                        SELECT * FROM orders 
                        WHERE city = '{city}' 
                        AND order_date >= date('now', '-30 days')
                        ORDER BY order_date DESC
                        LIMIT 50
                    """
                    
                    df = pd.read_sql_query(sql_query, self.data_foundation.db)
                    
                    if not df.empty:
                        logger.info(f"Found {len(df)} recent orders for {city}")
                        
                        for _, row in df.iterrows():
                            doc_text = self.vector_db._create_document_text(row, "orders")
                            metadata = self.vector_db._create_metadata(row, "orders", f"direct_{row['order_id']}")
                            
                            all_documents.append({
                                "id": f"direct_{row['order_id']}",
                                "document": doc_text,
                                "metadata": metadata,
                                "distance": 0.1  # Recent data gets high priority
                            })
            
            # If we have dates but no specific cities, get all cities for those dates
            elif dates and not cities:
                for date_str in dates:
                    try:
                        datetime.strptime(date_str, '%Y-%m-%d')
                        
                        sql_query = f"""
                            SELECT * FROM orders 
                            WHERE DATE(order_date) = '{date_str}'
                            ORDER BY order_date
                            LIMIT 100
                        """
                        
                        df = pd.read_sql_query(sql_query, self.data_foundation.db)
                        
                        if not df.empty:
                            logger.info(f"Found {len(df)} orders for date {date_str}")
                            
                            for _, row in df.iterrows():
                                doc_text = self.vector_db._create_document_text(row, "orders")
                                metadata = self.vector_db._create_metadata(row, "orders", f"direct_{row['order_id']}")
                                
                                all_documents.append({
                                    "id": f"direct_{row['order_id']}",
                                    "document": doc_text,
                                    "metadata": metadata,
                                    "distance": 0.2  # Date-specific data gets medium priority
                                })
                    
                    except ValueError:
                        logger.warning(f"Invalid date format: {date_str}")
                        continue
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Error in direct database context retrieval: {str(e)}")
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
            
            # Check if response is valid
            if not response or not hasattr(response, 'content'):
                logger.error(f"Invalid LLM response: {response}")
                return {
                    "response": "Error: Invalid response from LLM",
                    "tokens_used": 0,
                    "error": "Invalid LLM response"
                }
            
            # Extract token usage if available
            tokens_used = 0
            if hasattr(response, 'response_metadata'):
                tokens_used = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            
            # Ensure response content is not None
            response_content = response.content if response.content else "No response content generated"
            
            return {
                "response": response_content,
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
            
            # Check if llm_result is valid
            if not llm_result or not isinstance(llm_result, dict):
                logger.error(f"Invalid LLM result: {llm_result}")
                llm_result = {
                    "response": "Error: Unable to generate response from LLM",
                    "tokens_used": 0,
                    "error": "Invalid LLM result"
                }
            
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
                response=llm_result.get("response", "No response generated"),
                context_documents=context_documents,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time,
                tokens_used=llm_result.get("tokens_used", 0)
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
