"""
AI-Powered Delivery Failure Analysis - Streamlit Web Interface

This module provides a user-friendly web interface that integrates all system
components for testing and demonstration of the delivery failure analysis POC.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import sys
import os
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(os.path.dirname(__file__))

from data_foundation import StreamingDataFoundation
from vector_database import VectorDatabase
from rag_engine import RAGEngine
from correlation_engine import CorrelationEngine
from advanced_processor import AdvancedQueryProcessor

# Page configuration
st.set_page_config(
    page_title="AI-Powered Delivery Failure Analysis",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .response-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'system_components' not in st.session_state:
    st.session_state.system_components = {}
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'current_response' not in st.session_state:
    st.session_state.current_response = None

@st.cache_resource
def initialize_system():
    """Initialize all system components."""
    try:
        # Initialize components
        foundation = StreamingDataFoundation()
        vector_db = VectorDatabase()
        rag_engine = RAGEngine(vector_db, foundation, provider="openrouter")
        correlation_engine = CorrelationEngine()
        advanced_processor = AdvancedQueryProcessor()
        
        # Load sample data
        with st.spinner("Loading sample data..."):
            for source in ['orders', 'fleet_logs', 'warehouse_logs', 'external_factors', 'feedback', 'clients', 'drivers', 'warehouses']:
                try:
                    foundation.load_data_streaming(source)
                except Exception as e:
                    st.warning(f"Warning loading {source}: {e}")
        
        return {
            'foundation': foundation,
            'vector_db': vector_db,
            'rag_engine': rag_engine,
            'correlation_engine': correlation_engine,
            'advanced_processor': advanced_processor
        }
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return None

def display_header():
    """Display the main header."""
    st.markdown('<h1 class="main-header">🚚 AI-Powered Delivery Failure Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent analysis of delivery failures with actionable insights")

def display_sidebar():
    """Display the sidebar with system status and controls."""
    with st.sidebar:
        st.header("🔧 System Status")
        
        # System status
        if st.session_state.system_components:
            st.success("✅ All systems operational")
            
            # Display component status
            components = st.session_state.system_components
            if 'rag_engine' in components:
                try:
                    status = components['rag_engine'].get_system_status()
                    st.metric("Vector DB Documents", status.get('total_documents', 0))
                    st.metric("LLM Available", "Yes" if status.get('llm_available', False) else "No")
                except:
                    st.warning("Status unavailable")
        else:
            st.error("❌ System not initialized")
        
        st.divider()
        
        # Query history
        st.header("📝 Query History")
        if st.session_state.query_history:
            for i, query in enumerate(reversed(st.session_state.query_history[-10:])):  # Show last 10
                if st.button(f"{i+1}. {query[:50]}...", key=f"history_{i}"):
                    st.session_state.current_query = query
                    st.rerun()
        else:
            st.info("No queries yet")
        
        st.divider()
        
        # Quick actions
        st.header("⚡ Quick Actions")
        if st.button("🔄 Refresh System"):
            st.cache_resource.clear()
            st.rerun()
        
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()
        
        if st.button("📊 System Stats"):
            display_system_stats()

def display_system_stats():
    """Display detailed system statistics."""
    if not st.session_state.system_components:
        st.error("System not initialized")
        return
    
    st.header("📊 System Statistics")
    
    try:
        components = st.session_state.system_components
        
        # Data foundation stats
        if 'foundation' in components:
            foundation = components['foundation']
            data_stats = foundation.get_data_statistics()
            
            st.subheader("📁 Data Sources")
            for source, stats in data_stats.get('sources', {}).items():
                st.metric(source.title(), stats.get('total_records', 0))
        
        # Vector database stats
        if 'vector_db' in components:
            vector_db = components['vector_db']
            vector_stats = vector_db.get_collection_stats()
            st.metric("Vector Documents", vector_stats.get('total_documents', 0))
        
        # RAG engine stats
        if 'rag_engine' in components:
            rag_engine = components['rag_engine']
            status = rag_engine.get_system_status()
            st.metric("LLM Model", status.get('model_name', 'N/A'))
            st.metric("Intent Patterns", len(status.get('intent_patterns', [])))
        
    except Exception as e:
        st.error(f"Error getting system stats: {e}")

def display_query_interface():
    """Display the main query interface."""
    st.header("💬 Ask Questions About Your Delivery Operations")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        value=st.session_state.current_query,
        height=100,
        placeholder="e.g., What are the main reasons for delivery failures? How can we improve performance?",
        help="Ask questions about delivery failures, performance, patterns, predictions, or recommendations."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔍 Analyze", type="primary"):
            if query.strip():
                process_query(query)
            else:
                st.warning("Please enter a question")
    
    with col2:
        if st.button("📋 Example Queries"):
            show_example_queries()
    
    with col3:
        st.info("💡 Try asking about patterns, predictions, or recommendations!")

def show_example_queries():
    """Show example queries."""
    example_queries = [
        "What are the main reasons for delivery failures?",
        "Which cities have the highest failure rates?",
        "How does weather affect delivery performance?",
        "What are the most common customer complaints?",
        "Predict the risk of delivery failures in the next month",
        "Compare performance across different warehouses",
        "Identify anomalies in our operations",
        "Recommend improvements for our delivery process"
    ]
    
    st.subheader("📋 Example Queries")
    for i, example in enumerate(example_queries):
        if st.button(f"{i+1}. {example}", key=f"example_{i}"):
            st.session_state.current_query = example
            st.rerun()

def process_query(query: str):
    """Process a query using the appropriate system component."""
    if not st.session_state.system_components:
        st.error("System not initialized. Please refresh the page.")
        return
    
    # Add to history
    if query not in st.session_state.query_history:
        st.session_state.query_history.append(query)
    
    st.session_state.current_query = query
    
    # Determine processing strategy
    processing_mode = st.selectbox(
        "Processing Mode:",
        ["Auto (Recommended)", "RAG Engine", "Advanced Processing", "Correlation Analysis"],
        index=0
    )
    
    with st.spinner("Analyzing your question..."):
        try:
            start_time = time.time()
            
            if processing_mode == "Auto (Recommended)" or processing_mode == "RAG Engine":
                # Use RAG engine for most queries
                result = process_with_rag(query)
            elif processing_mode == "Advanced Processing":
                # Use advanced processor for complex analysis
                result = process_with_advanced(query)
            elif processing_mode == "Correlation Analysis":
                # Use correlation engine for multi-source analysis
                result = process_with_correlation(query)
            
            processing_time = time.time() - start_time
            
            # Store response
            st.session_state.current_response = {
                'query': query,
                'result': result,
                'processing_time': processing_time,
                'mode': processing_mode,
                'timestamp': datetime.now()
            }
            
            # Display results
            display_response(result, processing_time)
            
        except Exception as e:
            st.error(f"Error processing query: {e}")
            st.exception(e)

def process_with_rag(query: str):
    """Process query using RAG engine."""
    rag_engine = st.session_state.system_components['rag_engine']
    result = rag_engine.query(query, n_context=5)
    
    return {
        'type': 'rag',
        'response': result.response,
        'sources': result.sources,
        'confidence': result.confidence,
        'tokens_used': result.tokens_used,
        'context_documents': len(result.context_documents)
    }

def process_with_advanced(query: str):
    """Process query using advanced processor."""
    advanced_processor = st.session_state.system_components['advanced_processor']
    result = advanced_processor.process_complex_query(query)
    
    return {
        'type': 'advanced',
        'response': result.get('analysis', {}),
        'patterns': result.get('patterns', []),
        'predictions': result.get('predictions', []),
        'recommendations': result.get('recommendations', []),
        'insights': result.get('insights', []),
        'confidence': result.get('confidence', 0.0)
    }

def process_with_correlation(query: str):
    """Process query using correlation engine."""
    correlation_engine = st.session_state.system_components['correlation_engine']
    
    # Simple correlation analysis
    if "city" in query.lower():
        geo_data = correlation_engine.geographic_correlation(city="City_1")
        return {
            'type': 'correlation',
            'response': f"Geographic analysis completed for {len(geo_data)} sources",
            'geo_data': geo_data,
            'confidence': 0.7
        }
    else:
        # General correlation analysis
        analysis = correlation_engine.cross_source_analysis({
            "order_ids": ["1", "2", "3"],
            "time_range": (datetime.now() - timedelta(days=30), datetime.now())
        })
        return {
            'type': 'correlation',
            'response': f"Cross-source analysis completed with {len(analysis)} result categories",
            'analysis': analysis,
            'confidence': 0.6
        }

def display_response(result: Dict[str, Any], processing_time: float):
    """Display the query response."""
    st.markdown('<div class="response-box">', unsafe_allow_html=True)
    
    # Response header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader("📊 Analysis Results")
    with col2:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    with col3:
        st.metric("Confidence", f"{result.get('confidence', 0.0):.2f}")
    
    # Main response
    if result['type'] == 'rag':
        display_rag_response(result)
    elif result['type'] == 'advanced':
        display_advanced_response(result)
    elif result['type'] == 'correlation':
        display_correlation_response(result)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_rag_response(result: Dict[str, Any]):
    """Display RAG engine response."""
    st.write(result['response'])
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sources", len(result.get('sources', [])))
    with col2:
        st.metric("Context Documents", result.get('context_documents', 0))
    with col3:
        st.metric("Tokens Used", result.get('tokens_used', 0))

def display_advanced_response(result: Dict[str, Any]):
    """Display advanced processor response."""
    # Patterns
    if result.get('patterns'):
        st.subheader("🔍 Detected Patterns")
        for i, pattern in enumerate(result['patterns'][:3]):  # Show first 3
            with st.expander(f"Pattern {i+1}: {pattern.pattern_type}"):
                st.write(f"**Description:** {pattern.description}")
                st.write(f"**Confidence:** {pattern.confidence:.2f}")
                st.write(f"**Severity:** {pattern.severity}")
                if pattern.recommendations:
                    st.write("**Recommendations:**")
                    for rec in pattern.recommendations:
                        st.write(f"• {rec}")
    
    # Predictions
    if result.get('predictions'):
        st.subheader("🔮 Predictions")
        for i, prediction in enumerate(result['predictions'][:3]):  # Show first 3
            with st.expander(f"Prediction {i+1}: {prediction.prediction_type}"):
                st.write(f"**Risk Level:** {prediction.risk_level}")
                st.write(f"**Probability:** {prediction.probability:.2f}")
                st.write(f"**Timeframe:** {prediction.timeframe}")
                if prediction.recommendations:
                    st.write("**Recommendations:**")
                    for rec in prediction.recommendations:
                        st.write(f"• {rec}")
    
    # Recommendations
    if result.get('recommendations'):
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(result['recommendations'][:5]):  # Show first 5
            with st.expander(f"Recommendation {i+1}: {rec.category}"):
                st.write(f"**Priority:** {rec.priority}")
                st.write(f"**Description:** {rec.description}")
                st.write(f"**Impact:** {rec.impact}")
                st.write(f"**Timeline:** {rec.timeline}")
                if rec.success_metrics:
                    st.write("**Success Metrics:**")
                    for metric in rec.success_metrics:
                        st.write(f"• {metric}")
    
    # Insights
    if result.get('insights'):
        st.subheader("🧠 Key Insights")
        for insight in result['insights']:
            st.write(f"• {insight}")

def display_correlation_response(result: Dict[str, Any]):
    """Display correlation engine response."""
    st.write(result['response'])
    
    if 'geo_data' in result:
        st.subheader("🌍 Geographic Data")
        for source, data in result['geo_data'].items():
            st.write(f"**{source.title()}:** {len(data)} records")
    
    if 'analysis' in result:
        st.subheader("📊 Analysis Results")
        analysis = result['analysis']
        for category, data in analysis.items():
            if isinstance(data, dict):
                st.write(f"**{category.title()}:** {len(data)} sub-categories")
            elif isinstance(data, list):
                st.write(f"**{category.title()}:** {len(data)} items")

def display_visualizations():
    """Display data visualizations."""
    st.header("📊 Data Visualizations")
    
    if not st.session_state.system_components:
        st.error("System not initialized")
        return
    
    try:
        components = st.session_state.system_components
        
        # Get sample data for visualization
        if 'foundation' in components:
            foundation = components['foundation']
            
            # Orders status distribution
            try:
                orders_df = foundation.get_data_from_db('orders', limit=1000)
                if not orders_df.empty and 'status' in orders_df.columns:
                    st.subheader("📦 Order Status Distribution")
                    status_counts = orders_df['status'].value_counts()
                    
                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title="Order Status Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # City-wise distribution
                if 'city' in orders_df.columns:
                    st.subheader("🏙️ Orders by City")
                    city_counts = orders_df['city'].value_counts().head(10)
                    
                    fig = px.bar(
                        x=city_counts.index,
                        y=city_counts.values,
                        title="Top 10 Cities by Order Count"
                    )
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not create visualizations: {e}")
    
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")

def display_about():
    """Display about information."""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### AI-Powered Delivery Failure Analysis POC
    
    This system provides intelligent analysis of delivery failures using advanced AI techniques:
    
    **🔧 Core Components:**
    - **Data Foundation**: Streaming data processing with SQLite storage
    - **Vector Database**: Semantic search using ChromaDB and sentence transformers
    - **RAG Engine**: Retrieval-Augmented Generation with OpenRouter/DeepSeek
    - **Correlation Engine**: Multi-source data correlation and analysis
    - **Advanced Processor**: ML-based pattern recognition and predictions
    
    **🚀 Key Features:**
    - Natural language query processing
    - Pattern recognition and anomaly detection
    - Predictive analysis and risk assessment
    - Actionable recommendations generation
    - Multi-dimensional data analysis
    - Real-time query processing
    
    **📊 Data Sources:**
    - Orders, Fleet Logs, Warehouse Logs
    - External Factors, Customer Feedback
    - Clients, Drivers, Warehouses
    
    **🎯 Use Cases:**
    - Root cause analysis of delivery failures
    - Performance optimization recommendations
    - Risk prediction and prevention
    - Cross-source data correlation
    - Operational insights and trends
    """)
    
    st.subheader("🔧 Technical Stack")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Backend:**
        - Python 3.10+
        - Pandas, NumPy, SciPy
        - SQLite, ChromaDB
        - Scikit-learn, NLTK
        """)
    
    with col2:
        st.markdown("""
        **AI/ML:**
        - Sentence Transformers
        - LangChain, OpenAI
        - OpenRouter API
        - Isolation Forest, DBSCAN
        """)

def main():
    """Main application function."""
    # Initialize system
    if not st.session_state.system_components:
        st.session_state.system_components = initialize_system()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Query Interface", "📊 Visualizations", "📈 System Stats", "ℹ️ About"])
    
    with tab1:
        display_query_interface()
        
        # Display current response if available
        if st.session_state.current_response:
            st.divider()
            st.subheader("📋 Last Response")
            display_response(
                st.session_state.current_response['result'],
                st.session_state.current_response['processing_time']
            )
    
    with tab2:
        display_visualizations()
    
    with tab3:
        display_system_stats()
    
    with tab4:
        display_about()

if __name__ == "__main__":
    main()
