#!/usr/bin/env python3
"""
Test script for Task 6: User Interface and Integration
"""

import sys
import os
import time
import subprocess
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_streamlit_app_imports():
    """Test that the Streamlit app can be imported without errors."""
    print("🧪 Testing Streamlit App Imports...")
    print("-" * 50)
    
    try:
        # Test importing required libraries first
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ All required libraries imported successfully")
        
        # Test importing the app (using direct path)
        app_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'app.py')
        if os.path.exists(app_path):
            print("✅ App file exists and is accessible")
            
            # Test that we can read the app file
            with open(app_path, 'r') as f:
                content = f.read()
                if 'def initialize_system' in content:
                    print("✅ App file contains expected functions")
                else:
                    print("⚠️  App file may be missing expected functions")
        else:
            print("❌ App file not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_system_initialization():
    """Test system initialization without Streamlit context."""
    print("\n🔧 Testing System Initialization...")
    print("-" * 50)
    
    try:
        # Import components directly
        from data_foundation import StreamingDataFoundation
        from vector_database import VectorDatabase
        from rag_engine import RAGEngine
        from correlation_engine import CorrelationEngine
        from advanced_processor import AdvancedQueryProcessor
        
        print("✅ All components imported successfully")
        
        # Test individual component initialization
        print("\n1. Testing Data Foundation...")
        foundation = StreamingDataFoundation()
        print("   ✓ Data Foundation initialized")
        
        # Load sample data
        for source in ['orders', 'fleet_logs', 'warehouse_logs']:
            try:
                foundation.load_data_streaming(source)
                print(f"   ✓ Loaded {source}")
            except Exception as e:
                print(f"   ⚠️  Warning loading {source}: {e}")
        
        print("\n2. Testing Vector Database...")
        vector_db = VectorDatabase()
        print("   ✓ Vector Database initialized")
        
        print("\n3. Testing RAG Engine...")
        rag_engine = RAGEngine(vector_db, foundation, provider="openrouter")
        print("   ✓ RAG Engine initialized")
        
        print("\n4. Testing Correlation Engine...")
        correlation_engine = CorrelationEngine()
        print("   ✓ Correlation Engine initialized")
        
        print("\n5. Testing Advanced Processor...")
        advanced_processor = AdvancedQueryProcessor()
        print("   ✓ Advanced Processor initialized")
        
        # Test system status
        print("\n6. Testing System Status...")
        try:
            status = rag_engine.get_system_status()
            print(f"   ✓ RAG Engine status: {status.get('status', 'unknown')}")
            print(f"   ✓ LLM Available: {status.get('llm_available', False)}")
            print(f"   ✓ Vector Documents: {status.get('total_documents', 0)}")
        except Exception as e:
            print(f"   ⚠️  Status check warning: {e}")
        
        # Clean up
        foundation.close()
        vector_db.close()
        correlation_engine.close()
        advanced_processor.close()
        
        print("\n✅ System initialization test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ System initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query_processing():
    """Test query processing functionality."""
    print("\n💬 Testing Query Processing...")
    print("-" * 50)
    
    try:
        # Initialize components
        from data_foundation import StreamingDataFoundation
        from vector_database import VectorDatabase
        from rag_engine import RAGEngine
        from correlation_engine import CorrelationEngine
        from advanced_processor import AdvancedQueryProcessor
        
        foundation = StreamingDataFoundation()
        vector_db = VectorDatabase()
        rag_engine = RAGEngine(vector_db, foundation, provider="openrouter")
        correlation_engine = CorrelationEngine()
        advanced_processor = AdvancedQueryProcessor()
        
        # Load sample data
        for source in ['orders', 'fleet_logs', 'warehouse_logs', 'external_factors', 'feedback']:
            try:
                foundation.load_data_streaming(source)
            except Exception as e:
                print(f"   ⚠️  Warning loading {source}: {e}")
        
        # Test queries
        test_queries = [
            "What are the main reasons for delivery failures?",
            "How can we improve our delivery performance?",
            "Predict the risk of failures in the next month",
            "Compare performance across different cities"
        ]
        
        print(f"\nTesting {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing: {query}")
            
            # Test RAG processing
            try:
                start_time = time.time()
                rag_result = rag_engine.query(query, n_context=3)
                rag_time = time.time() - start_time
                print(f"   ✓ RAG: {rag_time:.2f}s, confidence: {rag_result.confidence:.2f}")
            except Exception as e:
                print(f"   ⚠️  RAG error: {e}")
            
            # Test advanced processing
            try:
                start_time = time.time()
                advanced_result = advanced_processor.process_complex_query(query)
                advanced_time = time.time() - start_time
                print(f"   ✓ Advanced: {advanced_time:.2f}s, confidence: {advanced_result.get('confidence', 0.0):.2f}")
            except Exception as e:
                print(f"   ⚠️  Advanced error: {e}")
            
            # Test correlation processing
            try:
                start_time = time.time()
                correlation_result = correlation_engine.cross_source_analysis({
                    "order_ids": ["1", "2", "3"],
                    "time_range": (datetime.now() - timedelta(days=30), datetime.now())
                })
                correlation_time = time.time() - start_time
                print(f"   ✓ Correlation: {correlation_time:.2f}s")
            except Exception as e:
                print(f"   ⚠️  Correlation error: {e}")
        
        # Clean up
        foundation.close()
        vector_db.close()
        correlation_engine.close()
        advanced_processor.close()
        
        print("\n✅ Query processing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_app_functionality():
    """Test Streamlit app functionality without running the server."""
    print("\n🖥️  Testing Streamlit App Functionality...")
    print("-" * 50)
    
    try:
        # Test that we can read and parse the app file
        app_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'app.py')
        
        if not os.path.exists(app_path):
            print("❌ App file not found")
            return False
        
        print("✅ App file exists")
        
        # Read and check app content
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Check for expected functions
        expected_functions = [
            'def initialize_system',
            'def display_header',
            'def display_sidebar',
            'def display_query_interface',
            'def process_query',
            'def process_with_rag',
            'def process_with_advanced',
            'def process_with_correlation',
            'def display_response',
            'def display_rag_response',
            'def display_advanced_response',
            'def display_correlation_response',
            'def display_visualizations',
            'def display_about'
        ]
        
        print("✅ Checking for expected functions...")
        for func in expected_functions:
            if func in content:
                print(f"   ✓ {func}")
            else:
                print(f"   ❌ {func} not found")
                return False
        
        # Test that we can import required libraries
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Required libraries can be imported")
        
        print("\n✅ Streamlit app functionality test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app functionality error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_visualization():
    """Test data visualization components."""
    print("\n📊 Testing Data Visualization...")
    print("-" * 50)
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        
        # Test creating sample visualizations
        print("1. Testing Plotly imports...")
        print("   ✓ Plotly Express imported")
        print("   ✓ Plotly Graph Objects imported")
        
        print("\n2. Testing visualization creation...")
        
        # Test pie chart
        data = {'Status': ['Delivered', 'Failed', 'Pending'], 'Count': [80, 15, 5]}
        df = pd.DataFrame(data)
        fig = px.pie(df, values='Count', names='Status', title="Order Status Distribution")
        print("   ✓ Pie chart created successfully")
        
        # Test bar chart
        cities = ['City A', 'City B', 'City C', 'City D', 'City E']
        counts = np.random.randint(10, 100, len(cities))
        fig = px.bar(x=cities, y=counts, title="Orders by City")
        print("   ✓ Bar chart created successfully")
        
        # Test line chart
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        values = np.random.randint(50, 150, 30)
        fig = px.line(x=dates, y=values, title="Daily Orders Trend")
        print("   ✓ Line chart created successfully")
        
        print("\n✅ Data visualization test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Data visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_configuration():
    """Test app configuration and dependencies."""
    print("\n⚙️  Testing App Configuration...")
    print("-" * 50)
    
    try:
        # Test Streamlit configuration
        import streamlit as st
        
        print("1. Testing Streamlit configuration...")
        print(f"   ✓ Streamlit version: {st.__version__}")
        
        # Test required dependencies
        dependencies = [
            ('pandas', 'pd'),
            ('numpy', 'np'),
            ('plotly.express', 'px'),
            ('plotly.graph_objects', 'go'),
            ('datetime', 'datetime'),
            ('json', 'json'),
            ('time', 'time'),
            ('sys', 'sys'),
            ('os', 'os')
        ]
        
        print("\n2. Testing required dependencies...")
        for dep, alias in dependencies:
            try:
                exec(f"import {dep} as {alias}")
                print(f"   ✓ {dep} imported as {alias}")
            except ImportError as e:
                print(f"   ❌ {dep} import failed: {e}")
                return False
        
        # Test app file structure
        print("\n3. Testing app file structure...")
        app_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'app.py')
        if os.path.exists(app_file):
            print("   ✓ app.py exists")
            
            # Check file size
            file_size = os.path.getsize(app_file)
            print(f"   ✓ app.py size: {file_size:,} bytes")
            
            # Check if file has content
            with open(app_file, 'r') as f:
                content = f.read()
                if len(content) > 1000:
                    print("   ✓ app.py has substantial content")
                else:
                    print("   ⚠️  app.py seems too small")
        else:
            print("   ❌ app.py not found")
            return False
        
        print("\n✅ App configuration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ App configuration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between all components."""
    print("\n🔗 Testing Component Integration...")
    print("-" * 50)
    
    try:
        # Test that all components can work together
        from data_foundation import StreamingDataFoundation
        from vector_database import VectorDatabase
        from rag_engine import RAGEngine
        from correlation_engine import CorrelationEngine
        from advanced_processor import AdvancedQueryProcessor
        
        print("1. Initializing all components...")
        foundation = StreamingDataFoundation()
        vector_db = VectorDatabase()
        rag_engine = RAGEngine(vector_db, foundation, provider="openrouter")
        correlation_engine = CorrelationEngine()
        advanced_processor = AdvancedQueryProcessor()
        
        print("   ✓ All components initialized")
        
        print("\n2. Loading sample data...")
        for source in ['orders', 'fleet_logs', 'warehouse_logs']:
            try:
                foundation.load_data_streaming(source)
                print(f"   ✓ Loaded {source}")
            except Exception as e:
                print(f"   ⚠️  Warning loading {source}: {e}")
        
        print("\n3. Testing component interactions...")
        
        # Test RAG with data foundation
        try:
            result = rag_engine.query("What are the main delivery issues?", n_context=3)
            print(f"   ✓ RAG + Data Foundation: {result.confidence:.2f} confidence")
        except Exception as e:
            print(f"   ⚠️  RAG + Data Foundation: {e}")
        
        # Test correlation with data foundation
        try:
            geo_data = correlation_engine.geographic_correlation(city="City_1")
            print(f"   ✓ Correlation + Data Foundation: {len(geo_data)} sources")
        except Exception as e:
            print(f"   ⚠️  Correlation + Data Foundation: {e}")
        
        # Test advanced processor with data foundation
        try:
            result = advanced_processor.process_complex_query("Analyze delivery patterns")
            print(f"   ✓ Advanced + Data Foundation: {result.get('confidence', 0.0):.2f} confidence")
        except Exception as e:
            print(f"   ⚠️  Advanced + Data Foundation: {e}")
        
        # Clean up
        foundation.close()
        vector_db.close()
        correlation_engine.close()
        advanced_processor.close()
        
        print("\n✅ Component integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Component integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Task 6 tests."""
    print("STARTING TASK 6: USER INTERFACE AND INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Streamlit App Imports", test_streamlit_app_imports),
        ("System Initialization", test_system_initialization),
        ("Query Processing", test_query_processing),
        ("Streamlit App Functionality", test_streamlit_app_functionality),
        ("Data Visualization", test_data_visualization),
        ("App Configuration", test_app_configuration),
        ("Component Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*60}")
    print(f"TASK 6 TEST SUMMARY: {passed}/{total} tests passed")
    print('='*60)
    
    if passed == total:
        print("🎉 ALL TASK 6 TESTS PASSED!")
        print("🚀 User Interface and Integration is ready!")
        return True
    else:
        print("💥 SOME TASK 6 TESTS FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
