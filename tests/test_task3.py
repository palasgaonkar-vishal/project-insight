"""
Test script for Task 3: Basic RAG Engine

This script tests the RAG engine functionality including:
- Query processing and intent classification
- Context retrieval from vector database
- LLM response generation
- System status and error handling
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_engine import RAGEngine
from vector_database import VectorDatabase
from data_foundation import StreamingDataFoundation


def test_rag_engine_basic():
    """Test basic RAG engine functionality."""
    print("=" * 60)
    print("TESTING BASIC RAG ENGINE FUNCTIONALITY")
    print("=" * 60)
    
    # Initialize components
    vector_db = VectorDatabase()
    data_foundation = StreamingDataFoundation()
    rag_engine = RAGEngine(vector_db, data_foundation)
    
    try:
        # Test 1: System status
        print("\n1. Testing system status...")
        status = rag_engine.get_system_status()
        print(f"✓ System status: {status['status']}")
        print(f"✓ LLM available: {status['llm_available']}")
        print(f"✓ Total documents: {status['total_documents']}")
        print(f"✓ Data sources: {list(status['data_sources'].keys())}")
        
        # Test 2: Query suggestions
        print("\n2. Testing query suggestions...")
        suggestions = rag_engine.get_query_suggestions()
        print(f"✓ Generated {len(suggestions)} query suggestions")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"  {i}. {suggestion}")
        
        # Test 3: Intent classification
        print("\n3. Testing intent classification...")
        test_queries = [
            "How many orders failed?",
            "Why did deliveries fail?",
            "Which cities have problems?",
            "Compare warehouse performance",
            "Predict future failures"
        ]
        
        for query in test_queries:
            intent = rag_engine._classify_query_intent(query)
            print(f"  '{query}' -> {intent}")
        
        print("\n✅ All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Error in basic tests: {str(e)}")
        raise
    finally:
        vector_db.close()
        data_foundation.close()


def test_rag_engine_without_llm():
    """Test RAG engine without LLM (API key not available)."""
    print("\n" + "=" * 60)
    print("TESTING RAG ENGINE WITHOUT LLM")
    print("=" * 60)
    
    # Initialize components
    vector_db = VectorDatabase()
    data_foundation = StreamingDataFoundation()
    
    # Initialize RAG engine without API key
    rag_engine = RAGEngine(vector_db, data_foundation, openai_api_key=None)
    
    try:
        print("\n1. Testing without LLM...")
        status = rag_engine.get_system_status()
        print(f"✓ LLM available: {status['llm_available']}")
        
        # Test query processing (should work but with limited response)
        print("\n2. Testing query processing without LLM...")
        result = rag_engine.query("How many orders failed?")
        print(f"✓ Query processed: {result.query}")
        print(f"✓ Response: {result.response}")
        print(f"✓ Context documents: {len(result.context_documents)}")
        print(f"✓ Sources: {result.sources}")
        print(f"✓ Processing time: {result.processing_time:.2f}s")
        
        print("\n✅ RAG engine works without LLM (with limited functionality)")
        
    except Exception as e:
        print(f"❌ Error in no-LLM tests: {str(e)}")
        raise
    finally:
        vector_db.close()
        data_foundation.close()


def test_rag_engine_with_llm():
    """Test RAG engine with LLM (if API key is available)."""
    print("\n" + "=" * 60)
    print("TESTING RAG ENGINE WITH LLM")
    print("=" * 60)
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OpenAI API key found. Skipping LLM tests.")
        print("   Set OPENAI_API_KEY environment variable to test with LLM.")
        return
    
    # Initialize components
    vector_db = VectorDatabase()
    data_foundation = StreamingDataFoundation()
    rag_engine = RAGEngine(vector_db, data_foundation, openai_api_key=api_key)
    
    try:
        print("\n1. Testing with LLM...")
        status = rag_engine.get_system_status()
        print(f"✓ LLM available: {status['llm_available']}")
        print(f"✓ Model: {status['model_name']}")
        
        # Test different types of queries
        test_queries = [
            "How many orders failed?",
            "What are the main reasons for delivery failures?",
            "Which cities have the highest failure rates?",
            "How does weather affect delivery performance?",
            "What are the most common customer complaints?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing query: '{query}'")
            result = rag_engine.query(query, n_context=5)
            
            print(f"   Response: {result.response[:200]}...")
            print(f"   Sources: {result.sources}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Processing time: {result.processing_time:.2f}s")
            print(f"   Tokens used: {result.tokens_used}")
        
        print("\n✅ All LLM tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in LLM tests: {str(e)}")
        raise
    finally:
        vector_db.close()
        data_foundation.close()


def test_rag_engine_performance():
    """Test RAG engine performance."""
    print("\n" + "=" * 60)
    print("TESTING RAG ENGINE PERFORMANCE")
    print("=" * 60)
    
    # Initialize components
    vector_db = VectorDatabase()
    data_foundation = StreamingDataFoundation()
    rag_engine = RAGEngine(vector_db, data_foundation)
    
    try:
        # Test with different context sizes
        context_sizes = [5, 10, 15]
        query = "What are the main delivery problems?"
        
        for n_context in context_sizes:
            print(f"\nTesting with {n_context} context documents...")
            
            start_time = datetime.now()
            result = rag_engine.query(query, n_context=n_context)
            end_time = datetime.now()
            
            total_time = (end_time - start_time).total_seconds()
            
            print(f"  ✓ Context documents: {len(result.context_documents)}")
            print(f"  ✓ Processing time: {result.processing_time:.2f}s")
            print(f"  ✓ Total time: {total_time:.2f}s")
            print(f"  ✓ Sources: {len(result.sources)}")
            print(f"  ✓ Confidence: {result.confidence:.2f}")
        
        print("\n✅ Performance tests completed!")
        
    except Exception as e:
        print(f"❌ Error in performance tests: {str(e)}")
        raise
    finally:
        vector_db.close()
        data_foundation.close()


def test_rag_engine_error_handling():
    """Test RAG engine error handling."""
    print("\n" + "=" * 60)
    print("TESTING RAG ENGINE ERROR HANDLING")
    print("=" * 60)
    
    # Initialize components
    vector_db = VectorDatabase()
    data_foundation = StreamingDataFoundation()
    rag_engine = RAGEngine(vector_db, data_foundation)
    
    try:
        # Test with empty query
        print("\n1. Testing empty query...")
        result = rag_engine.query("")
        print(f"✓ Empty query handled: {result.response[:100]}...")
        
        # Test with very long query
        print("\n2. Testing very long query...")
        long_query = "What are the delivery problems? " * 50
        result = rag_engine.query(long_query)
        print(f"✓ Long query handled: {len(result.response)} chars")
        
        # Test with special characters
        print("\n3. Testing special characters...")
        special_query = "What's the problem with delivery @#$%^&*()?"
        result = rag_engine.query(special_query)
        print(f"✓ Special characters handled: {result.response[:100]}...")
        
        print("\n✅ Error handling tests completed!")
        
    except Exception as e:
        print(f"❌ Error in error handling tests: {str(e)}")
        raise
    finally:
        vector_db.close()
        data_foundation.close()


def main():
    """Run all RAG engine tests."""
    print("STARTING TASK 3: RAG ENGINE TESTS")
    print("=" * 60)
    
    try:
        # Run basic functionality tests
        test_rag_engine_basic()
        
        # Run tests without LLM
        test_rag_engine_without_llm()
        
        # Run tests with LLM (if available)
        test_rag_engine_with_llm()
        
        # Run performance tests
        test_rag_engine_performance()
        
        # Run error handling tests
        test_rag_engine_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TASK 3 TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TASK 3 TESTS FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
