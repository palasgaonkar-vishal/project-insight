"""
Test script for Task 2: Vector Database Setup

This script tests the vector database functionality including:
- Document vectorization and storage
- Semantic search capabilities
- Metadata filtering
- Collection statistics
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vector_database import VectorDatabase
from data_foundation import StreamingDataFoundation


def test_vector_database_basic():
    """Test basic vector database functionality."""
    print("=" * 60)
    print("TESTING BASIC VECTOR DATABASE FUNCTIONALITY")
    print("=" * 60)
    
    # Initialize vector database
    vector_db = VectorDatabase()
    
    try:
        # Test 1: Add sample documents
        print("\n1. Testing document addition...")
        sample_data = pd.DataFrame({
            'order_id': [1, 2, 3, 4, 5],
            'client_id': [101, 102, 103, 104, 105],
            'status': ['delivered', 'failed', 'delayed', 'delivered', 'failed'],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
            'state': ['NY', 'CA', 'IL', 'TX', 'AZ'],
            'failure_reason': ['', 'Weather delay', 'Traffic congestion', '', 'Driver issue']
        })
        
        result = vector_db.add_documents(sample_data, "orders", "test_chunk_1")
        print(f"✓ Added {result['documents_added']} documents: {result['status']}")
        
        # Test 2: Add more documents from different source
        print("\n2. Testing multiple data sources...")
        fleet_data = pd.DataFrame({
            'driver_id': [201, 202, 203],
            'vehicle_id': ['V001', 'V002', 'V003'],
            'status': ['active', 'maintenance', 'active'],
            'location': ['Warehouse A', 'Service Center', 'Warehouse B']
        })
        
        result = vector_db.add_documents(fleet_data, "fleet_logs", "test_chunk_2")
        print(f"✓ Added {result['documents_added']} fleet documents: {result['status']}")
        
        # Test 3: Get collection statistics
        print("\n3. Testing collection statistics...")
        stats = vector_db.get_collection_stats()
        print(f"✓ Total documents: {stats['total_documents']}")
        print(f"✓ Sources: {stats['sources']}")
        
        # Test 4: Basic search
        print("\n4. Testing basic search...")
        search_result = vector_db.search("delivery failure", n_results=3)
        print(f"✓ Found {search_result['total_results']} results for 'delivery failure'")
        for i, result in enumerate(search_result['results'][:2]):
            print(f"  Result {i+1}: {result['document'][:100]}...")
        
        # Test 5: Source-specific search
        print("\n5. Testing source-specific search...")
        orders_search = vector_db.search("failed orders", n_results=3, source_filter="orders")
        print(f"✓ Found {orders_search['total_results']} order results for 'failed orders'")
        
        # Test 6: Metadata filtering
        print("\n6. Testing metadata filtering...")
        failed_search = vector_db.search("delivery issues", n_results=5, 
                                       metadata_filter={"status": "failed"})
        print(f"✓ Found {failed_search['total_results']} failed delivery results")
        
        print("\n✅ All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Error in basic tests: {str(e)}")
        raise
    finally:
        vector_db.close()


def test_vector_database_with_real_data():
    """Test vector database with real sample data."""
    print("\n" + "=" * 60)
    print("TESTING WITH REAL SAMPLE DATA")
    print("=" * 60)
    
    # Initialize both data foundation and vector database
    foundation = StreamingDataFoundation(chunk_size=1000)
    vector_db = VectorDatabase()
    
    try:
        # Load real data and vectorize it
        print("\n1. Loading and vectorizing real data...")
        
        # Process each data source
        data_sources = ['orders', 'fleet_logs', 'warehouse_logs', 'external_factors', 'feedback']
        total_docs = 0
        
        for source in data_sources:
            print(f"\nProcessing {source}...")
            
            # Load data from foundation
            data_generator = foundation.load_data_streaming(source)
            
            chunk_count = 0
            for chunk in data_generator:
                chunk_count += 1
                chunk_id = f"{source}_chunk_{chunk_count}"
                
                # Add to vector database
                result = vector_db.add_documents(chunk, source, chunk_id)
                if result['status'] == 'success':
                    total_docs += result['documents_added']
                    print(f"  ✓ Added {result['documents_added']} documents from {chunk_id}")
                else:
                    print(f"  ❌ Error adding {chunk_id}: {result.get('error', 'Unknown error')}")
        
        print(f"\n✓ Total documents vectorized: {total_docs}")
        
        # Test search with real data
        print("\n2. Testing search with real data...")
        
        test_queries = [
            "delivery failures in New York",
            "weather related delays",
            "driver performance issues",
            "warehouse dispatch problems",
            "customer complaints about late delivery"
        ]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            result = vector_db.search(query, n_results=3)
            print(f"  Found {result['total_results']} results")
            
            for i, res in enumerate(result['results'][:2]):
                print(f"    {i+1}. {res['metadata']['source']}: {res['document'][:80]}...")
        
        # Get final statistics
        print("\n3. Final collection statistics...")
        stats = vector_db.get_collection_stats()
        print(f"✓ Total documents: {stats['total_documents']}")
        print(f"✓ Sources breakdown: {json.dumps(stats['sources'], indent=2)}")
        
        print("\n✅ Real data tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in real data tests: {str(e)}")
        raise
    finally:
        foundation.close()
        vector_db.close()


def test_performance():
    """Test vector database performance."""
    print("\n" + "=" * 60)
    print("TESTING VECTOR DATABASE PERFORMANCE")
    print("=" * 60)
    
    vector_db = VectorDatabase()
    
    try:
        # Clear collection first
        vector_db.clear_collection()
        
        # Test with different chunk sizes
        chunk_sizes = [100, 500, 1000]
        
        for chunk_size in chunk_sizes:
            print(f"\nTesting with chunk size: {chunk_size}")
            
            # Create test data
            test_data = pd.DataFrame({
                'order_id': range(1, chunk_size + 1),
                'client_id': [101 + (i % 50) for i in range(chunk_size)],
                'status': ['delivered' if i % 3 == 0 else 'failed' if i % 3 == 1 else 'delayed' 
                          for i in range(chunk_size)],
                'city': [f'City_{i % 20}' for i in range(chunk_size)],
                'failure_reason': [f'Reason_{i % 10}' if i % 3 != 0 else '' 
                                 for i in range(chunk_size)]
            })
            
            # Time the vectorization
            start_time = datetime.now()
            result = vector_db.add_documents(test_data, "orders", f"perf_test_{chunk_size}")
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            docs_per_second = result['documents_added'] / duration if duration > 0 else 0
            
            print(f"  ✓ Added {result['documents_added']} documents in {duration:.2f}s")
            print(f"  ✓ Rate: {docs_per_second:.1f} docs/second")
            
            # Test search performance
            search_start = datetime.now()
            search_result = vector_db.search("delivery failure", n_results=10)
            search_end = datetime.now()
            
            search_duration = (search_end - search_start).total_seconds()
            print(f"  ✓ Search completed in {search_duration:.3f}s")
        
        print("\n✅ Performance tests completed!")
        
    except Exception as e:
        print(f"❌ Error in performance tests: {str(e)}")
        raise
    finally:
        vector_db.close()


def main():
    """Run all tests."""
    print("STARTING TASK 2: VECTOR DATABASE TESTS")
    print("=" * 60)
    
    try:
        # Run basic functionality tests
        test_vector_database_basic()
        
        # Run real data tests
        test_vector_database_with_real_data()
        
        # Run performance tests
        test_performance()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TASK 2 TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TASK 2 TESTS FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
