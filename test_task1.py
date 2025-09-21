#!/usr/bin/env python3
"""
Test script for Task 1: Streaming-First Data Foundation
"""

import sys
import os
sys.path.append('src')

from data_foundation import StreamingDataFoundation
import pandas as pd
import time

def test_data_foundation():
    """Test the streaming data foundation implementation."""
    print("🧪 Testing Task 1: Streaming-First Data Foundation")
    print("=" * 60)
    
    # Initialize foundation
    foundation = StreamingDataFoundation(chunk_size=500)  # Smaller chunks for testing
    
    try:
        # Test 1: Load a single data source
        print("\n📊 Test 1: Loading single data source (orders)")
        print("-" * 40)
        
        start_time = time.time()
        chunk_count = 0
        total_records = 0
        
        for chunk in foundation.load_data_streaming('orders'):
            chunk_count += 1
            total_records += len(chunk)
            print(f"   Chunk {chunk_count}: {len(chunk)} records")
            
            # Test chunk validation
            assert 'order_id' in chunk.columns, "Primary key missing"
            assert chunk['order_id'].notna().all(), "Null primary keys found"
            
            if chunk_count >= 3:  # Test first 3 chunks only
                break
        
        end_time = time.time()
        print(f"   ✅ Processed {chunk_count} chunks, {total_records} records in {end_time - start_time:.2f}s")
        
        # Test 2: Database storage and retrieval
        print("\n📊 Test 2: Database storage and retrieval")
        print("-" * 40)
        
        # Get data from database
        db_chunk = foundation.get_data_chunk('orders', offset=0, limit=100)
        print(f"   Retrieved {len(db_chunk)} records from database")
        print(f"   Columns: {list(db_chunk.columns)}")
        print(f"   Sample order_id: {db_chunk['order_id'].iloc[0] if len(db_chunk) > 0 else 'N/A'}")
        
        # Test 3: Data statistics
        print("\n📊 Test 3: Data statistics")
        print("-" * 40)
        
        stats = foundation.get_data_statistics()
        for source_name, stat in stats.items():
            if 'error' not in stat:
                print(f"   {source_name}: {stat['record_count']} records")
            else:
                print(f"   {source_name}: ERROR - {stat['error']}")
        
        # Test 4: Memory usage monitoring
        print("\n📊 Test 4: Memory usage monitoring")
        print("-" * 40)
        
        memory_usage = foundation.memory_usage
        if memory_usage:
            print(f"   Memory usage range: {min(memory_usage):.2f} - {max(memory_usage):.2f} MB")
            print(f"   Memory usage trend: {'Stable' if max(memory_usage) - min(memory_usage) < 100 else 'Variable'}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        foundation.close()

def test_performance():
    """Test performance with different chunk sizes."""
    print("\n🚀 Performance Test: Different chunk sizes")
    print("=" * 60)
    
    chunk_sizes = [100, 500, 1000]
    
    for chunk_size in chunk_sizes:
        print(f"\n📊 Testing chunk size: {chunk_size}")
        print("-" * 30)
        
        foundation = StreamingDataFoundation(chunk_size=chunk_size)
        
        try:
            # Clear database before each test
            foundation.clear_database()
            
            start_time = time.time()
            chunk_count = 0
            total_records = 0
            
            for chunk in foundation.load_data_streaming('orders'):
                chunk_count += 1
                total_records += len(chunk)
                
                if chunk_count >= 5:  # Test first 5 chunks only
                    break
            
            end_time = time.time()
            duration = end_time - start_time
            records_per_second = total_records / duration if duration > 0 else 0
            
            print(f"   Chunks processed: {chunk_count}")
            print(f"   Total records: {total_records}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Records/second: {records_per_second:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        finally:
            foundation.close()

if __name__ == "__main__":
    # Run basic tests
    success = test_data_foundation()
    
    if success:
        # Run performance tests
        test_performance()
        print("\n🎉 Task 1 implementation is working correctly!")
    else:
        print("\n💥 Task 1 implementation has issues that need to be fixed.")
        sys.exit(1)
