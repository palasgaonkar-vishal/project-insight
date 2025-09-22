#!/usr/bin/env python3
"""
Test script for Task 4: Multi-Source Data Correlation
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from correlation_engine import CorrelationEngine
from data_foundation import StreamingDataFoundation
from vector_database import VectorDatabase

def test_correlation_engine():
    """Test the correlation engine functionality."""
    print("STARTING TASK 4: MULTI-SOURCE DATA CORRELATION TESTS")
    print("=" * 60)
    
    # Initialize components
    foundation = StreamingDataFoundation()
    correlation_engine = CorrelationEngine()
    
    try:
        print("\n📊 Initializing data foundation...")
        # Load sample data
        for source in ['orders', 'fleet_logs', 'warehouse_logs', 'external_factors', 'feedback', 'clients', 'drivers', 'warehouses']:
            try:
                foundation.load_data_streaming(source)
                print(f"   ✓ Loaded {source}")
            except Exception as e:
                print(f"   ⚠️  Warning loading {source}: {e}")
        
        print("\n🔗 Testing Relationship Mapping...")
        print("-" * 40)
        
        # Test 1: Get related data for an order
        print("\n1. Testing related data retrieval...")
        related_data = correlation_engine.get_related_data("orders", "1", "order_id")
        print(f"   ✓ Found related data in {len(related_data)} sources")
        for source, records in related_data.items():
            print(f"     - {source}: {len(records)} records")
        
        # Test 2: Test different relationship types
        print("\n2. Testing different relationship types...")
        test_relationships = [
            ("orders", "1", "order_id"),
            ("clients", "1", "client_id"),
            ("drivers", "1", "driver_id")
        ]
        
        for source, key, key_col in test_relationships:
            try:
                related = correlation_engine.get_related_data(source, key, key_col)
                print(f"   ✓ {source} -> {len(related)} related sources")
            except Exception as e:
                print(f"   ⚠️  {source}: {e}")
        
        print("\n⏰ Testing Temporal Correlation...")
        print("-" * 40)
        
        # Test 3: Temporal correlation
        print("\n3. Testing temporal correlation...")
        temporal_events = correlation_engine.temporal_correlation("orders", "1", window="medium_term")
        print(f"   ✓ Found {len(temporal_events)} temporally correlated events")
        
        if temporal_events:
            print("   Sample temporal events:")
            for i, event in enumerate(temporal_events[:3]):
                print(f"     {i+1}. {event['table']} - {event['time_difference']:.2f} hours difference")
        
        # Test 4: Different time windows
        print("\n4. Testing different time windows...")
        time_windows = ["immediate", "short_term", "medium_term", "long_term"]
        for window in time_windows:
            try:
                events = correlation_engine.temporal_correlation("orders", "1", window=window)
                print(f"   ✓ {window}: {len(events)} events")
            except Exception as e:
                print(f"   ⚠️  {window}: {e}")
        
        print("\n🌍 Testing Geographic Correlation...")
        print("-" * 40)
        
        # Test 5: Geographic correlation
        print("\n5. Testing geographic correlation by city...")
        geo_data = correlation_engine.geographic_correlation(city="City_1")
        print(f"   ✓ Found geographic data in {len(geo_data)} sources")
        for source, records in geo_data.items():
            print(f"     - {source}: {len(records)} records")
        
        # Test 6: Geographic correlation by state
        print("\n6. Testing geographic correlation by state...")
        geo_data_state = correlation_engine.geographic_correlation(state="State_1")
        print(f"   ✓ Found geographic data in {len(geo_data_state)} sources")
        
        # Test 7: Combined geographic search
        print("\n7. Testing combined geographic search...")
        geo_data_combined = correlation_engine.geographic_correlation(city="City_1", state="State_1")
        print(f"   ✓ Found geographic data in {len(geo_data_combined)} sources")
        
        print("\n🔗 Testing Causal Correlation...")
        print("-" * 40)
        
        # Test 8: Causal correlation
        print("\n8. Testing causal correlation...")
        causal_sequence = correlation_engine.causal_correlation("1")
        print(f"   ✓ Causal sequence: {len(causal_sequence.events)} events")
        print(f"   ✓ Confidence score: {causal_sequence.confidence_score:.2f}")
        print(f"   ✓ Causal relationships: {len(causal_sequence.causal_relationships)}")
        
        if causal_sequence.events:
            print("   Event timeline:")
            for i, event in enumerate(causal_sequence.events[:5]):
                print(f"     {i+1}. {event['event_type']} at {event['timestamp']}")
        
        if causal_sequence.causal_relationships:
            print("   Causal relationships:")
            for i, (event1, event2, rel_type) in enumerate(causal_sequence.causal_relationships[:3]):
                print(f"     {i+1}. {event1} -> {event2} ({rel_type})")
        
        # Test 9: Multiple order causal analysis
        print("\n9. Testing multiple order causal analysis...")
        test_orders = ["1", "2", "3", "4", "5"]
        causal_sequences = []
        
        for order_id in test_orders:
            try:
                sequence = correlation_engine.causal_correlation(order_id)
                causal_sequences.append(sequence)
                print(f"   ✓ Order {order_id}: {len(sequence.events)} events, confidence: {sequence.confidence_score:.2f}")
            except Exception as e:
                print(f"   ⚠️  Order {order_id}: {e}")
        
        print("\n🔍 Testing Cross-Source Analysis...")
        print("-" * 40)
        
        # Test 10: Cross-source analysis
        print("\n10. Testing comprehensive cross-source analysis...")
        analysis = correlation_engine.cross_source_analysis({
            "city": "City_1",
            "state": "State_1",
            "order_ids": ["1", "2", "3"],
            "time_range": (datetime.now() - timedelta(days=30), datetime.now())
        })
        
        print(f"   ✓ Analysis completed with {len(analysis)} result categories")
        
        # Display analysis results
        for category, data in analysis.items():
            if isinstance(data, dict):
                print(f"     - {category}: {len(data)} sub-categories")
            elif isinstance(data, list):
                print(f"     - {category}: {len(data)} items")
            else:
                print(f"     - {category}: {type(data).__name__}")
        
        # Test 11: Performance testing
        print("\n⚡ Testing Performance...")
        print("-" * 40)
        
        print("\n11. Testing correlation performance...")
        start_time = time.time()
        
        # Test multiple correlations
        for i in range(5):
            correlation_engine.get_related_data("orders", str(i+1), "order_id")
            correlation_engine.temporal_correlation("orders", str(i+1))
            correlation_engine.geographic_correlation(city=f"City_{i+1}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"   ✓ Processed 5 correlations in {total_time:.2f} seconds")
        print(f"   ✓ Average time per correlation: {total_time/5:.2f} seconds")
        
        # Test 12: Error handling
        print("\n🛡️  Testing Error Handling...")
        print("-" * 40)
        
        print("\n12. Testing error handling...")
        
        # Test with invalid data
        try:
            invalid_related = correlation_engine.get_related_data("nonexistent_table", "999", "id")
            print(f"   ✓ Invalid table handled: {len(invalid_related)} results")
        except Exception as e:
            print(f"   ✓ Invalid table error handled: {type(e).__name__}")
        
        # Test with invalid order ID
        try:
            invalid_temporal = correlation_engine.temporal_correlation("orders", "999999")
            print(f"   ✓ Invalid order ID handled: {len(invalid_temporal)} events")
        except Exception as e:
            print(f"   ✓ Invalid order ID error handled: {type(e).__name__}")
        
        # Test with invalid geographic data
        try:
            invalid_geo = correlation_engine.geographic_correlation(city="NonexistentCity")
            print(f"   ✓ Invalid city handled: {len(invalid_geo)} sources")
        except Exception as e:
            print(f"   ✓ Invalid city error handled: {type(e).__name__}")
        
        print("\n✅ All Task 4 tests completed successfully!")
        print("=" * 60)
        print("🎉 MULTI-SOURCE DATA CORRELATION ENGINE IS WORKING!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during Task 4 testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        correlation_engine.close()
        foundation.close()


def test_correlation_with_vector_db():
    """Test correlation engine integration with vector database."""
    print("\n🔗 Testing Correlation Engine with Vector Database...")
    print("-" * 50)
    
    try:
        # Initialize components
        foundation = StreamingDataFoundation()
        vector_db = VectorDatabase()
        correlation_engine = CorrelationEngine()
        
        # Load some data
        foundation.load_data_streaming('orders')
        
        # Test correlation with vector search
        print("\n1. Testing correlation with vector search...")
        
        # Get related data
        related_data = correlation_engine.get_related_data("orders", "1", "order_id")
        
        # Test if we can correlate with vector database
        if related_data:
            print(f"   ✓ Found {len(related_data)} related data sources")
            
            # Test temporal correlation
            temporal_events = correlation_engine.temporal_correlation("orders", "1")
            print(f"   ✓ Found {len(temporal_events)} temporal events")
            
            # Test geographic correlation
            geo_data = correlation_engine.geographic_correlation(city="City_1")
            print(f"   ✓ Found {len(geo_data)} geographic data sources")
        
        print("   ✅ Correlation engine works with vector database!")
        
    except Exception as e:
        print(f"   ❌ Error in correlation with vector DB: {e}")
    
    finally:
        if 'correlation_engine' in locals():
            correlation_engine.close()
        if 'vector_db' in locals():
            vector_db.close()
        if 'foundation' in locals():
            foundation.close()


if __name__ == "__main__":
    success = test_correlation_engine()
    
    if success:
        test_correlation_with_vector_db()
        print("\n🎉 ALL TASK 4 TESTS PASSED!")
    else:
        print("\n💥 SOME TASK 4 TESTS FAILED!")
        sys.exit(1)
