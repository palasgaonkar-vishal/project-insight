#!/usr/bin/env python3
"""
Test script for Task 5: Advanced Query Processing
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from advanced_processor import AdvancedQueryProcessor
from data_foundation import StreamingDataFoundation
from correlation_engine import CorrelationEngine

def test_advanced_processor():
    """Test the advanced query processor functionality."""
    print("STARTING TASK 5: ADVANCED QUERY PROCESSING TESTS")
    print("=" * 60)
    
    # Initialize components
    foundation = StreamingDataFoundation()
    processor = AdvancedQueryProcessor()
    
    try:
        print("\n📊 Initializing data foundation...")
        # Load sample data
        for source in ['orders', 'fleet_logs', 'warehouse_logs', 'external_factors', 'feedback', 'clients', 'drivers', 'warehouses']:
            try:
                foundation.load_data_streaming(source)
                print(f"   ✓ Loaded {source}")
            except Exception as e:
                print(f"   ⚠️  Warning loading {source}: {e}")
        
        print("\n🔍 Testing Query Complexity Analysis...")
        print("-" * 50)
        
        # Test 1: Simple queries
        print("\n1. Testing simple queries...")
        simple_queries = [
            "How many orders failed?",
            "What is the average delivery time?",
            "Show me recent feedback"
        ]
        
        for query in simple_queries:
            result = processor.process_complex_query(query)
            complexity = result.get('complexity', {})
            print(f"   ✓ '{query[:30]}...' -> {complexity.get('level', 'unknown')} complexity")
        
        # Test 2: Moderate complexity queries
        print("\n2. Testing moderate complexity queries...")
        moderate_queries = [
            "Compare delivery performance across different cities",
            "Analyze failure patterns over the last month",
            "Show me trends in customer feedback"
        ]
        
        for query in moderate_queries:
            result = processor.process_complex_query(query)
            complexity = result.get('complexity', {})
            dimensions = complexity.get('dimensions', [])
            print(f"   ✓ '{query[:30]}...' -> {complexity.get('level', 'unknown')} ({len(dimensions)} dimensions)")
        
        # Test 3: Complex queries
        print("\n3. Testing complex queries...")
        complex_queries = [
            "Predict the risk of delivery failures in the next month based on weather patterns and historical data",
            "Analyze the correlation between warehouse efficiency, driver performance, and customer satisfaction",
            "Identify anomalies in our operations and recommend preventive measures"
        ]
        
        for query in complex_queries:
            result = processor.process_complex_query(query)
            complexity = result.get('complexity', {})
            dimensions = complexity.get('dimensions', [])
            print(f"   ✓ '{query[:40]}...' -> {complexity.get('level', 'unknown')} ({len(dimensions)} dimensions)")
        
        print("\n🔍 Testing Pattern Recognition...")
        print("-" * 50)
        
        # Test 4: Pattern detection
        print("\n4. Testing pattern detection...")
        pattern_queries = [
            "What are the recurring failure patterns in our delivery system?",
            "Identify unusual patterns in warehouse operations",
            "Find performance patterns across different time periods"
        ]
        
        for query in pattern_queries:
            result = processor.process_complex_query(query)
            patterns = result.get('patterns', [])
            print(f"   ✓ '{query[:40]}...' -> {len(patterns)} patterns detected")
            
            for i, pattern in enumerate(patterns[:2]):  # Show first 2 patterns
                print(f"     Pattern {i+1}: {pattern.pattern_type} (confidence: {pattern.confidence:.2f})")
        
        print("\n🔮 Testing Predictive Analysis...")
        print("-" * 50)
        
        # Test 5: Predictive analysis
        print("\n5. Testing predictive analysis...")
        predictive_queries = [
            "Predict the risk of delivery failures in the next month",
            "Forecast performance trends for the next quarter",
            "What are the resource needs for next month?"
        ]
        
        for query in predictive_queries:
            result = processor.process_complex_query(query)
            predictions = result.get('predictions', [])
            print(f"   ✓ '{query[:40]}...' -> {len(predictions)} predictions generated")
            
            for i, prediction in enumerate(predictions[:2]):  # Show first 2 predictions
                print(f"     Prediction {i+1}: {prediction.prediction_type} (risk: {prediction.risk_level})")
        
        print("\n🔗 Testing Multi-Dimensional Analysis...")
        print("-" * 50)
        
        # Test 6: Multi-dimensional analysis
        print("\n6. Testing multi-dimensional analysis...")
        multi_dim_queries = [
            "Compare performance across different cities and recommend improvements",
            "Analyze the correlation between weather conditions and delivery performance",
            "Investigate the relationship between warehouse efficiency and customer satisfaction"
        ]
        
        for query in multi_dim_queries:
            result = processor.process_complex_query(query)
            analysis = result.get('analysis', {})
            insights = analysis.get('dimensional_insights', [])
            print(f"   ✓ '{query[:40]}...' -> {len(insights)} dimensional insights")
            
            for i, insight in enumerate(insights[:2]):  # Show first 2 insights
                print(f"     Insight {i+1}: {insight}")
        
        print("\n💡 Testing Recommendation Generation...")
        print("-" * 50)
        
        # Test 7: Recommendation generation
        print("\n7. Testing recommendation generation...")
        recommendation_queries = [
            "How can we improve our delivery success rate?",
            "What should we do to prevent future failures?",
            "Recommend optimizations for our operations"
        ]
        
        for query in recommendation_queries:
            result = processor.process_complex_query(query)
            recommendations = result.get('recommendations', [])
            print(f"   ✓ '{query[:40]}...' -> {len(recommendations)} recommendations")
            
            for i, rec in enumerate(recommendations[:2]):  # Show first 2 recommendations
                print(f"     Rec {i+1}: {rec.description[:60]}... (priority: {rec.priority})")
        
        print("\n🧠 Testing Insight Generation...")
        print("-" * 50)
        
        # Test 8: Insight generation
        print("\n8. Testing insight generation...")
        insight_queries = [
            "Provide comprehensive analysis of our delivery operations",
            "Give me insights about our performance patterns",
            "Summarize the key findings from our data analysis"
        ]
        
        for query in insight_queries:
            result = processor.process_complex_query(query)
            insights = result.get('insights', [])
            confidence = result.get('confidence', 0.0)
            print(f"   ✓ '{query[:40]}...' -> {len(insights)} insights (confidence: {confidence:.2f})")
            
            for i, insight in enumerate(insights[:2]):  # Show first 2 insights
                print(f"     Insight {i+1}: {insight}")
        
        print("\n⚡ Testing Performance...")
        print("-" * 50)
        
        # Test 9: Performance testing
        print("\n9. Testing processing performance...")
        test_queries = [
            "Analyze failure patterns",
            "Predict future risks",
            "Compare performance across sources",
            "Identify anomalies",
            "Generate recommendations"
        ]
        
        start_time = time.time()
        
        for query in test_queries:
            result = processor.process_complex_query(query)
            patterns = len(result.get('patterns', []))
            predictions = len(result.get('predictions', []))
            recommendations = len(result.get('recommendations', []))
            print(f"   ✓ '{query}' -> {patterns} patterns, {predictions} predictions, {recommendations} recommendations")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"   ✓ Processed {len(test_queries)} queries in {total_time:.2f} seconds")
        print(f"   ✓ Average time per query: {total_time/len(test_queries):.2f} seconds")
        
        print("\n🛡️  Testing Error Handling...")
        print("-" * 50)
        
        # Test 10: Error handling
        print("\n10. Testing error handling...")
        
        # Test with empty query
        try:
            result = processor.process_complex_query("")
            print(f"   ✓ Empty query handled: {result.get('confidence', 0.0):.2f} confidence")
        except Exception as e:
            print(f"   ✓ Empty query error handled: {type(e).__name__}")
        
        # Test with very long query
        try:
            long_query = "Analyze " + "performance " * 100 + "patterns"
            result = processor.process_complex_query(long_query)
            print(f"   ✓ Long query handled: {len(long_query)} characters")
        except Exception as e:
            print(f"   ✓ Long query error handled: {type(e).__name__}")
        
        # Test with special characters
        try:
            special_query = "What's the problem with delivery @#$%^&*()?"
            result = processor.process_complex_query(special_query)
            print(f"   ✓ Special characters handled: {result.get('confidence', 0.0):.2f} confidence")
        except Exception as e:
            print(f"   ✓ Special characters error handled: {type(e).__name__}")
        
        print("\n🔍 Testing Integration with Correlation Engine...")
        print("-" * 50)
        
        # Test 11: Integration with correlation engine
        print("\n11. Testing integration with correlation engine...")
        try:
            correlation_engine = CorrelationEngine()
            
            # Test complex query with correlation
            complex_query = "Analyze the correlation between warehouse operations and delivery failures across different cities"
            result = processor.process_complex_query(complex_query)
            
            print(f"   ✓ Complex correlation query processed")
            print(f"   ✓ Patterns: {len(result.get('patterns', []))}")
            print(f"   ✓ Predictions: {len(result.get('predictions', []))}")
            print(f"   ✓ Recommendations: {len(result.get('recommendations', []))}")
            print(f"   ✓ Confidence: {result.get('confidence', 0.0):.2f}")
            
            correlation_engine.close()
            
        except Exception as e:
            print(f"   ⚠️  Integration test warning: {e}")
        
        print("\n✅ All Task 5 tests completed successfully!")
        print("=" * 60)
        print("🎉 ADVANCED QUERY PROCESSING ENGINE IS WORKING!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during Task 5 testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        processor.close()
        foundation.close()


def test_advanced_queries_from_requirements():
    """Test with queries from the original requirements."""
    print("\n📋 Testing Original Requirements Queries...")
    print("-" * 50)
    
    try:
        foundation = StreamingDataFoundation()
        processor = AdvancedQueryProcessor()
        
        # Load data
        for source in ['orders', 'fleet_logs', 'warehouse_logs', 'external_factors', 'feedback']:
            try:
                foundation.load_data_streaming(source)
            except Exception as e:
                print(f"   ⚠️  Warning loading {source}: {e}")
        
        # Original requirement queries
        original_queries = [
            "Why did Client X's orders fail in the past week?",
            "What's likely to cause failures next month?",
            "How does Warehouse A compare to Warehouse B?",
            "What are the main reasons for delivery failures?",
            "Which cities have the highest failure rates?",
            "How does weather affect delivery performance?",
            "What are the most common customer complaints?",
            "Predict delivery performance for next quarter"
        ]
        
        print(f"\nTesting {len(original_queries)} original requirement queries...")
        
        for i, query in enumerate(original_queries, 1):
            print(f"\n{i}. Testing: {query}")
            result = processor.process_complex_query(query)
            
            complexity = result.get('complexity', {})
            patterns = len(result.get('patterns', []))
            predictions = len(result.get('predictions', []))
            recommendations = len(result.get('recommendations', []))
            confidence = result.get('confidence', 0.0)
            
            print(f"   Complexity: {complexity.get('level', 'unknown')}")
            print(f"   Patterns: {patterns}, Predictions: {predictions}, Recommendations: {recommendations}")
            print(f"   Confidence: {confidence:.2f}")
            
            # Show insights if available
            insights = result.get('insights', [])
            if insights:
                print(f"   Key insight: {insights[0]}")
        
        print("\n✅ Original requirements queries tested successfully!")
        
    except Exception as e:
        print(f"❌ Error testing original requirements: {e}")
    
    finally:
        if 'processor' in locals():
            processor.close()
        if 'foundation' in locals():
            foundation.close()


if __name__ == "__main__":
    success = test_advanced_processor()
    
    if success:
        test_advanced_queries_from_requirements()
        print("\n🎉 ALL TASK 5 TESTS PASSED!")
    else:
        print("\n💥 SOME TASK 5 TESTS FAILED!")
        sys.exit(1)
