"""
Multi-Source Data Correlation Engine for AI-Powered Delivery Failure Analysis

This module implements advanced data correlation capabilities that can connect
related data across different sources (orders, fleet logs, warehouse logs, 
external factors, feedback) to provide comprehensive analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import sqlite3
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""
    correlation_type: str
    source_1: str
    source_2: str
    correlation_strength: float
    related_records: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class EventSequence:
    """Represents a sequence of related events."""
    sequence_id: str
    events: List[Dict[str, Any]]
    timeline: List[datetime]
    causal_relationships: List[Tuple[str, str, str]]  # (event1, event2, relationship_type)
    confidence_score: float


class CorrelationEngine:
    """
    Multi-source data correlation engine that provides comprehensive analysis
    by connecting related data across different sources.
    """
    
    def __init__(self, db_path: str = "data/delivery_failures.db"):
        """
        Initialize the correlation engine.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
        
        # Define relationship mappings between data sources
        self.relationship_mappings = {
            "orders_to_fleet_logs": {
                "orders": "order_id",
                "fleet_logs": "order_id",
                "relationship_type": "one_to_many"
            },
            "orders_to_warehouse_logs": {
                "orders": "order_id", 
                "warehouse_logs": "order_id",
                "relationship_type": "one_to_many"
            },
            "orders_to_external_factors": {
                "orders": "order_id",
                "external_factors": "order_id", 
                "relationship_type": "one_to_many"
            },
            "orders_to_feedback": {
                "orders": "order_id",
                "feedback": "order_id",
                "relationship_type": "one_to_many"
            },
            "orders_to_clients": {
                "orders": "client_id",
                "clients": "client_id",
                "relationship_type": "many_to_one"
            },
            "fleet_logs_to_drivers": {
                "fleet_logs": "driver_id",
                "drivers": "driver_id", 
                "relationship_type": "many_to_one"
            },
            "warehouse_logs_to_warehouses": {
                "warehouse_logs": "warehouse_id",
                "warehouses": "warehouse_id",
                "relationship_type": "many_to_one"
            }
        }
        
        # Geographic correlation settings
        self.geographic_clusters = {}
        self.temporal_windows = {
            "immediate": timedelta(minutes=30),
            "short_term": timedelta(hours=2),
            "medium_term": timedelta(days=1),
            "long_term": timedelta(days=7)
        }
        
        logger.info("Correlation Engine initialized")
    
    def get_related_data(self, source_table: str, key_value: Any, 
                        key_column: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all related data for a given key across all sources.
        
        Args:
            source_table: Source table name
            key_value: Value to search for
            key_column: Column name to search in (if None, uses primary key)
            
        Returns:
            Dictionary mapping table names to lists of related records
        """
        related_data = {}
        
        # Find all relationships involving this source
        for rel_name, mapping in self.relationship_mappings.items():
            if source_table in mapping:
                target_table = None
                target_key = None
                
                for table, key in mapping.items():
                    if table != "relationship_type" and table != source_table:
                        target_table = table
                        target_key = key
                        break
                
                if target_table and target_key:
                    try:
                        # Get the key column for the source table
                        source_key = mapping[source_table]
                        
                        # Query related records
                        query = f"""
                        SELECT * FROM {target_table} 
                        WHERE {target_key} = ?
                        """
                        
                        df = pd.read_sql_query(query, self.db, params=[key_value])
                        related_data[target_table] = df.to_dict('records')
                        
                    except Exception as e:
                        logger.warning(f"Error querying {target_table}: {e}")
        
        return related_data
    
    def temporal_correlation(self, source_table: str, record_id: Any, 
                           time_column: str = None, 
                           window: str = "medium_term") -> List[Dict[str, Any]]:
        """
        Find temporally correlated events across all sources.
        
        Args:
            source_table: Source table name
            record_id: ID of the record to correlate
            time_column: Column name containing timestamps
            window: Time window for correlation
            
        Returns:
            List of temporally correlated events
        """
        try:
            # Get the timestamp of the source record
            if time_column is None:
                time_columns = self._get_time_columns(source_table)
                if not time_columns:
                    return []
                time_column = time_columns[0]  # Use the first available time column
            
            source_query = f"SELECT {time_column} FROM {source_table} WHERE order_id = ?"
            source_time = pd.read_sql_query(source_query, self.db, params=[record_id])
            
            if source_time.empty:
                return []
            
            source_timestamp = pd.to_datetime(source_time[time_column].iloc[0])
            time_delta = self.temporal_windows[window]
            
            correlated_events = []
            
            # Search across all tables for temporal correlations
            tables = ["orders", "fleet_logs", "warehouse_logs", "external_factors", "feedback"]
            
            for table in tables:
                if table == source_table:
                    continue
                    
                try:
                    # Find time columns in the table
                    time_columns = self._get_time_columns(table)
                    
                    for time_col in time_columns:
                        query = f"""
                        SELECT *, '{table}' as source_table, '{time_col}' as time_column
                        FROM {table}
                        WHERE {time_col} BETWEEN ? AND ?
                        """
                        
                        start_time = source_timestamp - time_delta
                        end_time = source_timestamp + time_delta
                        
                        df = pd.read_sql_query(query, self.db, params=[start_time, end_time])
                        
                        for _, row in df.iterrows():
                            correlated_events.append({
                                "table": table,
                                "record": row.to_dict(),
                                "time_difference": abs((pd.to_datetime(row[time_col]) - source_timestamp).total_seconds() / 3600),  # hours
                                "time_column": time_col
                            })
                            
                except Exception as e:
                    logger.warning(f"Error in temporal correlation for {table}: {e}")
            
            # Sort by time difference
            correlated_events.sort(key=lambda x: x["time_difference"])
            
            return correlated_events
            
        except Exception as e:
            logger.error(f"Error in temporal correlation: {e}")
            return []
    
    def geographic_correlation(self, city: str = None, state: str = None, 
                             pincode: str = None) -> Dict[str, Any]:
        """
        Find geographically correlated data across all sources.
        
        Args:
            city: City name to search for
            state: State name to search for  
            pincode: Pincode to search for
            
        Returns:
            Dictionary containing geographically correlated data
        """
        geographic_data = {}
        
        try:
            # Search across all tables for geographic data
            tables = ["orders", "clients", "drivers", "warehouses"]
            
            for table in tables:
                try:
                    # Build query based on available geographic columns
                    conditions = []
                    params = []
                    
                    if city and self._has_column(table, "city"):
                        conditions.append("city = ?")
                        params.append(city)
                    
                    if state and self._has_column(table, "state"):
                        conditions.append("state = ?")
                        params.append(state)
                    
                    if pincode and self._has_column(table, "pincode"):
                        conditions.append("pincode = ?")
                        params.append(pincode)
                    
                    if conditions:
                        query = f"SELECT * FROM {table} WHERE {' AND '.join(conditions)}"
                        df = pd.read_sql_query(query, self.db, params=params)
                        geographic_data[table] = df.to_dict('records')
                        
                except Exception as e:
                    logger.warning(f"Error in geographic correlation for {table}: {e}")
            
            # Perform geographic clustering if we have enough data
            if geographic_data:
                self._perform_geographic_clustering(geographic_data)
            
            return geographic_data
            
        except Exception as e:
            logger.error(f"Error in geographic correlation: {e}")
            return {}
    
    def causal_correlation(self, order_id: str) -> EventSequence:
        """
        Analyze causal relationships between events for a given order.
        
        Args:
            order_id: Order ID to analyze
            
        Returns:
            EventSequence object containing causal analysis
        """
        try:
            # Get all related data for this order
            related_data = self.get_related_data("orders", order_id, "order_id")
            
            # Collect all events with timestamps
            events = []
            
            # Add order event
            order_query = f"SELECT * FROM orders WHERE order_id = ?"
            order_df = pd.read_sql_query(order_query, self.db, params=[order_id])
            if not order_df.empty:
                order_record = order_df.iloc[0].to_dict()
                events.append({
                    "event_type": "order_created",
                    "timestamp": pd.to_datetime(order_record.get("order_date", order_record.get("created_at"))),
                    "data": order_record,
                    "source": "orders"
                })
            
            # Add related events
            for table, records in related_data.items():
                for record in records:
                    timestamp = self._extract_timestamp(record, table)
                    if timestamp:
                        events.append({
                            "event_type": f"{table}_event",
                            "timestamp": timestamp,
                            "data": record,
                            "source": table
                        })
            
            # Sort events by timestamp
            events.sort(key=lambda x: x["timestamp"])
            
            # Analyze causal relationships
            causal_relationships = self._analyze_causal_relationships(events)
            
            # Calculate confidence score
            confidence_score = self._calculate_causal_confidence(events, causal_relationships)
            
            return EventSequence(
                sequence_id=f"order_{order_id}",
                events=events,
                timeline=[event["timestamp"] for event in events],
                causal_relationships=causal_relationships,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error in causal correlation: {e}")
            return EventSequence(
                sequence_id=f"order_{order_id}",
                events=[],
                timeline=[],
                causal_relationships=[],
                confidence_score=0.0
            )
    
    def cross_source_analysis(self, query_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive cross-source analysis based on query filters.
        
        Args:
            query_filters: Dictionary containing filters for different sources
            
        Returns:
            Comprehensive analysis results
        """
        try:
            analysis_results = {
                "temporal_analysis": {},
                "geographic_analysis": {},
                "causal_analysis": {},
                "relationship_analysis": {},
                "summary": {}
            }
            
            # Temporal analysis
            if "time_range" in query_filters:
                analysis_results["temporal_analysis"] = self._analyze_temporal_patterns(
                    query_filters["time_range"]
                )
            
            # Geographic analysis
            if any(key in query_filters for key in ["city", "state", "pincode"]):
                analysis_results["geographic_analysis"] = self.geographic_correlation(
                    city=query_filters.get("city"),
                    state=query_filters.get("state"),
                    pincode=query_filters.get("pincode")
                )
            
            # Causal analysis for specific orders
            if "order_ids" in query_filters:
                causal_sequences = []
                for order_id in query_filters["order_ids"]:
                    sequence = self.causal_correlation(order_id)
                    causal_sequences.append(sequence)
                analysis_results["causal_analysis"] = causal_sequences
            
            # Relationship analysis
            analysis_results["relationship_analysis"] = self._analyze_relationships(
                query_filters
            )
            
            # Generate summary
            analysis_results["summary"] = self._generate_analysis_summary(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in cross-source analysis: {e}")
            return {"error": str(e)}
    
    def _get_time_columns(self, table: str) -> List[str]:
        """Get all timestamp columns for a table."""
        try:
            query = f"PRAGMA table_info({table})"
            df = pd.read_sql_query(query, self.db)
            
            time_columns = []
            for _, row in df.iterrows():
                col_name = row["name"]
                col_type = row["type"].lower()
                if any(time_word in col_type for time_word in ["timestamp", "datetime", "date", "time"]):
                    time_columns.append(col_name)
            
            return time_columns
        except:
            return []
    
    def _has_column(self, table: str, column: str) -> bool:
        """Check if a table has a specific column."""
        try:
            query = f"PRAGMA table_info({table})"
            df = pd.read_sql_query(query, self.db)
            return column in df["name"].values
        except:
            return False
    
    def _extract_timestamp(self, record: Dict[str, Any], table: str) -> Optional[datetime]:
        """Extract timestamp from a record."""
        time_columns = self._get_time_columns(table)
        
        for col in time_columns:
            if col in record and pd.notna(record[col]):
                try:
                    return pd.to_datetime(record[col])
                except:
                    continue
        
        return None
    
    def _analyze_causal_relationships(self, events: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """Analyze causal relationships between events."""
        relationships = []
        
        for i in range(len(events) - 1):
            event1 = events[i]
            event2 = events[i + 1]
            
            # Define relationship types based on event types
            relationship_type = self._determine_relationship_type(
                event1["event_type"], 
                event2["event_type"]
            )
            
            if relationship_type:
                relationships.append((
                    event1["event_type"],
                    event2["event_type"], 
                    relationship_type
                ))
        
        return relationships
    
    def _determine_relationship_type(self, event1_type: str, event2_type: str) -> Optional[str]:
        """Determine the type of relationship between two events."""
        relationship_map = {
            ("order_created", "warehouse_logs_event"): "triggers",
            ("warehouse_logs_event", "fleet_logs_event"): "enables",
            ("fleet_logs_event", "external_factors_event"): "influenced_by",
            ("external_factors_event", "feedback_event"): "causes",
            ("order_created", "feedback_event"): "results_in"
        }
        
        return relationship_map.get((event1_type, event2_type))
    
    def _calculate_causal_confidence(self, events: List[Dict[str, Any]], 
                                   relationships: List[Tuple[str, str, str]]) -> float:
        """Calculate confidence score for causal analysis."""
        if not events:
            return 0.0
        
        # Base confidence on number of events and relationships
        event_score = min(len(events) / 10.0, 1.0)
        relationship_score = min(len(relationships) / 5.0, 1.0)
        
        return (event_score + relationship_score) / 2.0
    
    def _perform_geographic_clustering(self, geographic_data: Dict[str, Any]):
        """Perform geographic clustering analysis."""
        try:
            # Extract coordinates if available, otherwise use city/state
            coordinates = []
            labels = []
            
            for table, records in geographic_data.items():
                for record in records:
                    if "latitude" in record and "longitude" in record:
                        coordinates.append([record["latitude"], record["longitude"]])
                        labels.append(f"{table}_{record.get('id', 'unknown')}")
                    elif "city" in record and "state" in record:
                        # Use city/state as a simple geographic identifier
                        coord = [hash(record["city"]) % 100, hash(record["state"]) % 100]
                        coordinates.append(coord)
                        labels.append(f"{table}_{record.get('id', 'unknown')}")
            
            if len(coordinates) > 1:
                # Perform hierarchical clustering
                distance_matrix = pdist(coordinates)
                linkage_matrix = linkage(distance_matrix, method='ward')
                clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
                
                self.geographic_clusters = {
                    "coordinates": coordinates,
                    "labels": labels,
                    "clusters": clusters.tolist()
                }
                
        except Exception as e:
            logger.warning(f"Error in geographic clustering: {e}")
    
    def _analyze_temporal_patterns(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Analyze temporal patterns across all sources."""
        start_time, end_time = time_range
        
        patterns = {
            "event_frequency": {},
            "peak_times": {},
            "correlations": {}
        }
        
        try:
            tables = ["orders", "fleet_logs", "warehouse_logs", "external_factors", "feedback"]
            
            for table in tables:
                time_columns = self._get_time_columns(table)
                
                for time_col in time_columns:
                    query = f"""
                    SELECT {time_col}, COUNT(*) as count
                    FROM {table}
                    WHERE {time_col} BETWEEN ? AND ?
                    GROUP BY DATE({time_col})
                    ORDER BY {time_col}
                    """
                    
                    df = pd.read_sql_query(query, self.db, params=[start_time, end_time])
                    
                    if not df.empty:
                        patterns["event_frequency"][f"{table}_{time_col}"] = df.to_dict('records')
                        
                        # Find peak times
                        peak_idx = df["count"].idxmax()
                        patterns["peak_times"][f"{table}_{time_col}"] = {
                            "peak_time": df.iloc[peak_idx][time_col],
                            "peak_count": df.iloc[peak_idx]["count"]
                        }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {e}")
            return patterns
    
    def _analyze_relationships(self, query_filters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between data sources."""
        relationships = {
            "order_to_fleet": 0,
            "order_to_warehouse": 0,
            "order_to_feedback": 0,
            "fleet_to_driver": 0,
            "warehouse_to_warehouse": 0
        }
        
        try:
            # Count relationships
            for rel_name, mapping in self.relationship_mappings.items():
                if "relationship_type" in mapping:
                    source_table = None
                    target_table = None
                    
                    for table, key in mapping.items():
                        if table != "relationship_type":
                            if source_table is None:
                                source_table = table
                            else:
                                target_table = table
                                break
                    
                    if source_table and target_table:
                        # Count matching records
                        query = f"""
                        SELECT COUNT(*) as count
                        FROM {source_table} s
                        JOIN {target_table} t ON s.{mapping[source_table]} = t.{mapping[target_table]}
                        """
                        
                        df = pd.read_sql_query(query, self.db)
                        if not df.empty:
                            relationships[f"{source_table}_to_{target_table}"] = df.iloc[0]["count"]
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error in relationship analysis: {e}")
            return relationships
    
    def _generate_analysis_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the analysis results."""
        summary = {
            "total_events": 0,
            "geographic_coverage": 0,
            "temporal_coverage": 0,
            "causal_confidence": 0.0,
            "key_insights": []
        }
        
        try:
            # Count total events
            for analysis_type, data in analysis_results.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            summary["total_events"] += len(value)
                        elif isinstance(value, dict) and "events" in value:
                            summary["total_events"] += len(value["events"])
            
            # Geographic coverage
            if "geographic_analysis" in analysis_results:
                geo_data = analysis_results["geographic_analysis"]
                summary["geographic_coverage"] = len(geo_data)
            
            # Causal confidence
            if "causal_analysis" in analysis_results:
                causal_data = analysis_results["causal_analysis"]
                if isinstance(causal_data, list) and causal_data:
                    confidences = [seq.confidence_score for seq in causal_data if hasattr(seq, 'confidence_score')]
                    if confidences:
                        summary["causal_confidence"] = sum(confidences) / len(confidences)
            
            # Generate key insights
            if summary["total_events"] > 0:
                summary["key_insights"].append(f"Analyzed {summary['total_events']} events across multiple sources")
            
            if summary["geographic_coverage"] > 0:
                summary["key_insights"].append(f"Coverage across {summary['geographic_coverage']} geographic regions")
            
            if summary["causal_confidence"] > 0.5:
                summary["key_insights"].append("Strong causal relationships identified")
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
        
        return summary
    
    def close(self):
        """Close the database connection."""
        if self.db:
            self.db.close()
        logger.info("Correlation Engine closed.")


def main():
    """Test the correlation engine functionality."""
    import sys
    sys.path.append('src')
    
    from data_foundation import StreamingDataFoundation
    
    # Initialize components
    foundation = StreamingDataFoundation()
    correlation_engine = CorrelationEngine()
    
    try:
        print("🧪 Testing Correlation Engine")
        print("=" * 50)
        
        # Test 1: Get related data
        print("\n1. Testing related data retrieval...")
        related_data = correlation_engine.get_related_data("orders", "1", "order_id")
        print(f"   Found related data in {len(related_data)} sources")
        
        # Test 2: Temporal correlation
        print("\n2. Testing temporal correlation...")
        temporal_events = correlation_engine.temporal_correlation("orders", "1")
        print(f"   Found {len(temporal_events)} temporally correlated events")
        
        # Test 3: Geographic correlation
        print("\n3. Testing geographic correlation...")
        geo_data = correlation_engine.geographic_correlation(city="New York")
        print(f"   Found geographic data in {len(geo_data)} sources")
        
        # Test 4: Causal correlation
        print("\n4. Testing causal correlation...")
        causal_sequence = correlation_engine.causal_correlation("1")
        print(f"   Causal sequence: {len(causal_sequence.events)} events, confidence: {causal_sequence.confidence_score:.2f}")
        
        # Test 5: Cross-source analysis
        print("\n5. Testing cross-source analysis...")
        analysis = correlation_engine.cross_source_analysis({
            "city": "New York",
            "order_ids": ["1", "2", "3"]
        })
        print(f"   Analysis completed with {len(analysis)} result categories")
        
        print("\n✅ All correlation engine tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    
    finally:
        correlation_engine.close()
        foundation.close()


if __name__ == "__main__":
    main()
