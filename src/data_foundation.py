"""
Streaming-First Data Foundation for AI-Powered Delivery Failure Analysis POC

This module implements a production-ready data ingestion pipeline using streaming
processing that can handle large datasets efficiently while maintaining memory
efficiency and scalability.
"""

import os
import sqlite3
import pandas as pd
import psutil
from typing import Dict, List, Generator, Optional, Any
from datetime import datetime
import logging
from functools import lru_cache
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingDataFoundation:
    """
    Streaming-first data foundation that processes CSV files in chunks
    and stores them in SQLite with proper indexing for efficient querying.
    """
    
    def __init__(self, 
                 chunk_size: Optional[int] = None, 
                 db_path: Optional[str] = None):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        self.db_path = db_path or os.getenv("DATABASE_PATH", "data/delivery_failures.db")
        self.data_sources = {}
        self.memory_usage = []
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self.db = self._setup_database()
        
        # Data source configurations
        self.data_dir = os.path.abspath(os.getenv("DATA_DIR", "sample-data-set"))
        self.source_configs = {
            'orders': {
                'file': 'orders.csv',
                'table': 'orders',
                'primary_key': 'order_id',
                'indexes': ['client_id', 'city', 'state', 'order_date', 'status']
            },
            'fleet_logs': {
                'file': 'fleet_logs.csv',
                'table': 'fleet_logs',
                'primary_key': 'fleet_log_id',
                'indexes': ['order_id', 'driver_id', 'departure_time']
            },
            'warehouse_logs': {
                'file': 'warehouse_logs.csv',
                'table': 'warehouse_logs',
                'primary_key': 'log_id',
                'indexes': ['order_id', 'warehouse_id', 'picking_start']
            },
            'external_factors': {
                'file': 'external_factors.csv',
                'table': 'external_factors',
                'primary_key': 'factor_id',
                'indexes': ['order_id', 'recorded_at', 'weather_condition']
            },
            'feedback': {
                'file': 'feedback.csv',
                'table': 'feedback',
                'primary_key': 'feedback_id',
                'indexes': ['order_id', 'sentiment', 'rating', 'created_at']
            },
            'clients': {
                'file': 'clients.csv',
                'table': 'clients',
                'primary_key': 'client_id',
                'indexes': ['city', 'state', 'created_at']
            },
            'drivers': {
                'file': 'drivers.csv',
                'table': 'drivers',
                'primary_key': 'driver_id',
                'indexes': ['city', 'state', 'status']
            },
            'warehouses': {
                'file': 'warehouses.csv',
                'table': 'warehouses',
                'primary_key': 'warehouse_id',
                'indexes': ['city', 'state', 'capacity']
            }
        }
    
    def _setup_database(self) -> sqlite3.Connection:
        """Set up SQLite database with optimized schema and indexing."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        
        # Create tables with optimized schemas
        self._create_tables(conn)
        return conn
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables with proper schema and indexing."""
        
        # Orders table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                client_id INTEGER,
                customer_name TEXT,
                customer_phone TEXT,
                delivery_address_line1 TEXT,
                delivery_address_line2 TEXT,
                city TEXT,
                state TEXT,
                pincode TEXT,
                order_date TIMESTAMP,
                promised_delivery_date TIMESTAMP,
                actual_delivery_date TIMESTAMP,
                status TEXT,
                payment_mode TEXT,
                amount REAL,
                failure_reason TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Fleet logs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fleet_logs (
                fleet_log_id INTEGER PRIMARY KEY,
                order_id INTEGER,
                driver_id INTEGER,
                vehicle_number TEXT,
                route_code TEXT,
                gps_delay_notes TEXT,
                departure_time TIMESTAMP,
                arrival_time TIMESTAMP,
                created_at TIMESTAMP
            )
        """)
        
        # Warehouse logs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS warehouse_logs (
                log_id INTEGER PRIMARY KEY,
                order_id INTEGER,
                warehouse_id INTEGER,
                picking_start TIMESTAMP,
                picking_end TIMESTAMP,
                dispatch_time TIMESTAMP,
                notes TEXT
            )
        """)
        
        # External factors table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS external_factors (
                factor_id INTEGER PRIMARY KEY,
                order_id INTEGER,
                traffic_condition TEXT,
                weather_condition TEXT,
                event_type TEXT,
                recorded_at TIMESTAMP
            )
        """)
        
        # Feedback table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY,
                order_id INTEGER,
                customer_name TEXT,
                feedback_text TEXT,
                sentiment TEXT,
                rating INTEGER,
                created_at TIMESTAMP
            )
        """)
        
        # Clients table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                client_id INTEGER PRIMARY KEY,
                client_name TEXT,
                gst_number TEXT,
                contact_person TEXT,
                contact_phone TEXT,
                contact_email TEXT,
                address_line1 TEXT,
                address_line2 TEXT,
                city TEXT,
                state TEXT,
                pincode TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Drivers table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                driver_id INTEGER PRIMARY KEY,
                driver_name TEXT,
                phone TEXT,
                license_number TEXT,
                partner_company TEXT,
                city TEXT,
                state TEXT,
                status TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Warehouses table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS warehouses (
                warehouse_id INTEGER PRIMARY KEY,
                warehouse_name TEXT,
                state TEXT,
                city TEXT,
                pincode TEXT,
                capacity INTEGER,
                manager_name TEXT,
                contact_phone TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        self._create_indexes(conn)
        conn.commit()
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """Create database indexes for optimal query performance."""
        
        indexes = [
            # Orders indexes
            "CREATE INDEX IF NOT EXISTS idx_orders_client_id ON orders(client_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_city_state ON orders(city, state)",
            "CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date)",
            "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_orders_failure_reason ON orders(failure_reason)",
            
            # Fleet logs indexes
            "CREATE INDEX IF NOT EXISTS idx_fleet_logs_order_id ON fleet_logs(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_fleet_logs_driver_id ON fleet_logs(driver_id)",
            "CREATE INDEX IF NOT EXISTS idx_fleet_logs_departure_time ON fleet_logs(departure_time)",
            
            # Warehouse logs indexes
            "CREATE INDEX IF NOT EXISTS idx_warehouse_logs_order_id ON warehouse_logs(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_warehouse_logs_warehouse_id ON warehouse_logs(warehouse_id)",
            "CREATE INDEX IF NOT EXISTS idx_warehouse_logs_picking_start ON warehouse_logs(picking_start)",
            
            # External factors indexes
            "CREATE INDEX IF NOT EXISTS idx_external_factors_order_id ON external_factors(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_external_factors_recorded_at ON external_factors(recorded_at)",
            "CREATE INDEX IF NOT EXISTS idx_external_factors_weather ON external_factors(weather_condition)",
            
            # Feedback indexes
            "CREATE INDEX IF NOT EXISTS idx_feedback_order_id ON feedback(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_sentiment ON feedback(sentiment)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at)",
            
            # Clients indexes
            "CREATE INDEX IF NOT EXISTS idx_clients_city_state ON clients(city, state)",
            "CREATE INDEX IF NOT EXISTS idx_clients_created_at ON clients(created_at)",
            
            # Drivers indexes
            "CREATE INDEX IF NOT EXISTS idx_drivers_city_state ON drivers(city, state)",
            "CREATE INDEX IF NOT EXISTS idx_drivers_status ON drivers(status)",
            
            # Warehouses indexes
            "CREATE INDEX IF NOT EXISTS idx_warehouses_city_state ON warehouses(city, state)",
            "CREATE INDEX IF NOT EXISTS idx_warehouses_capacity ON warehouses(capacity)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
    
    def _monitor_memory(self):
        """Monitor memory usage and log it."""
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / (1024 * 1024)
        self.memory_usage.append(memory_mb)
        logger.info(f"Memory usage: {memory_mb:.2f} MB")
        return memory_mb
    
    def load_data_streaming(self, source_name: str) -> Generator[pd.DataFrame, None, None]:
        """
        Load data from CSV file using streaming processing.
        
        Args:
            source_name: Name of the data source to load
            
        Yields:
            pd.DataFrame: Chunks of data as they are processed
        """
        if source_name not in self.source_configs:
            raise ValueError(f"Unknown data source: {source_name}")
        
        config = self.source_configs[source_name]
        file_path = os.path.join(self.data_dir, config['file'])
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading {source_name} from {file_path} with chunk size {self.chunk_size}")
        
        # Monitor initial memory
        initial_memory = self._monitor_memory()
        
        try:
            # Process data in chunks
            for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=self.chunk_size)):
                logger.info(f"Processing chunk {chunk_num + 1} for {source_name}")
                
                # Validate chunk
                validated_chunk = self._validate_chunk(chunk, source_name)
                
                # Store chunk in database (first chunk replaces, others append)
                if_exists_mode = 'replace' if chunk_num == 0 else 'append'
                self._store_chunk(validated_chunk, source_name, if_exists=if_exists_mode)
                
                # Monitor memory usage
                current_memory = self._monitor_memory()
                
                # Yield chunk for further processing
                yield validated_chunk
                
                # Log progress
                if (chunk_num + 1) % 10 == 0:
                    logger.info(f"Processed {chunk_num + 1} chunks for {source_name}")
            
            logger.info(f"Successfully loaded {source_name}")
            
        except Exception as e:
            logger.error(f"Error loading {source_name}: {str(e)}")
            raise
    
    def _validate_chunk(self, chunk: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Validate a chunk of data for required fields and data types.
        
        Args:
            chunk: DataFrame chunk to validate
            source_name: Name of the data source
            
        Returns:
            pd.DataFrame: Validated chunk
        """
        config = self.source_configs[source_name]
        primary_key = config['primary_key']
        
        # Check if primary key exists and is not null
        if primary_key not in chunk.columns:
            raise ValueError(f"Primary key {primary_key} not found in {source_name}")
        
        if chunk[primary_key].isnull().any():
            logger.warning(f"Null values found in primary key {primary_key} for {source_name}")
        
        # Remove duplicates based on primary key
        chunk = chunk.drop_duplicates(subset=[primary_key], keep='first')
        
        # Convert timestamp columns
        timestamp_columns = [col for col in chunk.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in timestamp_columns:
            if col in chunk.columns:
                chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
        
        return chunk
    
    def _store_chunk(self, chunk: pd.DataFrame, source_name: str, if_exists: str = 'append'):
        """
        Store a chunk of data in the database.
        
        Args:
            chunk: DataFrame chunk to store
            source_name: Name of the data source
            if_exists: How to behave if the table exists ('replace', 'append', 'fail')
        """
        config = self.source_configs[source_name]
        table_name = config['table']
        
        try:
            chunk.to_sql(table_name, self.db, if_exists=if_exists, index=False)
            logger.debug(f"Stored {len(chunk)} records in {table_name}")
        except Exception as e:
            logger.error(f"Error storing chunk in {table_name}: {str(e)}")
            raise
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all data sources using streaming processing.
        
        Returns:
            Dict containing statistics for each data source
        """
        logger.info("Starting data loading process for all sources")
        start_time = datetime.now()
        
        results = {}
        
        for source_name in self.source_configs.keys():
            logger.info(f"Loading {source_name}...")
            source_start_time = datetime.now()
            
            try:
                chunk_count = 0
                total_records = 0
                
                for chunk in self.load_data_streaming(source_name):
                    chunk_count += 1
                    total_records += len(chunk)
                
                source_end_time = datetime.now()
                duration = (source_end_time - source_start_time).total_seconds()
                
                results[source_name] = {
                    'status': 'success',
                    'chunks_processed': chunk_count,
                    'total_records': total_records,
                    'duration_seconds': duration,
                    'records_per_second': total_records / duration if duration > 0 else 0
                }
                
                logger.info(f"✓ {source_name}: {total_records} records in {duration:.2f}s")
                
            except Exception as e:
                results[source_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"✗ {source_name}: {str(e)}")
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Get final memory usage
        final_memory = self._monitor_memory()
        
        results['summary'] = {
            'total_duration_seconds': total_duration,
            'final_memory_mb': final_memory,
            'sources_loaded': len([r for r in results.values() if r.get('status') == 'success']),
            'sources_failed': len([r for r in results.values() if r.get('status') == 'error'])
        }
        
        logger.info(f"Data loading completed in {total_duration:.2f}s")
        logger.info(f"Final memory usage: {final_memory:.2f} MB")
        
        return results
    
    def get_data_chunk(self, source_name: str, offset: int = 0, limit: int = 1000) -> pd.DataFrame:
        """
        Retrieve data in chunks from the database.
        
        Args:
            source_name: Name of the data source
            offset: Starting position
            limit: Maximum number of records to return
            
        Returns:
            pd.DataFrame: Chunk of data
        """
        if source_name not in self.source_configs:
            raise ValueError(f"Unknown data source: {source_name}")
        
        config = self.source_configs[source_name]
        table_name = config['table']
        
        query = f"SELECT * FROM {table_name} LIMIT {limit} OFFSET {offset}"
        return pd.read_sql_query(query, self.db)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all loaded data sources.
        
        Returns:
            Dict containing statistics for each data source
        """
        stats = {}
        
        for source_name, config in self.source_configs.items():
            table_name = config['table']
            
            try:
                # Get record count
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_result = self.db.execute(count_query).fetchone()
                record_count = count_result[0] if count_result else 0
                
                # Get date range if applicable
                date_columns = [col for col in ['order_date', 'created_at', 'departure_time', 'picking_start', 'recorded_at'] 
                              if col in config.get('indexes', [])]
                
                date_range = {}
                for date_col in date_columns:
                    try:
                        min_query = f"SELECT MIN({date_col}) as min_date FROM {table_name}"
                        max_query = f"SELECT MAX({date_col}) as max_date FROM {table_name}"
                        
                        min_result = self.db.execute(min_query).fetchone()
                        max_result = self.db.execute(max_query).fetchone()
                        
                        if min_result and max_result:
                            date_range[date_col] = {
                                'min': min_result[0],
                                'max': max_result[0]
                            }
                    except:
                        pass
                
                stats[source_name] = {
                    'record_count': record_count,
                    'date_range': date_range
                }
                
            except Exception as e:
                stats[source_name] = {
                    'error': str(e)
                }
        
        return stats
    
    def clear_database(self):
        """Clear all data from the database (useful for testing)."""
        if self.db:
            for source_name, config in self.source_configs.items():
                table_name = config['table']
                self.db.execute(f"DELETE FROM {table_name}")
            self.db.commit()
            logger.info("Database cleared")
    
    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()
            logger.info("Database connection closed")


def main():
    """Main function to test the data foundation."""
    print("🚀 Starting AI-Powered Delivery Failure Analysis POC - Task 1")
    print("=" * 60)
    
    # Initialize data foundation
    foundation = StreamingDataFoundation(chunk_size=1000)
    
    try:
        # Load all data
        print("📊 Loading all data sources...")
        results = foundation.load_all_data()
        
        # Print results
        print("\n📈 Loading Results:")
        print("-" * 40)
        
        for source_name, result in results.items():
            if source_name == 'summary':
                continue
                
            if result['status'] == 'success':
                print(f"✅ {source_name}: {result['total_records']} records "
                      f"({result['chunks_processed']} chunks, {result['duration_seconds']:.2f}s)")
            else:
                print(f"❌ {source_name}: {result['error']}")
        
        # Print summary
        summary = results['summary']
        print(f"\n📊 Summary:")
        print(f"   Total Duration: {summary['total_duration_seconds']:.2f}s")
        print(f"   Sources Loaded: {summary['sources_loaded']}/8")
        print(f"   Final Memory: {summary['final_memory_mb']:.2f} MB")
        
        # Get data statistics
        print("\n📋 Data Statistics:")
        print("-" * 40)
        stats = foundation.get_data_statistics()
        
        for source_name, stat in stats.items():
            if 'error' not in stat:
                print(f"📁 {source_name}: {stat['record_count']} records")
                if stat['date_range']:
                    for date_col, date_info in stat['date_range'].items():
                        print(f"   📅 {date_col}: {date_info['min']} to {date_info['max']}")
            else:
                print(f"❌ {source_name}: {stat['error']}")
        
        print("\n✅ Task 1 completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise
    finally:
        foundation.close()


if __name__ == "__main__":
    main()
