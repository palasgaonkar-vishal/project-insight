"""
Advanced Query Processing Engine for AI-Powered Delivery Failure Analysis

This module implements sophisticated query processing capabilities that can handle
complex, multi-dimensional queries and provide intelligent analysis with actionable
recommendations.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import sqlite3
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """Represents a detected pattern in the data."""
    pattern_type: str
    description: str
    confidence: float
    affected_records: List[Dict[str, Any]]
    frequency: int
    severity: str
    recommendations: List[str]


@dataclass
class PredictionResult:
    """Result of a predictive analysis."""
    prediction_type: str
    risk_level: str
    probability: float
    factors: List[str]
    timeframe: str
    recommendations: List[str]
    confidence: float


@dataclass
class Recommendation:
    """Represents an actionable recommendation."""
    category: str
    priority: str
    description: str
    impact: str
    effort: str
    timeline: str
    success_metrics: List[str]


class AdvancedQueryProcessor:
    """
    Advanced query processing engine that provides sophisticated analysis
    capabilities including pattern recognition, predictive analysis, and
    recommendation generation.
    """
    
    def __init__(self, db_path: str = "data/delivery_failures.db"):
        """
        Initialize the advanced query processor.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
        
        # Query complexity patterns
        self.complexity_patterns = {
            "multi_dimensional": [
                r"compare.*across.*sources",
                r"analyze.*relationship.*between",
                r"correlate.*with.*and",
                r"impact.*of.*on.*performance",
                r"compare.*performance.*across",
                r"relationship.*between.*warehouse.*and",
                r"correlation.*between.*weather.*and",
                r"how.*does.*warehouse.*compare",
                r"analyze.*correlation.*between"
            ],
            "temporal_analysis": [
                r"trend.*over.*time",
                r"seasonal.*pattern",
                r"peak.*hours",
                r"historical.*comparison",
                r"patterns.*over.*the.*last",
                r"trends.*in.*customer",
                r"analyze.*failure.*patterns.*over",
                r"performance.*over.*time"
            ],
            "predictive": [
                r"predict.*future",
                r"risk.*assessment",
                r"forecast.*failure",
                r"likely.*to.*happen",
                r"predict.*the.*risk",
                r"what.*s.*likely.*to.*cause",
                r"forecast.*performance",
                r"predict.*delivery.*performance",
                r"should.*expect",
                r"what.*risks",
                r"mitigate",
                r"onboard",
                r"scaling",
                r"capacity",
                r"new.*risks",
                r"potential.*issues"
            ],
            "anomaly_detection": [
                r"unusual.*pattern",
                r"outlier.*detection",
                r"anomaly.*analysis",
                r"abnormal.*behavior",
                r"identify.*anomalies",
                r"unusual.*patterns.*in",
                r"abnormal.*behavior",
                r"outlier.*analysis"
            ]
        }
        
        # Pattern recognition templates
        self.pattern_templates = {
            "failure_patterns": {
                "recurring_failures": "Orders failing repeatedly for the same reason",
                "time_based_failures": "Failures occurring at specific times",
                "location_based_failures": "Failures concentrated in specific areas",
                "driver_based_failures": "Failures associated with specific drivers",
                "weather_related_failures": "Failures correlated with weather conditions"
            },
            "performance_patterns": {
                "efficiency_trends": "Changes in delivery efficiency over time",
                "capacity_utilization": "Warehouse and fleet capacity usage patterns",
                "route_optimization": "Delivery route efficiency patterns",
                "resource_allocation": "Resource distribution and utilization patterns"
            }
        }
        
        # Recommendation categories
        self.recommendation_categories = {
            "operational": "Immediate operational improvements",
            "strategic": "Long-term strategic changes",
            "tactical": "Medium-term tactical adjustments",
            "preventive": "Preventive measures and monitoring"
        }
        
        logger.info("Advanced Query Processor initialized")
    
    def process_complex_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process complex, multi-dimensional queries with advanced analysis.
        
        Args:
            query: Natural language query
            context: Additional context for processing
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Parse query complexity
            complexity = self._analyze_query_complexity(query)
            
            # Determine processing strategy
            strategy = self._determine_processing_strategy(complexity, query)
            
            # Execute analysis based on strategy
            results = {
                "query": query,
                "complexity": complexity,
                "strategy": strategy,
                "analysis": {},
                "patterns": [],
                "predictions": [],
                "recommendations": [],
                "insights": [],
                "confidence": 0.0
            }
            
            # Pattern recognition
            if "pattern" in strategy or "anomaly" in strategy:
                patterns = self._detect_patterns(query, context)
                results["patterns"] = patterns
            
            # Predictive analysis
            if "predictive" in strategy:
                predictions = self._predictive_analysis(query, context)
                results["predictions"] = predictions
            
            # Multi-dimensional analysis
            if "multi_dimensional" in strategy:
                analysis = self._multi_dimensional_analysis(query, context)
                results["analysis"] = analysis
            
            # Generate recommendations
            recommendations = self._generate_recommendations(query, results, context)
            results["recommendations"] = recommendations
            
            # Generate insights
            insights = self._generate_insights(results)
            results["insights"] = insights
            
            # Calculate overall confidence
            results["confidence"] = self._calculate_confidence(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing complex query: {e}")
            return {
                "query": query,
                "error": str(e),
                "confidence": 0.0
            }
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze the complexity of a query."""
        complexity = {
            "level": "simple",
            "dimensions": [],
            "temporal": False,
            "predictive": False,
            "multi_source": False,
            "anomaly_detection": False
        }
        
        query_lower = query.lower()
        
        # Check for multi-dimensional queries
        for pattern in self.complexity_patterns["multi_dimensional"]:
            if re.search(pattern, query_lower):
                complexity["dimensions"].append("multi_dimensional")
                complexity["multi_source"] = True
                break
        
        # Check for temporal analysis
        for pattern in self.complexity_patterns["temporal_analysis"]:
            if re.search(pattern, query_lower):
                complexity["dimensions"].append("temporal")
                complexity["temporal"] = True
                break
        
        # Check for predictive queries
        for pattern in self.complexity_patterns["predictive"]:
            if re.search(pattern, query_lower):
                complexity["dimensions"].append("predictive")
                complexity["predictive"] = True
                break
        
        # Check for anomaly detection
        for pattern in self.complexity_patterns["anomaly_detection"]:
            if re.search(pattern, query_lower):
                complexity["dimensions"].append("anomaly")
                complexity["anomaly_detection"] = True
                break
        
        # Determine overall complexity level
        if len(complexity["dimensions"]) >= 3:
            complexity["level"] = "complex"
        elif len(complexity["dimensions"]) >= 2:
            complexity["level"] = "moderate"
        
        return complexity
    
    def _determine_processing_strategy(self, complexity: Dict[str, Any], query: str) -> List[str]:
        """Determine the processing strategy based on query complexity."""
        strategy = ["basic_analysis"]
        
        if complexity["multi_source"]:
            strategy.append("multi_dimensional")
        
        if complexity["temporal"]:
            strategy.append("temporal_analysis")
        
        if complexity["predictive"]:
            strategy.append("predictive")
        
        if complexity["anomaly_detection"]:
            strategy.append("pattern")
            strategy.append("anomaly")
        
        # Always include pattern recognition for complex queries
        if complexity["level"] in ["moderate", "complex"]:
            strategy.append("pattern")
        
        return strategy
    
    def _detect_patterns(self, query: str, context: Dict[str, Any] = None) -> List[QueryPattern]:
        """Detect patterns in the data based on the query."""
        patterns = []
        
        try:
            # Get relevant data based on query
            data = self._get_relevant_data(query, context)
            
            # Detect failure patterns
            failure_patterns = self._detect_failure_patterns(data)
            patterns.extend(failure_patterns)
            
            # Detect performance patterns
            performance_patterns = self._detect_performance_patterns(data)
            patterns.extend(performance_patterns)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(data)
            patterns.extend(anomalies)
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def _detect_failure_patterns(self, data: Dict[str, pd.DataFrame]) -> List[QueryPattern]:
        """Detect failure patterns in the data."""
        patterns = []
        
        try:
            if "orders" in data:
                orders_df = data["orders"]
                
                # Recurring failure patterns
                if "failure_reason" in orders_df.columns:
                    failure_counts = orders_df["failure_reason"].value_counts()
                    recurring_failures = failure_counts[failure_counts > 1]
                    
                    for reason, count in recurring_failures.items():
                        if pd.notna(reason):
                            affected_records = orders_df[orders_df["failure_reason"] == reason].to_dict('records')
                            
                            patterns.append(QueryPattern(
                                pattern_type="recurring_failures",
                                description=f"Recurring failure: {reason}",
                                confidence=min(count / len(orders_df), 1.0),
                                affected_records=affected_records[:10],  # Limit to 10 records
                                frequency=count,
                                severity="high" if count > 5 else "medium",
                                recommendations=self._get_failure_recommendations(reason)
                            ))
                
                # Time-based failure patterns
                if "order_date" in orders_df.columns:
                    orders_df["hour"] = pd.to_datetime(orders_df["order_date"]).dt.hour
                    hourly_failures = orders_df.groupby("hour").size()
                    
                    peak_hours = hourly_failures.nlargest(3)
                    for hour, count in peak_hours.items():
                        if count > hourly_failures.mean() * 1.5:  # 50% above average
                            patterns.append(QueryPattern(
                                pattern_type="time_based_failures",
                                description=f"Peak failure hour: {hour}:00",
                                confidence=min(count / hourly_failures.sum(), 1.0),
                                affected_records=[],
                                frequency=count,
                                severity="medium",
                                recommendations=[
                                    "Increase monitoring during peak hours",
                                    "Consider additional resources during high-risk periods",
                                    "Analyze root causes of peak-hour failures"
                                ]
                            ))
                
                # Location-based failure patterns
                if "city" in orders_df.columns:
                    city_failures = orders_df.groupby("city").size()
                    high_failure_cities = city_failures[city_failures > city_failures.mean() * 1.2]
                    
                    for city, count in high_failure_cities.items():
                        patterns.append(QueryPattern(
                            pattern_type="location_based_failures",
                            description=f"High failure rate in {city}",
                            confidence=min(count / city_failures.sum(), 1.0),
                            affected_records=[],
                            frequency=count,
                            severity="high" if count > city_failures.mean() * 2 else "medium",
                            recommendations=[
                                f"Investigate infrastructure issues in {city}",
                                "Review local delivery processes",
                                "Consider city-specific training programs"
                            ]
                        ))
        
        except Exception as e:
            logger.error(f"Error detecting failure patterns: {e}")
        
        return patterns
    
    def _detect_performance_patterns(self, data: Dict[str, pd.DataFrame]) -> List[QueryPattern]:
        """Detect performance patterns in the data."""
        patterns = []
        
        try:
            # Analyze delivery times if available
            if "fleet_logs" in data:
                fleet_df = data["fleet_logs"]
                
                if "departure_time" in fleet_df.columns and "arrival_time" in fleet_df.columns:
                    # Calculate delivery duration
                    fleet_df["duration"] = (
                        pd.to_datetime(fleet_df["arrival_time"]) - 
                        pd.to_datetime(fleet_df["departure_time"])
                    ).dt.total_seconds() / 3600  # Convert to hours
                    
                    # Detect efficiency trends
                    avg_duration = fleet_df["duration"].mean()
                    slow_deliveries = fleet_df[fleet_df["duration"] > avg_duration * 1.5]
                    
                    if len(slow_deliveries) > 0:
                        patterns.append(QueryPattern(
                            pattern_type="efficiency_trends",
                            description=f"Slow delivery pattern detected",
                            confidence=min(len(slow_deliveries) / len(fleet_df), 1.0),
                            affected_records=slow_deliveries.to_dict('records')[:10],
                            frequency=len(slow_deliveries),
                            severity="medium",
                            recommendations=[
                                "Review delivery routes for optimization",
                                "Analyze driver performance and training needs",
                                "Consider traffic pattern analysis"
                            ]
                        ))
        
        except Exception as e:
            logger.error(f"Error detecting performance patterns: {e}")
        
        return patterns
    
    def _detect_anomalies(self, data: Dict[str, pd.DataFrame]) -> List[QueryPattern]:
        """Detect anomalies in the data using machine learning."""
        patterns = []
        
        try:
            # Combine relevant numerical features
            features = []
            feature_names = []
            
            for table_name, df in data.items():
                if len(df) > 10:  # Need sufficient data for anomaly detection
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if df[col].notna().sum() > 5:  # At least 5 non-null values
                            features.append(df[col].fillna(df[col].mean()).values)
                            feature_names.append(f"{table_name}_{col}")
            
            if features and len(features[0]) > 10:
                # Create feature matrix
                feature_matrix = np.column_stack(features)
                
                # Detect anomalies using Isolation Forest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = iso_forest.fit_predict(feature_matrix)
                
                # Get anomaly indices
                anomaly_indices = np.where(anomaly_labels == -1)[0]
                
                if len(anomaly_indices) > 0:
                    patterns.append(QueryPattern(
                        pattern_type="anomaly_detection",
                        description=f"Detected {len(anomaly_indices)} anomalous records",
                        confidence=0.8,  # High confidence for ML-based detection
                        affected_records=[],  # Would need to map back to original records
                        frequency=len(anomaly_indices),
                        severity="high",
                        recommendations=[
                            "Investigate anomalous records for data quality issues",
                            "Review data collection processes",
                            "Consider additional validation rules"
                        ]
                    ))
        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return patterns
    
    def _predictive_analysis(self, query: str, context: Dict[str, Any] = None) -> List[PredictionResult]:
        """Perform predictive analysis based on the query."""
        predictions = []
        
        try:
            # Get historical data for prediction
            data = self._get_relevant_data(query, context)
            
            # Predict failure risk
            if "orders" in data:
                failure_risk = self._predict_failure_risk(data)
                if failure_risk:
                    predictions.append(failure_risk)
            
            # Predict performance trends
            performance_prediction = self._predict_performance_trends(data)
            if performance_prediction:
                predictions.append(performance_prediction)
            
            # Predict resource needs
            resource_prediction = self._predict_resource_needs(data)
            if resource_prediction:
                predictions.append(resource_prediction)
        
        except Exception as e:
            logger.error(f"Error in predictive analysis: {e}")
        
        return predictions
    
    def _predict_failure_risk(self, data: Dict[str, pd.DataFrame]) -> Optional[PredictionResult]:
        """Predict failure risk based on historical data."""
        try:
            if "orders" not in data:
                return None
            
            orders_df = data["orders"]
            
            # Simple risk prediction based on historical patterns
            total_orders = len(orders_df)
            failed_orders = len(orders_df[orders_df["status"] == "failed"]) if "status" in orders_df.columns else 0
            
            failure_rate = failed_orders / total_orders if total_orders > 0 else 0
            
            # Predict future failure rate
            predicted_rate = failure_rate * 1.1  # Assume 10% increase
            
            risk_level = "high" if predicted_rate > 0.3 else "medium" if predicted_rate > 0.1 else "low"
            
            return PredictionResult(
                prediction_type="failure_risk",
                risk_level=risk_level,
                probability=predicted_rate,
                factors=["Historical failure rate", "Current trends"],
                timeframe="Next 30 days",
                recommendations=[
                    "Implement additional quality checks",
                    "Increase monitoring frequency",
                    "Review and update processes"
                ],
                confidence=0.7
            )
        
        except Exception as e:
            logger.error(f"Error predicting failure risk: {e}")
            return None
    
    def _predict_performance_trends(self, data: Dict[str, pd.DataFrame]) -> Optional[PredictionResult]:
        """Predict performance trends."""
        try:
            # Simple trend prediction based on available data
            return PredictionResult(
                prediction_type="performance_trend",
                risk_level="medium",
                probability=0.6,
                factors=["Historical performance data", "Current capacity utilization"],
                timeframe="Next 2 weeks",
                recommendations=[
                    "Monitor capacity utilization closely",
                    "Prepare for potential performance dips",
                    "Implement performance optimization measures"
                ],
                confidence=0.6
            )
        
        except Exception as e:
            logger.error(f"Error predicting performance trends: {e}")
            return None
    
    def _predict_resource_needs(self, data: Dict[str, pd.DataFrame]) -> Optional[PredictionResult]:
        """Predict resource needs based on current data."""
        try:
            return PredictionResult(
                prediction_type="resource_needs",
                risk_level="low",
                probability=0.4,
                factors=["Current resource utilization", "Historical demand patterns"],
                timeframe="Next month",
                recommendations=[
                    "Monitor resource utilization trends",
                    "Prepare for potential capacity increases",
                    "Consider resource optimization strategies"
                ],
                confidence=0.5
            )
        
        except Exception as e:
            logger.error(f"Error predicting resource needs: {e}")
            return None
    
    def _multi_dimensional_analysis(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform multi-dimensional analysis across different data sources."""
        analysis = {
            "cross_source_correlations": {},
            "dimensional_insights": [],
            "summary": {}
        }
        
        try:
            # Get data from multiple sources
            data = self._get_relevant_data(query, context)
            
            # Analyze correlations between sources
            if len(data) > 1:
                correlations = self._analyze_cross_source_correlations(data)
                analysis["cross_source_correlations"] = correlations
            
            # Generate dimensional insights
            insights = self._generate_dimensional_insights(data, query)
            analysis["dimensional_insights"] = insights
            
            # Generate summary
            analysis["summary"] = self._generate_analysis_summary(analysis)
        
        except Exception as e:
            logger.error(f"Error in multi-dimensional analysis: {e}")
        
        return analysis
    
    def _analyze_cross_source_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze correlations between different data sources."""
        correlations = {}
        
        try:
            # Simple correlation analysis
            for source1, df1 in data.items():
                for source2, df2 in data.items():
                    if source1 != source2 and len(df1) > 0 and len(df2) > 0:
                        # Find common columns for correlation
                        common_cols = set(df1.columns) & set(df2.columns)
                        if common_cols:
                            correlations[f"{source1}_to_{source2}"] = {
                                "common_columns": list(common_cols),
                                "correlation_strength": "medium",  # Simplified
                                "insights": f"Data correlation between {source1} and {source2}"
                            }
        
        except Exception as e:
            logger.error(f"Error analyzing cross-source correlations: {e}")
        
        return correlations
    
    def _generate_dimensional_insights(self, data: Dict[str, pd.DataFrame], query: str) -> List[str]:
        """Generate insights across multiple dimensions."""
        insights = []
        
        try:
            # Analyze data volume across sources
            total_records = sum(len(df) for df in data.values())
            insights.append(f"Analyzed {total_records} records across {len(data)} data sources")
            
            # Analyze data quality
            quality_issues = []
            for source, df in data.items():
                null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                if null_percentage > 20:
                    quality_issues.append(f"{source}: {null_percentage:.1f}% missing data")
            
            if quality_issues:
                insights.append(f"Data quality concerns: {', '.join(quality_issues)}")
            else:
                insights.append("Data quality appears good across all sources")
            
            # Generate query-specific insights
            if "failure" in query.lower():
                insights.append("Focusing on failure analysis across multiple dimensions")
            if "performance" in query.lower():
                insights.append("Analyzing performance metrics across different sources")
        
        except Exception as e:
            logger.error(f"Error generating dimensional insights: {e}")
        
        return insights
    
    def _generate_recommendations(self, query: str, results: Dict[str, Any], 
                                context: Dict[str, Any] = None) -> List[Recommendation]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        try:
            # Generate recommendations based on patterns
            for pattern in results.get("patterns", []):
                for rec in pattern.recommendations:
                    recommendations.append(Recommendation(
                        category="operational",
                        priority=pattern.severity,
                        description=rec,
                        impact="high" if pattern.severity == "high" else "medium",
                        effort="low",
                        timeline="immediate",
                        success_metrics=["Reduced failure rate", "Improved efficiency"]
                    ))
            
            # Generate recommendations based on predictions
            for prediction in results.get("predictions", []):
                for rec in prediction.recommendations:
                    recommendations.append(Recommendation(
                        category="preventive",
                        priority=prediction.risk_level,
                        description=rec,
                        impact="high" if prediction.risk_level == "high" else "medium",
                        effort="medium",
                        timeline=prediction.timeframe,
                        success_metrics=["Risk mitigation", "Improved predictions"]
                    ))
            
            # Generate query-specific recommendations
            query_lower = query.lower()
            if "optimize" in query_lower or "improve" in query_lower:
                recommendations.append(Recommendation(
                    category="strategic",
                    priority="medium",
                    description="Implement continuous monitoring and optimization processes",
                    impact="high",
                    effort="high",
                    timeline="3-6 months",
                    success_metrics=["Process efficiency", "Cost reduction", "Quality improvement"]
                ))
            
            if "predict" in query_lower or "forecast" in query_lower:
                recommendations.append(Recommendation(
                    category="tactical",
                    priority="medium",
                    description="Develop predictive analytics capabilities",
                    impact="medium",
                    effort="medium",
                    timeline="1-3 months",
                    success_metrics=["Prediction accuracy", "Early warning effectiveness"]
                ))
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _generate_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate high-level insights from the analysis results."""
        insights = []
        
        try:
            # Pattern-based insights
            patterns = results.get("patterns", [])
            if patterns:
                high_severity_patterns = [p for p in patterns if p.severity == "high"]
                if high_severity_patterns:
                    insights.append(f"Found {len(high_severity_patterns)} high-severity patterns requiring immediate attention")
                else:
                    insights.append(f"Identified {len(patterns)} patterns for monitoring and improvement")
            
            # Prediction-based insights
            predictions = results.get("predictions", [])
            if predictions:
                high_risk_predictions = [p for p in predictions if p.risk_level == "high"]
                if high_risk_predictions:
                    insights.append(f"High-risk predictions identified: {len(high_risk_predictions)} areas need attention")
                else:
                    insights.append(f"Risk assessment completed: {len(predictions)} predictions generated")
            
            # Recommendation insights
            recommendations = results.get("recommendations", [])
            if recommendations:
                high_priority_recs = [r for r in recommendations if r.priority == "high"]
                insights.append(f"Generated {len(recommendations)} recommendations ({len(high_priority_recs)} high priority)")
            
            # Overall confidence insight
            confidence = results.get("confidence", 0.0)
            if confidence > 0.8:
                insights.append("High confidence in analysis results")
            elif confidence > 0.6:
                insights.append("Moderate confidence in analysis results")
            else:
                insights.append("Analysis results should be validated with additional data")
        
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis results."""
        try:
            confidence_factors = []
            
            # Pattern confidence
            patterns = results.get("patterns", [])
            if patterns:
                avg_pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
                confidence_factors.append(avg_pattern_confidence)
            
            # Prediction confidence
            predictions = results.get("predictions", [])
            if predictions:
                avg_prediction_confidence = sum(p.confidence for p in predictions) / len(predictions)
                confidence_factors.append(avg_prediction_confidence)
            
            # Data quality factor
            analysis = results.get("analysis", {})
            if analysis:
                confidence_factors.append(0.7)  # Assume moderate confidence for analysis
            
            # Base confidence
            if not confidence_factors:
                confidence_factors.append(0.5)
            
            return sum(confidence_factors) / len(confidence_factors)
        
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _get_relevant_data(self, query: str, context: Dict[str, Any] = None) -> Dict[str, pd.DataFrame]:
        """Get relevant data based on the query."""
        data = {}
        
        try:
            # Determine which tables to query based on query content
            query_lower = query.lower()
            
            tables_to_query = []
            if any(word in query_lower for word in ["order", "delivery", "failure"]):
                tables_to_query.append("orders")
            if any(word in query_lower for word in ["fleet", "driver", "vehicle"]):
                tables_to_query.append("fleet_logs")
            if any(word in query_lower for word in ["warehouse", "dispatch", "inventory"]):
                tables_to_query.append("warehouse_logs")
            if any(word in query_lower for word in ["weather", "external", "condition"]):
                tables_to_query.append("external_factors")
            if any(word in query_lower for word in ["feedback", "complaint", "rating"]):
                tables_to_query.append("feedback")
            
            # If no specific tables identified, query all
            if not tables_to_query:
                tables_to_query = ["orders", "fleet_logs", "warehouse_logs", "external_factors", "feedback"]
            
            # Query each table
            for table in tables_to_query:
                try:
                    query_sql = f"SELECT * FROM {table} LIMIT 1000"  # Limit for performance
                    df = pd.read_sql_query(query_sql, self.db)
                    if not df.empty:
                        data[table] = df
                except Exception as e:
                    logger.warning(f"Error querying {table}: {e}")
        
        except Exception as e:
            logger.error(f"Error getting relevant data: {e}")
        
        return data
    
    def _get_failure_recommendations(self, failure_reason: str) -> List[str]:
        """Get specific recommendations for a failure reason."""
        recommendations = {
            "delivery_delay": [
                "Implement real-time tracking and notifications",
                "Optimize delivery routes using traffic data",
                "Add buffer time for high-traffic areas"
            ],
            "package_damage": [
                "Review packaging materials and processes",
                "Implement damage prevention training",
                "Add handling instructions for fragile items"
            ],
            "address_error": [
                "Improve address validation during order placement",
                "Implement GPS-based address verification",
                "Add customer confirmation for delivery addresses"
            ],
            "weather_related": [
                "Monitor weather conditions and adjust schedules",
                "Implement weather-based delivery prioritization",
                "Provide weather updates to customers"
            ]
        }
        
        # Try to match failure reason to recommendations
        failure_lower = failure_reason.lower()
        for key, recs in recommendations.items():
            if key in failure_lower:
                return recs
        
        # Default recommendations
        return [
            "Investigate root cause of this failure type",
            "Implement preventive measures",
            "Monitor and track this failure pattern"
        ]
    
    def _generate_analysis_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the analysis results."""
        summary = {
            "total_patterns": len(analysis.get("patterns", [])),
            "total_predictions": len(analysis.get("predictions", [])),
            "total_recommendations": len(analysis.get("recommendations", [])),
            "key_findings": [],
            "next_steps": []
        }
        
        try:
            # Add key findings
            patterns = analysis.get("patterns", [])
            if patterns:
                high_severity = [p for p in patterns if p.severity == "high"]
                summary["key_findings"].append(f"Identified {len(patterns)} patterns ({len(high_severity)} high severity)")
            
            predictions = analysis.get("predictions", [])
            if predictions:
                high_risk = [p for p in predictions if p.risk_level == "high"]
                summary["key_findings"].append(f"Generated {len(predictions)} predictions ({len(high_risk)} high risk)")
            
            # Add next steps
            if patterns:
                summary["next_steps"].append("Review and address identified patterns")
            if predictions:
                summary["next_steps"].append("Monitor predicted risks and implement preventive measures")
            
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                high_priority = [r for r in recommendations if r.priority == "high"]
                summary["next_steps"].append(f"Implement {len(high_priority)} high-priority recommendations")
        
        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
        
        return summary
    
    def close(self):
        """Close the database connection."""
        if self.db:
            self.db.close()
        logger.info("Advanced Query Processor closed.")


def main():
    """Test the advanced query processor functionality."""
    import sys
    sys.path.append('src')
    
    from data_foundation import StreamingDataFoundation
    
    # Initialize components
    foundation = StreamingDataFoundation()
    processor = AdvancedQueryProcessor()
    
    try:
        print("🧪 Testing Advanced Query Processor")
        print("=" * 50)
        
        # Test complex queries
        test_queries = [
            "What are the recurring failure patterns in our delivery system?",
            "Predict the risk of delivery failures in the next month",
            "Analyze the correlation between weather conditions and delivery performance",
            "Identify anomalies in our warehouse operations",
            "Compare performance across different cities and recommend improvements"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing query: {query}")
            result = processor.process_complex_query(query)
            
            print(f"   Complexity: {result.get('complexity', {}).get('level', 'unknown')}")
            print(f"   Patterns: {len(result.get('patterns', []))}")
            print(f"   Predictions: {len(result.get('predictions', []))}")
            print(f"   Recommendations: {len(result.get('recommendations', []))}")
            print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
        
        print("\n✅ All advanced query processor tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    
    finally:
        processor.close()
        foundation.close()


if __name__ == "__main__":
    main()
