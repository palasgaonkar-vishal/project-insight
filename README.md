# 🚚 AI-Powered Delivery Failure Analysis

A comprehensive AI-powered system for analyzing delivery failures and delays in logistics operations. This system aggregates multi-domain data, correlates events automatically, and generates human-readable insights with actionable recommendations.

## 🌟 Features

- **Natural Language Queries**: Ask questions in plain English about your delivery operations
- **Multi-Source Data Analysis**: Correlates data across orders, fleet logs, warehouse operations, weather, and customer feedback
- **AI-Powered Insights**: Uses advanced machine learning and LLM technology for intelligent analysis
- **Pattern Recognition**: Automatically detects recurring failure patterns and anomalies
- **Predictive Analysis**: Forecasts risks and provides early warnings
- **Interactive Visualizations**: Dynamic charts and graphs for data exploration
- **Real-Time Processing**: Fast query processing with immediate results

## 🏗️ System Architecture

The system consists of 6 integrated components:

1. **Data Foundation** - Streaming data processing with SQLite storage
2. **Vector Database** - Semantic search using ChromaDB and sentence transformers
3. **RAG Engine** - Retrieval-Augmented Generation with OpenRouter/DeepSeek
4. **Correlation Engine** - Multi-source data correlation and analysis
5. **Advanced Processor** - ML-based pattern recognition and predictions
6. **Web Interface** - Streamlit-based user interface

## 📊 Data Sources

The system analyzes data from 8 different sources:

- **Orders** - Order details, status, and failure reasons
- **Fleet Logs** - Driver and vehicle tracking information
- **Warehouse Logs** - Dispatch and inventory management
- **External Factors** - Weather conditions and external events
- **Customer Feedback** - Complaints, ratings, and feedback
- **Clients** - Customer information and preferences
- **Drivers** - Driver profiles and performance data
- **Warehouses** - Facility information and capacity

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/palasgaonkar-vishal/project-insight.git
   cd project-insight
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

4. **Run the application:**
   ```bash
   python run_app.py
   ```

   Or directly with Streamlit:
   ```bash
   streamlit run src/app.py
   ```

5. **Open your browser:**
   The app will automatically open at `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# LLM Provider (choose 'openai' or 'openrouter')
LLM_PROVIDER="openrouter"

# OpenRouter API Key (for DeepSeek models)
OPENROUTER_API_KEY="your_openrouter_api_key_here"

# OpenRouter Model Configuration
OPENROUTER_MODEL="deepseek/deepseek-r1-0528:free"
OPENROUTER_TEMPERATURE="0.1"

# Data Foundation Configuration
CHUNK_SIZE="1000"
DATABASE_PATH="data/delivery_failures.db"
DATA_DIR="sample-data-set"

# Vector Database Configuration
VECTOR_DB_PATH="data/vector_db"
SENTENCE_TRANSFORMER_MODEL="all-MiniLM-L6-v2"
```

### API Keys

- **OpenRouter API Key**: Get your free API key from [OpenRouter](https://openrouter.ai/)
- **DeepSeek Model**: The system uses DeepSeek R1 model for free via OpenRouter

## 🖥️ Using the Web Interface

### Main Interface

The web interface has 4 main tabs:

1. **💬 Query Interface** - Ask questions and get AI-powered answers
2. **📊 Visualizations** - Interactive charts and data exploration
3. **📈 System Stats** - Monitor system performance and data metrics
4. **ℹ️ About** - System information and technical details

### Asking Questions

1. **Enter your question** in the text area on the Query Interface tab
2. **Choose processing mode**:
   - **Auto (Recommended)** - Automatically selects the best processing method
   - **RAG Engine** - Uses retrieval-augmented generation for detailed answers
   - **Advanced Processing** - Uses ML for pattern recognition and predictions
   - **Correlation Analysis** - Analyzes relationships between different data sources

3. **Click "🔍 Analyze"** to process your question

4. **View results** with patterns, predictions, recommendations, and insights

### Example Questions

Here are some example questions you can ask:

#### 📦 Order Analysis
- "What are the main reasons for delivery failures?"
- "Which cities have the highest failure rates?"
- "How many orders failed last week?"
- "What is the average delivery time for successful orders?"

#### 🚛 Fleet & Driver Performance
- "How does driver performance affect delivery success?"
- "Which drivers have the highest failure rates?"
- "What are the most common issues with fleet operations?"
- "How can we optimize our delivery routes?"

#### 🏭 Warehouse Operations
- "How does warehouse efficiency impact delivery performance?"
- "What are the bottlenecks in our warehouse operations?"
- "Which warehouses have the best performance?"
- "How can we improve warehouse dispatch times?"

#### 🌤️ External Factors
- "How does weather affect delivery performance?"
- "What external factors cause the most delays?"
- "How do seasonal patterns impact our operations?"
- "What weather conditions lead to failures?"

#### 🔮 Predictive Analysis
- "Predict the risk of delivery failures in the next month"
- "What's likely to cause failures next week?"
- "Forecast delivery performance for next quarter"
- "Identify potential problem areas before they occur"

#### 🔍 Pattern Recognition
- "What recurring patterns do you see in our failures?"
- "Identify anomalies in our delivery operations"
- "Find unusual patterns in warehouse performance"
- "Detect seasonal trends in customer complaints"

#### 📊 Comparative Analysis
- "Compare performance across different cities"
- "How does Warehouse A compare to Warehouse B?"
- "Analyze the correlation between weather and performance"
- "Compare driver performance across different regions"

### Understanding Results

The system provides different types of results based on your question:

#### 🔍 Patterns
- **Recurring Failures**: Identifies repeated issues
- **Time-based Patterns**: Peak failure hours or days
- **Location-based Patterns**: Geographic concentration of issues
- **Performance Patterns**: Efficiency trends and bottlenecks

#### 🔮 Predictions
- **Risk Assessment**: Probability of future failures
- **Performance Forecasts**: Expected trends and changes
- **Resource Needs**: Predicted capacity requirements
- **Confidence Scores**: Reliability of predictions

#### 💡 Recommendations
- **Operational**: Immediate improvements you can make
- **Strategic**: Long-term changes for better performance
- **Tactical**: Medium-term adjustments and optimizations
- **Preventive**: Measures to avoid future issues

#### 🧠 Insights
- **Key Findings**: Summary of important discoveries
- **Data Quality**: Assessment of data completeness
- **Correlations**: Relationships between different factors
- **Trends**: Changes over time and patterns

## 📊 Data Visualizations

The Visualizations tab provides interactive charts:

- **Order Status Distribution** - Pie chart showing delivery success rates
- **City-wise Performance** - Bar chart comparing different locations
- **Time Series Trends** - Line charts showing performance over time
- **Failure Reason Analysis** - Breakdown of common failure causes

## 🔧 System Monitoring

The System Stats tab shows:

- **Data Sources**: Number of records in each data source
- **Vector Database**: Document count and search performance
- **LLM Status**: Model availability and response times
- **Processing Performance**: Query response times and throughput

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python tests/run_tests.py

# Run specific task tests
python tests/test_task1.py  # Data Foundation
python tests/test_task2.py  # Vector Database
python tests/test_task3.py  # RAG Engine
python tests/test_task4.py  # Correlation Engine
python tests/test_task5.py  # Advanced Processing
python tests/test_task6.py  # User Interface
```

## 📁 Project Structure

```
project-insight/
├── src/                          # Source code
│   ├── app.py                    # Streamlit web interface
│   ├── data_foundation.py        # Data processing and storage
│   ├── vector_database.py        # Vector database operations
│   ├── rag_engine.py            # RAG and LLM integration
│   ├── correlation_engine.py    # Multi-source correlation
│   └── advanced_processor.py    # ML and pattern recognition
├── tests/                        # Test suite
│   ├── test_task1.py            # Data Foundation tests
│   ├── test_task2.py            # Vector Database tests
│   ├── test_task3.py            # RAG Engine tests
│   ├── test_task4.py            # Correlation Engine tests
│   ├── test_task5.py            # Advanced Processing tests
│   ├── test_task6.py            # User Interface tests
│   └── run_tests.py             # Test runner
├── tasks/                        # Task documentation
│   ├── task-01-data-foundation.md
│   ├── task-02-vector-database.md
│   ├── task-03-basic-rag.md
│   ├── task-04-multi-source-correlation.md
│   ├── task-05-advanced-query-processing.md
│   ├── task-06-user-interface.md
│   └── progress.md
├── sample-data-set/              # Sample data files
│   ├── orders.csv
│   ├── fleet_logs.csv
│   ├── warehouse_logs.csv
│   ├── external_factors.csv
│   ├── feedback.csv
│   ├── clients.csv
│   ├── drivers.csv
│   └── warehouses.csv
├── data/                         # Generated data storage
│   ├── delivery_failures.db     # SQLite database
│   └── vector_db/               # Vector database files
├── requirements.txt              # Python dependencies
├── .env.example                 # Environment variables template
├── run_app.py                   # Application launcher
└── README.md                    # This file
```

## 🎯 Use Cases

### Operations Managers
- **Incident Investigation**: Quickly identify root causes of delivery failures
- **Performance Monitoring**: Track KPIs and identify improvement areas
- **Resource Optimization**: Optimize driver assignments and warehouse operations
- **Risk Management**: Predict and prevent potential issues

### Data Analysts
- **Pattern Discovery**: Find hidden patterns in operational data
- **Trend Analysis**: Understand long-term performance trends
- **Correlation Analysis**: Identify relationships between different factors
- **Predictive Modeling**: Build models for future performance

### Business Leaders
- **Strategic Planning**: Make data-driven decisions for business growth
- **Cost Optimization**: Identify areas for cost reduction
- **Customer Satisfaction**: Improve service quality and customer experience
- **Competitive Advantage**: Gain insights for market positioning

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your OpenRouter API key is correctly set in `.env`
   - Check that the API key has sufficient credits

2. **Data Loading Issues**
   - Verify that sample data files exist in `sample-data-set/`
   - Check file permissions and formats

3. **Memory Issues**
   - The system uses streaming data processing to handle large datasets
   - If you encounter memory issues, reduce `CHUNK_SIZE` in `.env`

4. **Performance Issues**
   - Vector database operations may take time on first run
   - LLM API calls depend on network speed and API response times

### Getting Help

- Check the System Stats tab for component status
- Review the logs in the terminal for error messages
- Ensure all dependencies are properly installed
- Verify environment variables are correctly set

## 🚀 Future Enhancements

- **Real-time Data Integration**: Connect to live data sources
- **Advanced Visualizations**: More interactive charts and dashboards
- **API Endpoints**: REST API for external integrations
- **Mobile Interface**: Mobile-optimized web interface
- **Multi-tenant Support**: Support for multiple organizations
- **Advanced Analytics**: More sophisticated ML models and algorithms

## 📄 License

This project is part of an AI Native Assignment and is for educational and demonstration purposes.

## 🤝 Contributing

This is a POC (Proof of Concept) project. For production use, consider:
- Adding comprehensive error handling
- Implementing proper logging and monitoring
- Adding security measures and authentication
- Optimizing performance for larger datasets
- Adding more sophisticated ML models

---

**🎉 Ready to analyze your delivery operations? Start by running the app and asking your first question!**
