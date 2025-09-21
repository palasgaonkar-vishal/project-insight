# Task 3: Basic RAG Engine

## Objective
Implement the core RAG (Retrieval-Augmented Generation) functionality that combines vector search with LLM to answer questions about delivery failures.

## Scope
- Create RAG query processing pipeline
- Implement context retrieval from vector database
- Add LLM integration for response generation
- Create basic prompt engineering for delivery analysis

## Technical Requirements
- Implement query parsing and intent classification
- Create context retrieval from multiple data sources
- Add OpenAI GPT-4 integration for response generation
- Implement basic prompt templates for delivery analysis

## Implementation Details
- **File**: `src/rag_engine.py`
- **Dependencies**: openai, langchain (optional)
- **LLM**: OpenAI GPT-4 for response generation
- **Prompt Templates**: Create templates for different query types

## Success Criteria
- RAG engine processes queries successfully
- Context retrieval works across multiple data sources
- LLM generates relevant responses based on retrieved context
- Basic delivery analysis queries are answered correctly

## Testing
- Test with simple queries like "How many orders failed?"
- Verify context retrieval from vector database
- Test LLM response generation
- Validate response quality with sample queries

## Estimated Time
4-5 hours

## Dependencies
- Task 1: Data Foundation Setup
- Task 2: Vector Database Setup

## Next Task
Task 4: Multi-Source Correlation
