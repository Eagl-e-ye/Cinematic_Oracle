# Cinematic Oracle

## Overview
**Cinematic Oracle** is an intelligent, agentic movie recommendation system designed to move beyond static filtering. By combining multi-modal data with **Large Language Model (LLM)** query understanding, the system delivers context-aware, explainable, and highly personalized suggestions.

---

## Key Features
**Agentic Architecture**: Uses an LLM-based parser to understand natural language and extract structured intent (filters, references, and preferences).
**Multi-Modal Similarity**: Computes recommendations based on a mix of semantic text (overviews), metadata (genres, cast, directors), and user signals (ratings, popularity).
**Dynamic Pipeline Selection**: Automatically chooses the best recommendation strategy—such as similarity-based, actor-focused, or discovery—based on the user's specific query.
**Hybrid Scoring**: Combines various signals using a weighted ranking system that adjusts dynamically to prioritize what matters most in a query.
**Advanced Filtering**: Post-scoring filters ensure results meet specific criteria for year, language, runtime, and quality thresholds (minimum ratings/votes).

---

## System Architecture
The system follows a modular pipeline to transform a raw user query into a ranked list of movies:

1.  **LLM Parser (Groq API)**: Converts natural language into structured JSON.
2.  **Pipeline Selector**: Identifies the query type (e.g., "Reference-based" or "Semantic") and assigns the appropriate logic.
3.  **Similarity Engine**: Calculates closeness using multiple methods:
    **Textual**: Cosine similarity via the `all-MiniLM-L6-v2` model.
    **Categorical**: Jaccard similarity for genre matching.
    **Relational**: Embeddings for actors and directors to find style similarities.
4.  **Scoring & Filtering**: Aggregates scores using the formula:
    $$Score = \sum (weight \times feature\_score)$$ 

---

## Data Sources & Tech Stack
**Datasets**: MovieLens (ratings, tags) and TMDB API (overviews, cast, directors).
* **Models**: 
    **LLM**: Groq API for intent extraction.
    **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformer).
**Core Tools**: `tmdbld` for merging, Python for normalization and feature engineering (e.g., `underrated_score`).

---

## Example Queries
The system handles a wide variety of complex requests:
**Reference-based**: "Suggest movies like Interstellar".
**Semantic**: "Space movies with AI".
**Filtered**: "Hindi thriller movies like Andhadhun with high ratings".


## Recommendation outputs: Prompts: 
• suggest movies like Interstellar: Illuminate | Cobra Gypsies | Le mystère de Fatima 
| Illuminate | Goddaddy | Unarmed Verses | The Kathy & Mo Show: Parallel | How Do 
You Write a Joe Schermann | Jean Renoir: Part One - From La Belle Époque to World | 
When I Find the Ocean  
• Hindi thriller movies like Andhadhun with high ratings: Dassehra | State of 
Siege: Temple Attack | The Bright Day | Supermen of Malegaon | Nero's Guests | One 
Heart: The A.R. Rahman Concert Film | Dilwale Dulhania Le Jayenge | Reason 
