# Topic Modeling with NLP: Biodiversity Consultation Analysis

## Project Overview

This project compares different Natural Language Processing approaches for topic modeling on citizen consultation data from Make.org. The consultation focused on the question: "How can we protect and restore biodiversity together?"

The analysis examines 5,550 citizen proposals written in French and 1.8 million associated votes to identify major themes and consensus areas.

## Research Question

Which major themes emerge from the biodiversity consultation, and how do they differ depending on the method used (Word2Vec, LLM embeddings, or direct LLM interrogation)?

## Methodology

### 1. Exploratory Data Analysis
- Examination of the corpus including vocabulary analysis and vote distribution
- Statistical overview of citizen proposals

### 2. Topic Modeling with Word2Vec
- Training Word2Vec on the corpus to obtain proposal embeddings
- Vector space projection of words and proposals
- Clustering methods applied to identify themes

### 3. Topic Modeling with LLMs
- Using pre-trained language models via Hugging Face
- Extracting embeddings and applying clustering
- Direct LLM querying for categorization (if time permits)

### 4. Comparative Analysis
- Cross-comparison of clusters and themes across methods
- Discussion of strengths and limitations of local vs. large-scale vs. generative approaches

## Data

- 5,550 citizen proposals in French
- 1.8 million votes (for/against/neutral)
- Source: Make.org platform

## Technologies

- Python
- Gensim (Word2Vec implementation)
- Hugging Face Transformers
- Standard NLP and ML libraries

## Repository

GitHub: https://github.com/lioula3/Topic-modeling-with-NLP

## Supervision

- Supervisor: Anne-Cécile GAY
- Organization: Independent Data Scientist, in collaboration with Make.org
- Contact: annececile.gay@gmail.com

## Language Note

Oral communication with the supervisor is conducted in French. Written documentation is maintained in English.

## References

- Make.org: https://make.org/
- Word2Vec paper: https://arxiv.org/pdf/1301.3781.pdf
- Gensim documentation: https://radimrehurek.com/gensim/models/word2vec.html