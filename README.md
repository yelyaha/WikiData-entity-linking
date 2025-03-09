# WikiData-based Entity Linking for Medical Applications

This repository contains a comprehensive solution for medical entity linking using Transformer-based models. The project leverages data from Wikidata and Wikipedia, eliminating the need for any commercial components.

## Overview

Medical entity linking task is crucial for connecting unstructured medical text terminology to structured knowledge bases. This project provides:

1. **Dataset construction**: Custom Wikidata-sourced corpora in the German language for ATC and ICD-10-GM medical terminology using SPARQL queries
2. **Model training**: Transformer-based models from the [Hugging Face Hub](https://huggingface.co) with contrastive learning fine-tuning
3. **Evaluation**: Benchmarking on the external dataset using FAISS search library
4. **Web Demo**: A ready-to-use Flask application with PostgreSQL vector database

## Repository Structure

### `/demo`
Contains a web application for entity linking:
- Flask-based API for entity linking
- PostgreSQL with pgvector for efficient dense retrieval
- Docker configuration for easy deployment
- Development and production environments

### `/named_entity_linking`
Contains code for model training and evaluation:
- Scripts for constructing custom Wikidata-sourced corpora using SPARQL queries
- Training pipeline for custom models with contrastive metric learning objectives similar to the ([SapBERT paper](https://aclanthology.org/2021.naacl-main.334.pdf))
- Evaluation pipeline using FAISS search library for efficient retrieval with custom external datasets