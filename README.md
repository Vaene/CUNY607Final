# Knowledge Graph Hate Incident Classification

A machine learning pipeline that uses knowledge graph-enhanced classification to process hate incident reports, comparing machine predictions against human expert classifications.

## Project Overview

This system:
1. Ingests 4 years of processed hate incident data
2. Processes ADL hate terms glossary
3. Generates semantic embeddings for incidents and terms
4. Constructs a Neo4j knowledge graph with contextual relationships
5. Processes raw incident reports using the trained knowledge graph
6. Compares machine classifications against human expert labels
7. Generates comprehensive evaluation reports with statistical analysis
8. Provides R-based visualization and analysis

## Prerequisites

### Python Requirements
- Python 3.8+
- See `requirements.txt` for package dependencies

### External Dependencies
- **Neo4j** (optional but recommended): For knowledge graph construction
  - Download: https://neo4j.com/download/
  - Default connection: `bolt://localhost:7687`
  - Set credentials in environment variables or `config.py`

### R Requirements (for analysis)
- R 4.0+
- See `install_r_packages.R` for required packages

## Installation

```bash
# 1. Clone or download the project
git@github.com:Vaene/CUNY607Final.git
cd CUNY607Final

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download spaCy model (if needed)
python -m spacy download en_core_web_lg

# 5. Install R packages (optional, for analysis)
Rscript install_r_packages.R
```