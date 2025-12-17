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

### System Requirements
- **Python 3.11+** (tested with 3.11.14)
- **16GB+ RAM** (for embeddings generation)
- **10GB+ disk space** (for embeddings, data, and knowledge graph)
- **macOS, Linux, or Windows** (tested on macOS)

### Python Requirements
- Python 3.11+
- See `requirements.txt` for package dependencies

### External Dependencies
- **Neo4j** (optional but recommended): For knowledge graph construction
  - Download: https://neo4j.com/download/
  - Default connection: `bolt://localhost:7687`
  - Set credentials in environment variables or `config.py`

### R Requirements (for analysis and presentation)
- R 4.0+
- RStudio (recommended)
- pandoc (for R Markdown rendering)
- See `install_r_packages.R` for required packages

---

## Installation (From Scratch)

### Step1: Clone the Repository

```bash
git clone git@github.com:Vaene/CUNY607Final.git
cd CUNY607Final
## Step2: Set Up Python Environment

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```
### Step3: Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Download spaCy language model (for NLP processing)
python -m spacy download en_core_web_lg
```
### Step4: Set Up Neo4j (Optional but Recommended)
**Option A:** Neo4j Desktop (Recommended)

Download Neo4j Desktop: https://neo4j.com/download/
Install and create a new project
Create a new database with these settings:
Name: hate-incidents (or your choice)
Password: Set a secure password
Version: Latest stable (5.x recommended)
Start the database
Note the connection details (typically bolt://localhost:7687)

**Option B:** Neo4j Docker

```bash
docker run \
    --name neo4j-hate-incidents \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    -v $HOME/neo4j/data:/data \
    neo4j:latest
```
### Step5: Configure Connection Settings
Edit config.py to set your Neo4j credentials:


```python
# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password_here"  # Change this!
```
Or set environment variables:


```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
```
### Step6: Set Up Data Directory Structure

```bash
# Create required directories (if they don't exist)
mkdir -p data/raw
mkdir -p data/processed
mkdir -p embeddings
mkdir -p output
mkdir -p results
mkdir -p logs
```
### Step7: Add Your Data Files
Place your data files in the appropriate directories:

data/raw/
  ├── human_processed_lastyear.csv        # Human-labeled incidents
  ├── glossary.csv                         # ADL hate terms glossary
  └── incidents_raw_lastyear.csv          # Raw incidents to process

data/processed/
  └── historical_processed.csv            # 4 years of historical data

### Step8: Install R and Dependencies (Optional)
**For macOS:**

```bash
# Install R
brew install r

# Install pandoc (required for R Markdown)
brew install pandoc

# Install R packages
Rscript install_r_packages.R
For Linux (Ubuntu/Debian):
```

```bash
# Install R
sudo apt update
sudo apt install r-base r-base-dev

# Install pandoc
sudo apt install pandoc

# Install system dependencies for R packages
sudo apt install libcurl4-openssl-dev libssl-dev libxml2-dev

# Install R packages
Rscript install_r_packages.R
```
**For Windows:**

Download R from https://cran.r-project.org/bin/windows/base/
Download RStudio from https://www.rstudio.com/products/rstudio/download/
Download pandoc from https://pandoc.org/installing.html
Open R or RStudio and run: source("install_r_packages.R")

## Running the Pipeline (Soup to Nuts)
### Quick Start: Run Entire Pipeline

```bash
# Activate virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Run all 7 phases
python main_pipeline.py
```
# This will execute:
# Phase 1: Data Ingestion
# Phase 2: Glossary Processing  
# Phase 3: Embedding Generation (takes 5-10 minutes)
# Phase 4: Knowledge Graph Construction
# Phase 5: Raw Incident Processing
# Phase 6: Comparison & Evaluation
# Phase 7: R Export

## Running Individual Phases

```bash
# Run specific phase
python main_pipeline.py --start 3 --end 3

# Run phases 1-4 only
python main_pipeline.py --start 1 --end 4

# Run phases 6-7 (evaluation and export)
python main_pipeline.py --start 6 --end 7

# Skip specific phases
python main_pipeline.py --skip 4  # Skip Neo4j phase if not installed
```
## Phase-by-Phase Execution
### Phase 1: Data Ingestion

```bash
python main_pipeline.py --start 1 --end 1
```
Loads historical incident data (16,893 incidents)
Validates data quality
Creates processed datasets
**Solutions:** data/processed/historical_processed.csv

### Phase 2: Glossary Processing

```bash
python main_pipeline.py --start 2 --end 2
```
Loads ADL glossary (1,142 hate terms)
Cleans and standardizes terminology
Extracts categories and definitions
**Solutions:** data/processed/glossary_final.csv
Phase 3: Embedding Generation

```bash
python main_pipeline.py --start 3 --end 3
```
Generates 768-dimensional embeddings for all incidents
Generates embeddings for all glossary terms
Uses sentence-transformers (all-MiniLM-L6-v2 model)
Time: 5-10 minutes for full dataset
**Solutions:**
embeddings/incident_embeddings.npy (16,893 × 768)
embeddings/glossary_embeddings.npy (1,142 × 768)

### Phase 4: Knowledge Graph Construction (Optional)

```bash
python main_pipeline.py --start 4 --end 4
```
Connects to Neo4j database
Creates nodes: Incidents, Terms, Categories, Locations, Groups
Creates relationships: USES_TERM, SIMILAR_TO, BELONGS_TO, OCCURRED_IN
Requires: Neo4j running
**Solutions:** Knowledge graph with 21,594 nodes, 28,394 relationships
Skip if: Neo4j not installed (add --skip 4)
Phase 5: Raw Incident Processing

```bash
python main_pipeline.py --start 5 --end 5
```
Processes 9,603 new raw incidents
Matches to glossary terms via semantic similarity
Assigns categories with confidence scores

Time: 2-5 minutes
**Solutions:** output/machine_processed_incidents.csv

### Phase 6: Comparison & Evaluation

```bash
python main_pipeline.py --start 6 --end 6
```
Compares machine vs human classifications
Calculates accuracy, precision, recall, F1 scores
Generates confusion matrices
Performs statistical tests (McNemar, Chi-square, t-test)
Creates data quality comparison

**Solutions:**
results/evaluation_report.txt
results/evaluation_metrics.json
results/comparison_results.csv
results/confusion_matrix.png
results/data_quality_comparison.json

###Phase 7: R Export

```bash
python main_pipeline.py --start 7 --end 7
```
Exports all data in R-friendly formats
Creates structured datasets for analysis
Generates metadata files

**Solutions:**
output/structured_incidents.csv
output/incident_embeddings_for_r.csv
output/glossary_embeddings_for_r.csv
output/category_performance.csv
output/confidence_analysis.csv
output/temporal_analysis.csv
output/geographic_analysis.csv
output/r_metadata.json
Generating Reports and Presentations
Generate Analysis Report (R Markdown)

```bash
# Render the comprehensive analysis report
Rscript -e "rmarkdown::render('analysis_report_final.Rmd')"
```

# Opens: analysis_report_final.html
The report includes:

Data quality comparison (machine vs human)
Category granularity analysis (23x more detailed)
Confidence score distributions
Geographic and temporal trends
Processing efficiency comparison
Production deployment recommendations
Generate Presentation Slides

```bash
# Render the 8-minute presentation
Rscript -e "rmarkdown::render('project_presentation.Rmd')"
```

# Opens: project_presentation.html
The presentation covers:

Original proposal and pivot rationale
Exponential growth problem (glossary vs incidents)
Knowledge graph solution
Machine vs human comparison
Production deployment strategy
Key achievements
Understanding the Output
Key Metrics
After running the full pipeline, you'll see:

```
✅ PIPELINE COMPLETE

Data Processing:
  - Historical incidents: 16,893
  - Glossary terms: 1,142
  - New incidents processed: 9,603
  - Knowledge graph nodes: 21,594
  - Knowledge graph relationships: 28,394

Performance:
  - Data completeness: 100% (machine and human)
  - Category granularity: 23 categories (machine) vs 1 (human)
  - Average confidence: 0.576
  - Processing time: Seconds vs ~1,164 hours (human)

Quality Comparison:
  - Machine matches human data quality ✓
  - Machine provides 23x more granular classification ✓
  - Machine quantifies uncertainty with confidence scores ✓
```
## Key Files Generated
File	Description
results/evaluation_report.txt	Comprehensive evaluation report
results/confusion_matrix.png	Visual confusion matrix
results/value_added_report.txt	Machine vs human comparison
output/machine_processed_incidents.csv	All machine classifications
output/structured_incidents.csv	Analysis-ready dataset
analysis_report_final.html	Interactive HTML report
project_presentation.html	Presentation slides

## Troubleshooting
### Common Issues
1. Neo4j Connection Failed
Error: Failed to connect to Neo4j

**Solutions:**

Verify Neo4j is running: Check Neo4j Desktop or docker ps
Check credentials in config.py
Test connection: Open Neo4j Browser at http://localhost:7474
OR Skip Phase 4: python main_pipeline.py --skip 4

2. Out of Memory (Embeddings)
Error: MemoryError or Killed during Phase 3

**Solutions:**

Reduce batch size in config.py: BATCH_SIZE = 32 → 16
Close other applications
Process in smaller chunks (modify phase3_embeddings.py)

3. Missing Data Files
Error: FileNotFoundError: data/raw/...

**Solutions:**

Verify data files are in correct directories
Check file names match exactly (case-sensitive)
Run python main_pipeline.py --start 1 --end 1 to validate

4. R Markdown Rendering Failed
Error: pandoc version 1.12.3 or higher is required

**Solutions:**

Install pandoc: brew install pandoc (macOS) or download from https://pandoc.org
Verify installation: pandoc --version
Install missing R packages: Rscript install_r_packages.R

5. Module Import Errors
Error: ModuleNotFoundError: No module named 'sentence_transformers'

**Solutions:**

Verify virtual environment is activated: which python should show .venv/bin/python
Reinstall dependencies: pip install -r requirements.txt
Check Python version: python --version (need 3.11+)
Fixing /tmp Directory Issues
If you encounter /tmp permission errors:


```bash
# Create local temp directory
mkdir -p ~/Documents/CUNY/607/FinalProject/Cassidy/tmp
export TMPDIR=~/Documents/CUNY/607/FinalProject/Cassidy/tmp

# Then run pipeline
python main_pipeline.py
```
## Project Structure


CUNY607Final/
├── main_pipeline.py              # Main execution script
├── config.py                      # Configuration settings
├── utils.py                       # Shared utilities
├── phase1_data_ingestion.py      # Phase 1: Data loading
├── phase2_glossary.py            # Phase 2: Glossary processing
├── phase3_embeddings.py          # Phase 3: Embedding generation
├── phase4_neo4j.py               # Phase 4: Knowledge graph
├── phase5_incident_processing.py # Phase 5: Incident classification
├── phase6_comparison_evaluation.py # Phase 6: Evaluation
├── phase6b_data_quality_comparison.py # Phase 6b: Quality analysis
├── phase7_export_for_r.py        # Phase 7: R export
├── analysis_report_final.Rmd     # Comprehensive report
├── project_presentation.Rmd      # Presentation slides
├── requirements.txt              # Python dependencies
├── install_r_packages.R          # R dependencies
├── data/
│   ├── raw/                      # Raw input data
│   └── processed/                # Processed datasets
├── embeddings/                   # Generated embeddings
├── output/                       # Pipeline outputs
├── results/                      # Evaluation results
└── logs/                         # Execution logs
Advanced Usage
Custom Configuration
Edit config.py to customize:


```python
# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight
# EMBEDDING_MODEL = "all-mpnet-base-v2"  # More accurate, slower

# Similarity threshold
SIMILARITY_THRESHOLD = 0.65  # Increase for stricter matching

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.85
LOW_CONFIDENCE_THRESHOLD = 0.50

# Batch processing
BATCH_SIZE = 64  # Reduce if memory issues
```
## Processing New Incidents
To classify new incidents:

Add new incidents to data/raw/new_incidents.csv
Run: python main_pipeline.py --start 5 --end 5
Results in: output/machine_processed_incidents.csv
Querying the Knowledge Graph
Open Neo4j Browser (http://localhost:7474) and run:

```cypher

// Find incidents using specific hate terms
MATCH (i:Incident)-[r:USES_TERM]->(t:Term {term: "swastika"})
RETURN i.description, r.detection_method, i.date

// Find most common hate categories
MATCH (i:Incident)-[:SIMILAR_TO]->(t:Term)-[:BELONGS_TO]->(c:Category)
RETURN c.name, COUNT(i) AS incident_count
ORDER BY incident_count DESC

// Find term co-occurrences
MATCH (t1:Term)-[r:CO_OCCURS_WITH]-(t2:Term)
WHERE r.count >= 5
RETURN t1.term, t2.term, r.count
ORDER BY r.count DESC
```

## Performance Benchmarks
Tested on MacBook Pro (M1, 16GB RAM):

Phase	Time	Memory
Phase 1: Data Ingestion	2 sec	500 MB
Phase 2: Glossary	1 sec	200 MB
Phase 3: Embeddings	8 min	4 GB
Phase 4: Knowledge Graph	45 sec	2 GB
Phase 5: Processing	3 min	3 GB
Phase 6: Evaluation	10 sec	1 GB
Phase 7: Export	15 sec	2 GB
Total	~12 min	4 GB peak

##Citation
If you use this pipeline in your research or work, please cite:

Howk, R. (2025). Knowledge Graph-Enhanced Hate Incident Classification:
Scaling Intelligence in an Age of Exponential Hate. 
CUNY DATA 607 Final Project.

##License
This project is for educational purposes as part of CUNY DATA 607.

## Contact
Author: Randy Howk
Course: CUNY DATA 607 - Data Acquisition and Management
Project: Final Project - Fall 2025

For questions or issues, please open an issue on the GitHub repository.

## Acknowledgments
Anti-Defamation League (ADL) for the hate terms glossary
Neo4j for graph database technology
Sentence Transformers for semantic embedding models
CUNY DATA 607 course instructors and peers