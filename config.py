"""
Configuration file for the Hate Incident Classification System
Knowledge Graph-Enhanced Classification of Hate Terminology
"""
import os
from pathlib import Path
from dataclasses import dataclass

# ============================================================
# PROJECT STRUCTURE
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create all directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODELS_DIR, EMBEDDINGS_DIR, OUTPUT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA FILES
# ============================================================

# Input Data
HATE_TERMS_DICT_PATH = RAW_DATA_DIR / "adl_hate_terms.json"
HISTORICAL_INCIDENTS_PATH = RAW_DATA_DIR / "historical_incidents_4years.csv"
RAW_INCIDENTS_LASTYEAR_PATH = RAW_DATA_DIR / "raw_incidents_lastyear.csv"
HUMAN_PROCESSED_LASTYEAR_PATH = RAW_DATA_DIR / "human_processed_lastyear.csv"

# Processed Data
PROCESSED_GLOSSARY_PATH = PROCESSED_DATA_DIR / "glossary_processed.csv"
PROCESSED_HISTORICAL_PATH = PROCESSED_DATA_DIR / "historical_processed.csv"
TRAINING_DATA_PATH = PROCESSED_DATA_DIR / "training_data.csv"

# ============================================================
# COLUMN MAPPINGS (Based on your CSV structure)
# ============================================================

# Incident Data Columns
INCIDENT_COLUMNS = {
    'date': 'Date of Incident',
    'case_number': 'Case Number',
    'heat_map_id': 'HEAT Map ID',
    'description': 'Audit/HEAT Map Text',
    'city': 'City of Incident',
    'state': 'State of Incident',
    'attack_type': 'Type of Attack',
    'ideology': 'Ideology',
    'group': 'Group'
}

# Standardized column names for processing
STANDARD_COLUMNS = {
    'incident_id': 'incident_id',
    'date': 'date',
    'description': 'description',
    'location': 'location',
    'state': 'state',
    'attack_type': 'attack_type',
    'ideology': 'ideology',
    'group': 'group',
    'terms_detected': 'terms_detected',
    'manual_classification': 'manual_classification'
}

# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Embedding Model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_LENGTH = 512

# Text Processing
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 5000

# Similarity Thresholds
SIMILARITY_THRESHOLD = 0.70
HIGH_CONFIDENCE_THRESHOLD = 0.85
LOW_CONFIDENCE_THRESHOLD = 0.50

# ============================================================
# NEO4J CONFIGURATION
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Graph Export
GRAPH_EXPORT_PATH = OUTPUT_DIR / "knowledge_graph_export.graphml"
GRAPH_JSON_PATH = OUTPUT_DIR / "knowledge_graph_export.json"

# ============================================================
# PROCESSING PARAMETERS
# ============================================================

BATCH_SIZE = 100
MAX_WORKERS = 4
RANDOM_STATE = 42

# Term Frequency Filters
MIN_TERM_FREQUENCY = 2
MIN_INCIDENT_COUNT = 5

# ============================================================
# OUTPUT PATHS
# ============================================================

# Embedding Outputs
INCIDENT_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "incident_embeddings.npy"
GLOSSARY_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "glossary_embeddings.npy"
EMBEDDING_METADATA_PATH = EMBEDDINGS_DIR / "embedding_metadata.json"

# Model Outputs
MACHINE_PROCESSED_OUTPUT = OUTPUT_DIR / "machine_processed_incidents.csv"
STRUCTURED_INCIDENTS_OUTPUT = OUTPUT_DIR / "structured_incidents.csv"

# Comparison Outputs
COMPARISON_RESULTS_PATH = RESULTS_DIR / "comparison_results.csv"
EVALUATION_METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"
EVALUATION_REPORT_PATH = RESULTS_DIR / "evaluation_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

# R Analysis Outputs
R_INPUT_PATH = OUTPUT_DIR / "r_analysis_input.csv"
R_METADATA_PATH = OUTPUT_DIR / "r_metadata.json"

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = PROJECT_ROOT / "pipeline.log"

# ============================================================
# GLOSSARY TERM CATEGORIES (ADL Structure)
# ============================================================

HATE_TERM_CATEGORIES = [
    "White Supremacy",
    "Antisemitism",
    "Anti-LGBTQ+",
    "Anti-Muslim",
    "General Hate",
    "Symbols",
    "Numeric Codes",
    "Organizations",
    "Ideologies"
]

@dataclass
class PipelineConfig:
    """Main pipeline configuration dataclass"""
    use_gpu: bool = True
    verbose: bool = True
    save_intermediate: bool = True
    overwrite_existing: bool = False
    
    # Phase controls
    run_phase1: bool = True  # Data Ingestion
    run_phase2: bool = True  # Glossary Processing
    run_phase3: bool = True  # Embeddings
    run_phase4: bool = True  # Knowledge Graph
    run_phase5: bool = True  # Raw Processing
    run_phase6: bool = True  # Comparison
    run_phase7: bool = True  # R Export

# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()

print(f"✓ Configuration loaded. Project root: {PROJECT_ROOT}")
