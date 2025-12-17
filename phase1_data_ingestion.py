"""
Phase 1: Data Ingestion and Preprocessing
-----------------------------------------
This module:
1. Loads the 4 years of historical incident data
2. Loads the ADL hate terms glossary
3. Cleans and standardizes data formats
4. Validates data quality
5. Creates training datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

from config import *
from utils import *

logger = setup_logging(__name__)

# ============================================================
# GLOSSARY INGESTION
# ============================================================

class GlossaryIngestor:
    """Handles ingestion and processing of ADL hate terms glossary"""
    
    def __init__(self, glossary_path: Path):
        self.glossary_path = glossary_path
        self.glossary_data = None
        self.terms_dict = {}
        
    def load_glossary(self) -> Dict:
        """Load glossary from JSON file"""
        logger.info("Loading ADL hate terms glossary...")
        
        if not check_file_exists(self.glossary_path, "Glossary JSON"):
            raise FileNotFoundError(f"Glossary file not found: {self.glossary_path}")
        
        self.glossary_data = load_json(self.glossary_path)
        logger.info(f"✓ Loaded glossary with {len(self.glossary_data)} entries")
        
        return self.glossary_data
    
    def convert_to_dataframe(self) -> pd.DataFrame:
        """Convert glossary JSON to structured DataFrame"""
        logger.info("Converting glossary to DataFrame...")
        
        if self.glossary_data is None:
            self.load_glossary()
        
        # Parse glossary structure
        # Assuming structure: {term: {definition, category, variations, etc.}}
        records = []
        
        for term, details in self.glossary_data.items():
            record = {
                'term': term,
                'definition': details.get('definition', ''),
                'category': details.get('category', 'General Hate'),
                'variations': json.dumps(details.get('variations', [])),
                'related_terms': json.dumps(details.get('related_terms', [])),
                'context': details.get('context', ''),
                'severity': details.get('severity', 'medium')
            }
            records.append(record)
        
        df_glossary = pd.DataFrame(records)
        logger.info(f"✓ Created glossary DataFrame: {len(df_glossary)} terms")
        
        return df_glossary
    
    def create_term_dictionary(self, df_glossary: pd.DataFrame) -> Dict:
        """Create searchable term dictionary"""
        logger.info("Creating term dictionary...")
        
        self.terms_dict = {}
        
        for _, row in df_glossary.iterrows():
            term = row['term']
            self.terms_dict[term.lower()] = {
                'term': term,
                'definition': row['definition'],
                'category': row['category'],
                'variations': json.loads(row['variations']) if row['variations'] else []
            }
            
            # Add variations to dictionary
            variations = json.loads(row['variations']) if row['variations'] else []
            for var in variations:
                self.terms_dict[var.lower()] = {
                    'term': term,  # Point to main term
                    'definition': row['definition'],
                    'category': row['category'],
                    'is_variation': True
                }
        
        logger.info(f"✓ Dictionary contains {len(self.terms_dict)} searchable terms")
        return self.terms_dict
    
    def save_processed_glossary(self, df_glossary: pd.DataFrame, output_path: Path):
        """Save processed glossary"""
        save_csv(df_glossary, output_path)
        
        # Also save term dictionary
        dict_path = output_path.parent / "term_dictionary.json"
        save_json(self.terms_dict, dict_path)

# ============================================================
# INCIDENT DATA INGESTION
# ============================================================

class IncidentIngestor:
    """Handles ingestion and processing of incident reports"""
    
    def __init__(self, incident_path: Path, column_mapping: Dict = None):
        self.incident_path = incident_path
        self.column_mapping = column_mapping or INCIDENT_COLUMNS
        self.df_incidents = None
        
    def load_incidents(self) -> pd.DataFrame:
        """Load incident data from CSV"""
        logger.info(f"Loading incident data from {self.incident_path}...")
        
        if not check_file_exists(self.incident_path, "Incident CSV"):
            raise FileNotFoundError(f"Incident file not found: {self.incident_path}")
        
        # Load with proper encoding
        self.df_incidents = load_csv(self.incident_path, encoding='utf-8')
        
        logger.info(f"✓ Loaded {len(self.df_incidents)} incident records")
        logger.info(f"  Columns: {list(self.df_incidents.columns)}")
        
        return self.df_incidents
    
    def clean_incidents(self) -> pd.DataFrame:
        """Clean and preprocess incident data"""
        logger.info("Cleaning incident data...")
        
        df = self.df_incidents.copy()
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Clean text fields
        text_columns = ['description', 'attack_type', 'ideology', 'group']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)
        
        # Create location field
        df['location'] = df['city'].fillna('') + ', ' + df['state'].fillna('')
        df['location'] = df['location'].str.strip(', ')
        
        # Handle dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Create incident_id if not exists
        if 'incident_id' not in df.columns:
            if 'heat_map_id' in df.columns:
                df['incident_id'] = df['heat_map_id']
            elif 'case_number' in df.columns:
                df['incident_id'] = df['case_number']
            else:
                df['incident_id'] = range(1, len(df) + 1)
        
        # Remove empty descriptions
        initial_count = len(df)
        df = df[df['description'].str.len() >= MIN_TEXT_LENGTH]
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"  Removed {removed_count} incidents with insufficient text")
        
        logger.info(f"✓ Cleaned incidents: {len(df)} records remain")
        
        self.df_incidents = df
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        # Rename columns based on mapping
        rename_dict = {}
        for standard_name, original_name in self.column_mapping.items():
            if original_name in df.columns:
                rename_dict[original_name] = standard_name
        
        df = df.rename(columns=rename_dict)
        
        return df
    
    def extract_explicit_terms(self, df: pd.DataFrame, terms_dict: Dict) -> pd.DataFrame:
        """Extract hate terms explicitly mentioned in incident text"""
        logger.info("Extracting explicit term mentions...")
        
        all_terms = list(terms_dict.keys())
        tracker = ProgressTracker(len(df), "Term extraction")
        
        def extract_row_terms(text):
            tracker.update()
            return extract_terms_from_text(text, all_terms)
        
        df['terms_detected'] = df['description'].apply(extract_row_terms)
        
        tracker.finish()
        
        # Count incidents with detected terms
        with_terms = df['terms_detected'].apply(len) > 0
        logger.info(f"✓ Found explicit terms in {with_terms.sum()} / {len(df)} incidents")
        
        return df
    
    def save_processed_incidents(self, df: pd.DataFrame, output_path: Path):
        """Save processed incident data"""
        save_csv(df, output_path)

# ============================================================
# DATA VALIDATION
# ============================================================

class DataValidator:
    """Validates data quality and completeness"""
    
    @staticmethod
    def validate_glossary(df_glossary: pd.DataFrame) -> bool:
        """Validate glossary data"""
        logger.info("Validating glossary data...")
        
        required_cols = ['term', 'definition', 'category']
        if not validate_dataframe(df_glossary, required_cols, "Glossary"):
            return False
        
        # Check for duplicates
        duplicates = df_glossary['term'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"⚠ Found {duplicates} duplicate terms")
        
        # Check for missing definitions
        missing_defs = df_glossary['definition'].isna().sum()
        if missing_defs > 0:
            logger.warning(f"⚠ Found {missing_defs} terms without definitions")
        
        logger.info("✓ Glossary validation complete")
        return True
    
    @staticmethod
    def validate_incidents(df_incidents: pd.DataFrame) -> bool:
        """Validate incident data"""
        logger.info("Validating incident data...")
        
        required_cols = ['incident_id', 'description']
        if not validate_dataframe(df_incidents, required_cols, "Incidents"):
            return False
        
        # Check for duplicates
        duplicates = df_incidents['incident_id'].duplicated().sum()
        if duplicates > 0:
            logger.error(f"✗ Found {duplicates} duplicate incident IDs")
            return False
        
        # Check description quality
        empty_desc = df_incidents['description'].str.len() < MIN_TEXT_LENGTH
        if empty_desc.sum() > 0:
            logger.warning(f"⚠ Found {empty_desc.sum()} incidents with short descriptions")
        
        logger.info("✓ Incident validation complete")
        return True

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_phase1_pipeline():
    """Execute complete Phase 1 pipeline"""
    print_phase_header(1, "Data Ingestion & Preprocessing")
    
    try:
        # Step 1: Load and process glossary
        print("\n📖 Step 1: Processing Hate Terms Glossary")
        glossary_ingestor = GlossaryIngestor(HATE_TERMS_DICT_PATH)
        glossary_data = glossary_ingestor.load_glossary()
        df_glossary = glossary_ingestor.convert_to_dataframe()
        terms_dict = glossary_ingestor.create_term_dictionary(df_glossary)
        
        # Validate glossary
        DataValidator.validate_glossary(df_glossary)
        
        # Save processed glossary
        glossary_ingestor.save_processed_glossary(df_glossary, PROCESSED_GLOSSARY_PATH)
        
        # Step 2: Load and process historical incidents
        print("\n📊 Step 2: Processing Historical Incident Data (4 years)")
        incident_ingestor = IncidentIngestor(HISTORICAL_INCIDENTS_PATH)
        df_incidents = incident_ingestor.load_incidents()
        df_incidents = incident_ingestor.clean_incidents()
        
        # Extract explicit term mentions
        df_incidents = incident_ingestor.extract_explicit_terms(df_incidents, terms_dict)
        
        # Validate incidents
        DataValidator.validate_incidents(df_incidents)
        
        # Save processed incidents
        incident_ingestor.save_processed_incidents(df_incidents, PROCESSED_HISTORICAL_PATH)
        
        # Step 3: Create training dataset
        print("\n🎯 Step 3: Creating Training Dataset")
        df_training = df_incidents.copy()
        
        # Add metadata
        df_training['source'] = 'historical'
        df_training['processed_date'] = datetime.now().isoformat()
        
        save_csv(df_training, TRAINING_DATA_PATH)
        
        # Step 4: Generate summary report
        print("\n📈 Step 4: Summary Statistics")
        print(f"\nGlossary:")
        print(f"  Total terms: {len(df_glossary)}")
        print(f"  Categories: {df_glossary['category'].nunique()}")
        print(f"  Searchable variations: {len(terms_dict)}")
        
        print(f"\nIncidents:")
        print(f"  Total incidents: {len(df_incidents)}")
        print(f"  Date range: {df_incidents['date'].min()} to {df_incidents['date'].max()}")
        print(f"  States: {df_incidents['state'].nunique()}")
        print(f"  With explicit terms: {(df_incidents['terms_detected'].apply(len) > 0).sum()}")
        
        # Category distribution
        print(f"\nTerm Categories:")
        for cat, count in df_glossary['category'].value_counts().head(10).items():
            print(f"  {cat}: {count}")
        
        print_phase_footer(1, "Data Ingestion & Preprocessing")
        
        return {
            'glossary': df_glossary,
            'incidents': df_incidents,
            'terms_dict': terms_dict
        }
        
    except Exception as e:
        logger.error(f"✗ Phase 1 failed: {e}")
        raise

if __name__ == "__main__":
    results = run_phase1_pipeline()
    print("\n✓ Phase 1 complete. Data ready for Phase 2.")
