"""
Prepare Incident Data for Pipeline
----------------------------------
Adapts raw and refined incident CSV files to the formats required by the pipeline.

This script handles:
1. Raw incidents (for processing in Phase 5)
2. Historical incidents (4 years of refined data for training in Phases 1-4)
3. Human-processed incidents (for comparison in Phase 6)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# COLUMN MAPPINGS
# ============================================================

# Raw incident format (from sample)
RAW_INCIDENT_COLUMNS = {
    'Date of Incident': 'date',
    'Case Number': 'case_number',
    'HEAT Map ID': 'heat_map_id',
    'Audit/HEAT Map Text': 'description',
    'City of Incident': 'city',
    'State of Incident': 'state',
    'Type of Attack': 'attack_type',
    'Ideology': 'ideology',
    'Group': 'group'
}

# Refined incident format (from sample)
REFINED_INCIDENT_COLUMNS = {
    'id': 'incident_id',
    'date': 'date',
    'city': 'city',
    'state': 'state',
    'type': 'type',
    'ideology': 'ideology',
    'subideology': 'subideology',
    'group': 'group',
    'description': 'description',
    'location_type': 'location_type',
    'type_of_attack': 'attack_type',
    'israel_zionism_related': 'israel_zionism_related'
}

# ============================================================
# DATA PROCESSING FUNCTIONS
# ============================================================

def clean_text(text):
    """Clean and normalize text fields"""
    if pd.isna(text) or text == '' or text == 'nan':
        return ''
    text = str(text).strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def standardize_date(date_str):
    """Standardize date format to YYYY-MM-DD"""
    if pd.isna(date_str) or date_str == '':
        return ''
    
    try:
        # Try parsing various date formats
        formats = [
            '%Y-%m-%d',      # 2023-09-30
            '%m/%d/%Y',      # 12/31/2024
            '%m/%d/%y',      # 12/31/24
            '%Y/%m/%d',      # 2023/09/30
            '%d/%m/%Y',      # 31/12/2024
        ]
        
        for fmt in formats:
            try:
                dt = pd.to_datetime(date_str, format=fmt)
                return dt.strftime('%Y-%m-%d')
            except:
                continue
        
        # If no format works, try pandas auto-parsing
        dt = pd.to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
        
    except Exception as e:
        print(f"Warning: Could not parse date '{date_str}': {e}")
        return str(date_str)

def extract_category_from_attack_type(attack_type: str) -> str:
    """
    Extract primary category from complex attack type strings
    e.g., "Antisemitic Incident:Vandalism" -> "Antisemitism"
    """
    if pd.isna(attack_type) or attack_type == '':
        return 'General Hate'
    
    attack_type = str(attack_type).lower()
    
    # Category mapping
    if 'antisemitic' in attack_type or 'antisemitism' in attack_type:
        return 'Antisemitism'
    elif 'lgbtq' in attack_type or 'anti-lgbtq' in attack_type:
        return 'Anti-LGBTQ+'
    elif 'white supremacist' in attack_type or 'white supremacy' in attack_type:
        return 'White Supremacy'
    elif 'islamophobic' in attack_type or 'anti-muslim' in attack_type:
        return 'Anti-Muslim'
    elif 'racist' in attack_type or 'racism' in attack_type:
        return 'Racism'
    else:
        return 'General Hate'

# ============================================================
# RAW INCIDENT PROCESSOR
# ============================================================

class RawIncidentProcessor:
    """Process raw incident data for Phase 5"""
    
    def __init__(self, input_file: str, output_file: str = None):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file) if output_file else None
        
    def process(self) -> pd.DataFrame:
        """Process raw incident CSV"""
        print(f"\n{'='*70}")
        print("PROCESSING RAW INCIDENTS")
        print(f"{'='*70}")
        print(f"Input: {self.input_file}")
        
        # Load data
        df = pd.read_csv(self.input_file)
        print(f"✓ Loaded {len(df):,} raw incidents")
        
        # Rename columns
        df = df.rename(columns=RAW_INCIDENT_COLUMNS)
        
        # Standardize dates
        print("Processing dates...")
        df['date'] = df['date'].apply(standardize_date)
        
        # Create incident_id if not exists
        if 'incident_id' not in df.columns:
            if 'heat_map_id' in df.columns:
                df['incident_id'] = df['heat_map_id'].astype(str)
            elif 'case_number' in df.columns:
                df['incident_id'] = df['case_number'].astype(str)
            else:
                df['incident_id'] = range(1, len(df) + 1)
        
        # Clean text fields
        print("Cleaning text fields...")
        text_columns = ['description', 'city', 'state', 'attack_type', 'ideology', 'group']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)
        
        # Create location field
        df['location'] = df.apply(
            lambda row: f"{row.get('city', '')}, {row.get('state', '')}".strip(', '),
            axis=1
        )
        
        # Filter out incidents with very short descriptions
        min_description_length = 20
        initial_count = len(df)
        df = df[df['description'].str.len() >= min_description_length]
        removed = initial_count - len(df)
        if removed > 0:
            print(f"⚠ Removed {removed} incidents with insufficient description")
        
        # Add metadata
        df['source'] = 'raw'
        df['processed_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Reorder columns for pipeline
        standard_columns = [
            'incident_id', 'date', 'description', 'city', 'state', 'location',
            'attack_type', 'ideology', 'group', 'case_number', 'heat_map_id',
            'source', 'processed_date'
        ]
        
        # Keep only columns that exist
        final_columns = [col for col in standard_columns if col in df.columns]
        df = df[final_columns]
        
        print(f"✓ Processed {len(df):,} raw incidents")
        
        # Save if output file specified
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.output_file, index=False)
            print(f"✓ Saved to: {self.output_file}")
        
        return df

# ============================================================
# REFINED INCIDENT PROCESSOR
# ============================================================

class RefinedIncidentProcessor:
    """Process refined/human-processed incident data"""
    
    def __init__(self, input_file: str, output_file: str = None, 
                 data_type: str = 'historical'):
        """
        Args:
            input_file: Path to refined CSV
            output_file: Output path
            data_type: 'historical' for training data or 'human_labeled' for comparison
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file) if output_file else None
        self.data_type = data_type
        
    def process(self) -> pd.DataFrame:
        """Process refined incident CSV"""
        print(f"\n{'='*70}")
        print(f"PROCESSING {self.data_type.upper()} INCIDENTS")
        print(f"{'='*70}")
        print(f"Input: {self.input_file}")
        
        # Load data
        df = pd.read_csv(self.input_file)
        print(f"✓ Loaded {len(df):,} incidents")
        
        # Rename columns to standard format
        df = df.rename(columns=REFINED_INCIDENT_COLUMNS)
        
        # Standardize dates
        print("Processing dates...")
        df['date'] = df['date'].apply(standardize_date)
        
        # Clean text fields
        print("Cleaning text fields...")
        text_columns = ['description', 'city', 'state', 'type', 'ideology', 
                       'subideology', 'group', 'location_type']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)
        
        # Create location field
        df['location'] = df.apply(
            lambda row: f"{row.get('city', '')}, {row.get('state', '')}".strip(', '),
            axis=1
        )
        
        # Ensure incident_id is string
        df['incident_id'] = df['incident_id'].astype(str)
        
        # Map attack_type to standardized format if missing
        if 'attack_type' not in df.columns and 'type' in df.columns:
            df['attack_type'] = df['type']
        
        # Extract primary category for machine learning
        if 'predicted_category' not in df.columns:
            if 'type' in df.columns:
                df['predicted_category'] = df['type'].apply(extract_category_from_attack_type)
            elif 'attack_type' in df.columns:
                df['predicted_category'] = df['attack_type'].apply(extract_category_from_attack_type)
        
        # For human-labeled data, this is the ground truth
        if self.data_type == 'human_labeled':
            # The 'type' or 'attack_type' column is the human classification
            if 'predicted_category' in df.columns:
                df['manual_term'] = df['predicted_category']
            elif 'type' in df.columns:
                df['manual_term'] = df['type']
        
        # Add terms_detected field (will be populated in Phase 1)
        if 'terms_detected' not in df.columns:
            df['terms_detected'] = ''
        
        # Filter out incidents with very short descriptions
        min_description_length = 20
        initial_count = len(df)
        df = df[df['description'].str.len() >= min_description_length]
        removed = initial_count - len(df)
        if removed > 0:
            print(f"⚠ Removed {removed} incidents with insufficient description")
        
        # Add metadata
        df['source'] = self.data_type
        df['processed_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Reorder columns for pipeline
        standard_columns = [
            'incident_id', 'date', 'description', 'city', 'state', 'location',
            'attack_type', 'ideology', 'subideology', 'group', 'location_type',
            'israel_zionism_related', 'predicted_category', 'terms_detected',
            'source', 'processed_date'
        ]
        
        # Add manual_term for human-labeled data
        if self.data_type == 'human_labeled':
            standard_columns.insert(standard_columns.index('predicted_category') + 1, 'manual_term')
        
        # Keep only columns that exist
        final_columns = [col for col in standard_columns if col in df.columns]
        df = df[final_columns]
        
        print(f"✓ Processed {len(df):,} incidents")
        
        # Save if output file specified
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.output_file, index=False)
            print(f"✓ Saved to: {self.output_file}")
        
        return df

# ============================================================
# DATA SPLITTER
# ============================================================

class DataSplitter:
    """Split data by date ranges for different pipeline phases"""
    
    @staticmethod
    def split_by_date(df: pd.DataFrame, 
                      historical_start: str = None,
                      historical_end: str = None,
                      lastyear_start: str = None,
                      lastyear_end: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into historical (training) and last year (test) sets
        
        Args:
            df: Input DataFrame
            historical_start: Start date for historical data (YYYY-MM-DD)
            historical_end: End date for historical data (YYYY-MM-DD)
            lastyear_start: Start date for last year data (YYYY-MM-DD)
            lastyear_end: End date for last year data (YYYY-MM-DD)
            
        Returns:
            Tuple of (historical_df, lastyear_df)
        """
        print(f"\n{'='*70}")
        print("SPLITTING DATA BY DATE")
        print(f"{'='*70}")
        
        # Convert date column to datetime
        df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        initial_count = len(df)
        df = df[df['date_dt'].notna()]
        removed = initial_count - len(df)
        if removed > 0:
            print(f"⚠ Removed {removed} incidents with invalid dates")
        
        # Determine date ranges if not provided
        if not all([historical_start, historical_end, lastyear_start, lastyear_end]):
            print("\nAuto-detecting date ranges...")
            min_date = df['date_dt'].min()
            max_date = df['date_dt'].max()
            print(f"Data range: {min_date.date()} to {max_date.date()}")
            
            # Last year: most recent 12 months
            lastyear_end = max_date
            lastyear_start = lastyear_end - pd.DateOffset(years=1)
            
            # Historical: everything before last year, up to 4 years before max
            historical_end = lastyear_start - pd.DateOffset(days=1)
            historical_start = historical_end - pd.DateOffset(years=4)
            
            print(f"\nHistorical: {historical_start.date()} to {historical_end.date()}")
            print(f"Last Year:  {lastyear_start.date()} to {lastyear_end.date()}")
        else:
            historical_start = pd.to_datetime(historical_start)
            historical_end = pd.to_datetime(historical_end)
            lastyear_start = pd.to_datetime(lastyear_start)
            lastyear_end = pd.to_datetime(lastyear_end)
        
        # Split data
        historical_mask = (df['date_dt'] >= historical_start) & (df['date_dt'] <= historical_end)
        lastyear_mask = (df['date_dt'] >= lastyear_start) & (df['date_dt'] <= lastyear_end)
        
        df_historical = df[historical_mask].copy()
        df_lastyear = df[lastyear_mask].copy()
        
        # Remove date_dt column
        df_historical = df_historical.drop(columns=['date_dt'])
        df_lastyear = df_lastyear.drop(columns=['date_dt'])
        
        print(f"\n✓ Historical incidents: {len(df_historical):,}")
        print(f"✓ Last year incidents: {len(df_lastyear):,}")
        
        return df_historical, df_lastyear

# ============================================================
# MAIN PREPARATION WORKFLOW
# ============================================================

def prepare_all_data(raw_file: str,
                     refined_file: str,
                     output_dir: str = 'data/raw',
                     historical_years: int = 4):
    """
    Complete data preparation workflow
    
    Args:
        raw_file: Path to raw incidents CSV (all dates)
        refined_file: Path to refined incidents CSV (all dates)
        output_dir: Output directory for prepared files
        historical_years: Number of years of historical data to use for training
    """
    
    print("\n" + "="*70)
    print("ADL INCIDENT DATA PREPARATION PIPELINE")
    print("="*70)
    print(f"\nRaw file: {raw_file}")
    print(f"Refined file: {refined_file}")
    print(f"Output directory: {output_dir}")
    print(f"Historical data: {historical_years} years")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # STEP 1: Process Refined Data (Full History)
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: Process Refined (Human-Processed) Data")
    print("="*70)
    
    refined_processor = RefinedIncidentProcessor(
        input_file=refined_file,
        output_file=None,
        data_type='refined'
    )
    df_refined = refined_processor.process()
    
    # Split into historical (training) and last year (test)
    df_historical, df_lastyear_human = DataSplitter.split_by_date(
        df_refined,
        historical_start=None,  # Will auto-detect
        historical_end=None,
        lastyear_start=None,
        lastyear_end=None
    )
    
    # Save historical data (4 years for training)
    historical_output = output_path / "historical_incidents_4years.csv"
    df_historical.to_csv(historical_output, index=False)
    print(f"\n✓ Saved historical training data: {historical_output}")
    
    # Save last year human-processed (for comparison)
    human_output = output_path / "human_processed_lastyear.csv"
    df_lastyear_human.to_csv(human_output, index=False)
    print(f"✓ Saved human-processed last year: {human_output}")
    
    # ========================================
    # STEP 2: Process Raw Data (Last Year Only)
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: Process Raw Data (Last Year)")
    print("="*70)
    
    raw_processor = RawIncidentProcessor(
        input_file=raw_file,
        output_file=None
    )
    df_raw = raw_processor.process()
    
    # Filter to last year only (matching human-processed dates)
    if len(df_lastyear_human) > 0:
        lastyear_start = pd.to_datetime(df_lastyear_human['date']).min()
        lastyear_end = pd.to_datetime(df_lastyear_human['date']).max()
        
        df_raw['date_dt'] = pd.to_datetime(df_raw['date'], errors='coerce')
        raw_mask = (df_raw['date_dt'] >= lastyear_start) & (df_raw['date_dt'] <= lastyear_end)
        df_raw_lastyear = df_raw[raw_mask].copy()
        df_raw_lastyear = df_raw_lastyear.drop(columns=['date_dt'])
    else:
        df_raw_lastyear = df_raw
    
    # Save raw last year data
    raw_output = output_path / "raw_incidents_lastyear.csv"
    df_raw_lastyear.to_csv(raw_output, index=False)
    print(f"\n✓ Saved raw incidents last year: {raw_output}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("DATA PREPARATION SUMMARY")
    print("="*70)
    
    print(f"\nFiles created in {output_path}:")
    print(f"  1. historical_incidents_4years.csv    ({len(df_historical):,} incidents)")
    print(f"     → Used for training in Phases 1-4")
    print(f"  2. raw_incidents_lastyear.csv         ({len(df_raw_lastyear):,} incidents)")
    print(f"     → Processed by machine in Phase 5")
    print(f"  3. human_processed_lastyear.csv       ({len(df_lastyear_human):,} incidents)")
    print(f"     → Ground truth for comparison in Phase 6")
    
    print(f"\nDate ranges:")
    if len(df_historical) > 0:
        hist_start = pd.to_datetime(df_historical['date']).min()
        hist_end = pd.to_datetime(df_historical['date']).max()
        print(f"  Historical: {hist_start.date()} to {hist_end.date()}")
    
    if len(df_lastyear_human) > 0:
        last_start = pd.to_datetime(df_lastyear_human['date']).min()
        last_end = pd.to_datetime(df_lastyear_human['date']).max()
        print(f"  Last year:  {last_start.date()} to {last_end.date()}")
    
    print("\n" + "="*70)
    print("✓ DATA PREPARATION COMPLETE")
    print("="*70)
    print("\nNext step: Run the main pipeline")
    print("  python main_pipeline.py")
    print("="*70 + "\n")
    
    return {
        'historical': df_historical,
        'raw_lastyear': df_raw_lastyear,
        'human_lastyear': df_lastyear_human
    }

# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare incident data for Knowledge Graph Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all data (auto-detect date ranges)
  python prepare_incident_data.py raw_incidents.csv refined_incidents.csv
  
  # Specify output directory
  python prepare_incident_data.py raw.csv refined.csv -o data/raw
  
  # Specify custom date ranges
  python prepare_incident_data.py raw.csv refined.csv \\
      --historical-start 2020-01-01 \\
      --historical-end 2023-12-31 \\
      --lastyear-start 2024-01-01 \\
      --lastyear-end 2024-12-31
        """
    )
    
    parser.add_argument(
        'raw_file',
        help='Path to raw incidents CSV file'
    )
    
    parser.add_argument(
        'refined_file',
        help='Path to refined/human-processed incidents CSV file'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default='data/raw',
        help='Output directory (default: data/raw)'
    )
    
    parser.add_argument(
        '--historical-years',
        type=int,
        default=4,
        help='Number of years of historical data (default: 4)'
    )
    
    parser.add_argument(
        '--historical-start',
        help='Start date for historical data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--historical-end',
        help='End date for historical data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--lastyear-start',
        help='Start date for last year data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--lastyear-end',
        help='End date for last year data (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    # Prepare data
    results = prepare_all_data(
        raw_file=args.raw_file,
        refined_file=args.refined_file,
        output_dir=args.output_dir,
        historical_years=args.historical_years
    )
    
    print("✓ Preparation successful!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
