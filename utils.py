"""
Utility functions for the Hate Incident Classification System
"""
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re

# Import configuration
from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(name: str = __name__, log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = LOG_FILE
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ============================================================
# FILE I/O OPERATIONS
# ============================================================

def load_json(filepath: Path) -> Dict:
    """Load JSON file safely"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✓ Loaded JSON from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"✗ File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"✗ Invalid JSON in {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Error loading JSON from {filepath}: {e}")
        raise

def save_json(data: Dict, filepath: Path, indent: int = 2) -> None:
    """Save data to JSON file with numpy type handling"""
    import numpy as np
    
    def convert_types(obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types before saving
        converted_data = convert_types(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=indent, ensure_ascii=False)
        logger.info(f"✓ Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"✗ Error saving JSON to {filepath}: {e}")
        raise


def load_csv(filepath: Path, **kwargs) -> pd.DataFrame:
    """Load CSV file with error handling"""
    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"✓ Loaded {len(df):,} records from {filepath}")
        return df
    except FileNotFoundError:
        logger.error(f"✗ File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"✗ Error loading CSV from {filepath}: {e}")
        raise

def save_csv(df: pd.DataFrame, filepath: Path, **kwargs) -> None:
    """Save DataFrame to CSV"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False, **kwargs)
        logger.info(f"✓ Saved {len(df):,} records to {filepath}")
    except Exception as e:
        logger.error(f"✗ Error saving CSV to {filepath}: {e}")
        raise

def load_numpy(filepath: Path) -> np.ndarray:
    """Load numpy array"""
    try:
        arr = np.load(filepath)
        logger.info(f"✓ Loaded numpy array {arr.shape} from {filepath}")
        return arr
    except Exception as e:
        logger.error(f"✗ Error loading numpy array from {filepath}: {e}")
        raise

def save_numpy(arr: np.ndarray, filepath: Path) -> None:
    """Save numpy array"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, arr)
        logger.info(f"✓ Saved numpy array {arr.shape} to {filepath}")
    except Exception as e:
        logger.error(f"✗ Error saving numpy array to {filepath}: {e}")
        raise

# ============================================================
# TEXT PROCESSING
# ============================================================

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.,;:!?\-\'\"()]', ' ', text)
    
    # Remove extra whitespace again
    text = ' '.join(text.split())
    
    return text.strip()

def extract_terms_from_text(text: str, glossary_terms: List[str]) -> List[str]:
    """Extract hate terms found in text"""
    text_lower = text.lower()
    found_terms = []
    
    for term in glossary_terms:
        term_lower = term.lower()
        # Use word boundaries for better matching
        pattern = r'\b' + re.escape(term_lower) + r'\b'
        if re.search(pattern, text_lower):
            found_terms.append(term)
    
    return found_terms

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Simple Jaccard similarity for text comparison"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                       name: str = "DataFrame") -> bool:
    """Validate DataFrame has required columns"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        logger.error(f"✗ {name} missing columns: {missing_cols}")
        return False
    
    logger.info(f"✓ {name} validation passed")
    return True

def check_file_exists(filepath: Path, file_type: str = "file") -> bool:
    """Check if file exists and log result"""
    if filepath.exists():
        logger.info(f"✓ Found {file_type}: {filepath}")
        return True
    else:
        logger.warning(f"⚠ Missing {file_type}: {filepath}")
        return False

# ============================================================
# PROGRESS TRACKING
# ============================================================

class ProgressTracker:
    """Simple progress tracker for pipelines"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        if self.current % max(1, self.total // 10) == 0:
            self._log_progress()
    
    def _log_progress(self):
        """Log current progress"""
        pct = (self.current / self.total) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        logger.info(f"{self.description}: {self.current}/{self.total} "
                   f"({pct:.1f}%) - {rate:.1f} items/sec")
    
    def finish(self):
        """Log completion"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"✓ {self.description} complete: {self.current} items "
                   f"in {elapsed:.1f}s")

# ============================================================
# STATISTICS HELPERS
# ============================================================

def calculate_accuracy(y_true: List, y_pred: List) -> float:
    """Calculate classification accuracy"""
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists must have same length")
    
    if len(y_true) == 0:
        return 0.0
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format float as percentage string"""
    return f"{value * 100:.{decimals}f}%"

def print_phase_header(phase_num: int, phase_name: str):
    """Print formatted phase header"""
    header = f"\n{'='*60}\nPHASE {phase_num}: {phase_name.upper()}\n{'='*60}\n"
    print(header)
    logger.info(f"Starting Phase {phase_num}: {phase_name}")

def print_phase_footer(phase_num: int, phase_name: str):
    """Print formatted phase footer"""
    footer = f"\n{'='*60}\n✓ PHASE {phase_num} COMPLETE: {phase_name.upper()}\n{'='*60}\n"
    print(footer)
    logger.info(f"Completed Phase {phase_num}: {phase_name}")

# ============================================================
# DATA SUMMARY
# ============================================================

def summarize_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> Dict:
    """Generate summary statistics for a DataFrame"""
    summary = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    logger.info(f"\n{name} Summary:")
    logger.info(f"  Rows: {summary['rows']:,}")
    logger.info(f"  Columns: {summary['columns']}")
    logger.info(f"  Memory: {summary['memory_usage_mb']:.2f} MB")
    
    return summary

# Export all utilities
__all__ = [
    'setup_logging',
    'load_json', 'save_json',
    'load_csv', 'save_csv',
    'load_numpy', 'save_numpy',
    'clean_text', 'extract_terms_from_text', 'calculate_text_similarity',
    'validate_dataframe', 'check_file_exists',
    'ProgressTracker',
    'calculate_accuracy', 'format_percentage',
    'print_phase_header', 'print_phase_footer',
    'summarize_dataframe'
]
