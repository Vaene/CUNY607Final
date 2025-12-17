"""
Phase 3: Embedding Pipeline Module
----------------------------------
This module:
1. Loads processed incident reports and glossary terms
2. Generates sentence embeddings using transformer models
3. Stores embeddings locally
4. Performs similarity search between incidents and glossary terms
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from config import *
from utils import *
from datetime import datetime

logger = setup_logging(__name__)

# ============================================================
# EMBEDDING CONFIG
# ============================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    incident_csv: Path
    glossary_csv: Path
    text_column_incident: str = "description"
    text_column_glossary: str = "embedding_text"
    model_name: str = EMBEDDING_MODEL_NAME
    save_directory: Path = EMBEDDINGS_DIR
    batch_size: int = EMBEDDING_BATCH_SIZE
    max_length: int = EMBEDDING_MAX_LENGTH
    use_gpu: bool = True

# ============================================================
# MAIN EMBEDDING PIPELINE
# ============================================================

class EmbeddingPipeline:
    """Main pipeline for generating and managing embeddings"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading embedding model: {config.model_name}")
        self.model = SentenceTransformer(config.model_name, device=self.device)
        logger.info(f"✓ Model loaded: {config.model_name}")
        
        self.df_incidents = None
        self.df_glossary = None
        self.incident_embeddings = None
        self.glossary_embeddings = None

    # ---------------------------------------------------------
    # Load Data
    # ---------------------------------------------------------
    def load_data(self):
        """Load incident and glossary data"""
        print("\n📥 Loading data...")
        
        self.df_incidents = load_csv(self.config.incident_csv)
        self.df_glossary = load_csv(self.config.glossary_csv)

        # Validate columns
        if self.config.text_column_incident not in self.df_incidents:
            raise ValueError(
                f"Incident CSV missing `{self.config.text_column_incident}` column. "
                f"Available: {list(self.df_incidents.columns)}"
            )

        if self.config.text_column_glossary not in self.df_glossary:
            raise ValueError(
                f"Glossary CSV missing `{self.config.text_column_glossary}` column. "
                f"Available: {list(self.df_glossary.columns)}"
            )

        logger.info(f"✓ Loaded {len(self.df_incidents):,} incidents")
        logger.info(f"✓ Loaded {len(self.df_glossary):,} glossary terms")

    # ---------------------------------------------------------
    # Generate Embeddings
    # ---------------------------------------------------------
    def embed_incidents(self):
        """Generate embeddings for incident descriptions"""
        print("\n🔍 Encoding incident descriptions...")
        
        texts = self.df_incidents[self.config.text_column_incident].astype(str).tolist()
        
        # Truncate long texts
        texts = [text[:self.config.max_length] for text in texts]
        
        self.incident_embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for better similarity
        )
        
        logger.info(f"✓ Generated {len(self.incident_embeddings)} incident embeddings")
        logger.info(f"  Shape: {self.incident_embeddings.shape}")

    def embed_glossary(self):
        """Generate embeddings for glossary definitions"""
        print("\n🔍 Encoding glossary terms...")
        
        texts = self.df_glossary[self.config.text_column_glossary].astype(str).tolist()
        
        # Truncate long texts
        texts = [text[:self.config.max_length] for text in texts]
        
        self.glossary_embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        logger.info(f"✓ Generated {len(self.glossary_embeddings)} glossary embeddings")
        logger.info(f"  Shape: {self.glossary_embeddings.shape}")

    # ---------------------------------------------------------
    # Save embeddings
    # ---------------------------------------------------------
    def save_embeddings(self):
        """Save embeddings to disk"""
        print("\n💾 Saving embeddings...")
        
        self.config.save_directory.mkdir(parents=True, exist_ok=True)

        save_numpy(self.incident_embeddings, INCIDENT_EMBEDDINGS_PATH)
        save_numpy(self.glossary_embeddings, GLOSSARY_EMBEDDINGS_PATH)
        
        # Save metadata
        metadata = {
            'model_name': self.config.model_name,
            'incident_count': len(self.incident_embeddings),
            'glossary_count': len(self.glossary_embeddings),
            'embedding_dim': self.incident_embeddings.shape[1],
            'generated_at': datetime.now().isoformat(),
            'device': self.device
        }
        save_json(metadata, EMBEDDING_METADATA_PATH)
        
        logger.info(f"✓ Embeddings saved to {self.config.save_directory}")

    # ---------------------------------------------------------
    # Load embeddings
    # ---------------------------------------------------------
    def load_embeddings(self):
        """Load pre-computed embeddings from disk"""
        print("\n📤 Loading embeddings from disk...")
        
        self.incident_embeddings = load_numpy(INCIDENT_EMBEDDINGS_PATH)
        self.glossary_embeddings = load_numpy(GLOSSARY_EMBEDDINGS_PATH)
        
        logger.info("✓ Embeddings loaded")

    # ---------------------------------------------------------
    # Similarity Search
    # ---------------------------------------------------------
    def match_incidents_to_glossary(self, top_k: int = 3) -> List[Dict]:
        """
        For each incident embedding, find the top K most similar glossary terms.
        
        Args:
            top_k: Number of top matches to return per incident
            
        Returns:
            List of match results with incident info and top matches
        """
        print(f"\n📡 Computing cosine similarity (top-{top_k} matches per incident)...")
        
        # Compute similarity matrix
        sims = cosine_similarity(self.incident_embeddings, self.glossary_embeddings)
        
        # Get top K matches
        top_matches = np.argsort(-sims, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(sims, top_matches, axis=1)

        results = []
        for i, incident_row in self.df_incidents.iterrows():
            matches = []
            for rank in range(top_k):
                glossary_index = top_matches[i][rank]
                glossary_row = self.df_glossary.iloc[glossary_index]
                
                matches.append({
                    "term": glossary_row.get("term", f"index_{glossary_index}"),
                    "category": glossary_row.get("category", "unknown"),
                    "score": float(top_scores[i][rank]),
                    "rank": rank + 1
                })

            results.append({
                "incident_id": incident_row.get("incident_id", i),
                "incident_text": incident_row[self.config.text_column_incident],
                "date": incident_row.get("date", ""),
                "location": incident_row.get("location", ""),
                "matches": matches,
                "top_match_term": matches[0]["term"],
                "top_match_score": matches[0]["score"],
                "top_match_category": matches[0]["category"]
            })
        
        logger.info(f"✓ Generated {len(results)} incident-to-term matches")
        
        return results
    
    def analyze_similarity_distribution(self, results: List[Dict]) -> Dict:
        """Analyze the distribution of similarity scores"""
        print("\n📊 Analyzing similarity score distribution...")
        
        all_scores = [r['top_match_score'] for r in results]
        
        analysis = {
            'mean_similarity': np.mean(all_scores),
            'median_similarity': np.median(all_scores),
            'std_similarity': np.std(all_scores),
            'min_similarity': np.min(all_scores),
            'max_similarity': np.max(all_scores),
            'high_confidence': sum(1 for s in all_scores if s >= HIGH_CONFIDENCE_THRESHOLD),
            'medium_confidence': sum(1 for s in all_scores if LOW_CONFIDENCE_THRESHOLD <= s < HIGH_CONFIDENCE_THRESHOLD),
            'low_confidence': sum(1 for s in all_scores if s < LOW_CONFIDENCE_THRESHOLD)
        }
        
        print(f"\nSimilarity Score Distribution:")
        print(f"  Mean: {analysis['mean_similarity']:.3f}")
        print(f"  Median: {analysis['median_similarity']:.3f}")
        print(f"  Std Dev: {analysis['std_similarity']:.3f}")
        print(f"  Range: [{analysis['min_similarity']:.3f}, {analysis['max_similarity']:.3f}]")
        print(f"\nConfidence Levels:")
        print(f"  High (≥{HIGH_CONFIDENCE_THRESHOLD}): {analysis['high_confidence']}")
        print(f"  Medium ({LOW_CONFIDENCE_THRESHOLD}-{HIGH_CONFIDENCE_THRESHOLD}): {analysis['medium_confidence']}")
        print(f"  Low (<{LOW_CONFIDENCE_THRESHOLD}): {analysis['low_confidence']}")
        
        return analysis

# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def run_phase3_pipeline():
    """Execute complete Phase 3 embedding pipeline"""
    print_phase_header(3, "Embedding Generation")
    
    try:
        # Setup configuration
        config = EmbeddingConfig(
            incident_csv=PROCESSED_HISTORICAL_PATH,
            glossary_csv=PROCESSED_DATA_DIR / "glossary_final.csv",
            text_column_incident="description",
            text_column_glossary="embedding_text"
        )
        
        # Initialize pipeline
        pipeline = EmbeddingPipeline(config)
        
        # Execute steps
        pipeline.load_data()
        pipeline.embed_incidents()
        pipeline.embed_glossary()
        pipeline.save_embeddings()
        
        # Perform similarity matching
        results = pipeline.match_incidents_to_glossary(top_k=3)
        
        # Analyze results
        analysis = pipeline.analyze_similarity_distribution(results)
        
        # Save results
        results_df = pd.DataFrame([
            {
                'incident_id': r['incident_id'],
                'incident_text': r['incident_text'][:200],  # Truncate for readability
                'top_match_term': r['top_match_term'],
                'top_match_category': r['top_match_category'],
                'top_match_score': r['top_match_score'],
                'second_match_term': r['matches'][1]['term'] if len(r['matches']) > 1 else '',
                'second_match_score': r['matches'][1]['score'] if len(r['matches']) > 1 else 0.0
            }
            for r in results
        ])
        
        results_path = OUTPUT_DIR / "embedding_matches.csv"
        save_csv(results_df, results_path)
        
        # Save full results as JSON
        results_json_path = OUTPUT_DIR / "embedding_matches_full.json"
        save_json({'results': results, 'analysis': analysis}, results_json_path)
        
        print_phase_footer(3, "Embedding Generation")
        
        return {
            'pipeline': pipeline,
            'results': results,
            'analysis': analysis
        }
        
    except Exception as e:
        logger.error(f"✗ Phase 3 failed: {e}")
        raise

# ============================================================
# PIPELINE WRAPPER FUNCTIONS
# ============================================================

def run_phase3_pipeline():
    """Execute complete Phase 3 embedding pipeline"""
    from config import (
        PROCESSED_HISTORICAL_PATH, 
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        INCIDENT_EMBEDDINGS_PATH,
        GLOSSARY_EMBEDDINGS_PATH,
        EMBEDDING_METADATA_PATH
    )
    from utils import print_phase_header, print_phase_footer, save_csv, save_json
    import pandas as pd
    from datetime import datetime
    
    print_phase_header(3, "Embedding Generation")
    
    try:
        # Setup configuration
        config = EmbeddingConfig(
            incident_csv=PROCESSED_HISTORICAL_PATH,
            glossary_csv=PROCESSED_DATA_DIR / "glossary_final.csv",
            text_column_incident="description",
            text_column_glossary="embedding_text"
        )
        
        # Initialize pipeline
        pipeline = EmbeddingPipeline(config)
        
        # Execute steps
        pipeline.load_data()
        pipeline.embed_incidents()
        pipeline.embed_glossary()
        pipeline.save_embeddings()
        
        # Perform similarity matching
        results = pipeline.match_incidents_to_glossary(top_k=3)
        
        # Analyze results
        analysis = pipeline.analyze_similarity_distribution(results)
        
        # Save results
        results_df = pd.DataFrame([
            {
                'incident_id': r['incident_id'],
                'incident_text': r['incident_text'][:200],
                'top_match_term': r['top_match_term'],
                'top_match_category': r['top_match_category'],
                'top_match_score': r['top_match_score'],
                'second_match_term': r['matches'][1]['term'] if len(r['matches']) > 1 else '',
                'second_match_score': r['matches'][1]['score'] if len(r['matches']) > 1 else 0.0
            }
            for r in results
        ])
        
        results_path = OUTPUT_DIR / "embedding_matches.csv"
        save_csv(results_df, results_path)
        
        # Save full results as JSON
        results_json_path = OUTPUT_DIR / "embedding_matches_full.json"
        save_json({'results': results, 'analysis': analysis}, results_json_path)
        
        print_phase_footer(3, "Embedding Generation")
        
        return {
            'pipeline': pipeline,
            'results': results,
            'analysis': analysis
        }
        
    except Exception as e:
        logger.error(f"✗ Phase 3 failed: {e}")
        raise


def run_full_embedding_pipeline(config: EmbeddingConfig):
    """
    Convenience function for running complete embedding pipeline
    (Backward compatibility wrapper)
    """
    pipeline = EmbeddingPipeline(config)
    
    pipeline.load_data()
    pipeline.embed_incidents()
    pipeline.embed_glossary()
    pipeline.save_embeddings()
    
    print("🎉 Phase 3 complete!")
    return pipeline

def run_phase3_pipeline():
    """Execute complete Phase 3 embedding pipeline"""
    from config import (
        PROCESSED_HISTORICAL_PATH,
        PROCESSED_DATA_DIR
    )
    from utils import print_phase_header, print_phase_footer
    
    print_phase_header(3, "Embedding Generation")
    
    try:
        # Setup configuration
        config = EmbeddingConfig(
            incident_csv=PROCESSED_HISTORICAL_PATH,
            glossary_csv=PROCESSED_DATA_DIR / "glossary_final.csv",
            text_column_incident="description",
            text_column_glossary="embedding_text"
        )
        
        # Run the pipeline
        pipeline = run_full_embedding_pipeline(config)
        
        print_phase_footer(3, "Embedding Generation")
        
        return {'pipeline': pipeline}
        
    except Exception as e:
        logger.error(f"✗ Phase 3 failed: {e}")
        raise


def run_phase3_pipeline():
    """Execute Phase 3: Embedding Generation"""
    from config import PROCESSED_HISTORICAL_PATH, PROCESSED_DATA_DIR
    from utils import print_phase_header, print_phase_footer
    
    print_phase_header(3, "Embedding Generation")
    
    try:
        config = EmbeddingConfig(
            incident_csv=PROCESSED_HISTORICAL_PATH,
            glossary_csv=PROCESSED_DATA_DIR / "glossary_final.csv",
            text_column_incident="description",
            text_column_glossary="embedding_text"
        )
        
        pipeline = run_full_embedding_pipeline(config)
        
        print_phase_footer(3, "Embedding Generation")
        return {'pipeline': pipeline}
    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        raise


if __name__ == "__main__":
    results = run_phase3_pipeline()
    print("\n✓ Phase 3 complete. Embeddings ready for knowledge graph construction.")
