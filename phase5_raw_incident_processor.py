"""
Phase 5: Raw Incident Processing Pipeline
-----------------------------------------
This module:
1. Loads raw incident reports from the past year
2. Uses the trained knowledge graph to process incidents
3. Applies embedding similarity matching
4. Queries Neo4j for contextual classification
5. Outputs structured incident classifications
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import *
from utils import *

logger = setup_logging(__name__)

# ============================================================
# RAW INCIDENT PROCESSOR
# ============================================================

class RawIncidentProcessor:
    """Processes raw incidents using trained knowledge graph"""
    
    def __init__(self, neo4j_uri: str = NEO4J_URI, 
                 neo4j_user: str = NEO4J_USER,
                 neo4j_password: str = NEO4J_PASSWORD):
        
        # Connect to Neo4j
        logger.info("Connecting to Neo4j knowledge graph...")
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Load glossary embeddings
        self.glossary_embeddings = load_numpy(GLOSSARY_EMBEDDINGS_PATH)
        self.df_glossary = load_csv(PROCESSED_DATA_DIR / "glossary_final.csv")
        
        logger.info("✓ Processor initialized")
    
    def __del__(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
    
    # ---------------------------------------------------------
    # Load Raw Incidents
    # ---------------------------------------------------------
    def load_raw_incidents(self, filepath: Path) -> pd.DataFrame:
        """Load raw incident data"""
        logger.info(f"Loading raw incidents from {filepath}...")
        
        df_raw = load_csv(filepath)
        
        # Standardize columns (same as Phase 1)
        if 'Audit/HEAT Map Text' in df_raw.columns:
            df_raw = df_raw.rename(columns={
                'Date of Incident': 'date',
                'Case Number': 'case_number',
                'HEAT Map ID': 'heat_map_id',
                'Audit/HEAT Map Text': 'description',
                'City of Incident': 'city',
                'State of Incident': 'state',
                'Type of Attack': 'attack_type',
                'Ideology': 'ideology',
                'Group': 'group'
            })
        
        # Create incident_id
        if 'incident_id' not in df_raw.columns:
            if 'heat_map_id' in df_raw.columns:
                df_raw['incident_id'] = df_raw['heat_map_id']
            elif 'case_number' in df_raw.columns:
                df_raw['incident_id'] = df_raw['case_number']
            else:
                df_raw['incident_id'] = range(1, len(df_raw) + 1)
        
        # Clean text
        df_raw['description'] = df_raw['description'].apply(clean_text)
        
        # Create location field
        df_raw['location'] = df_raw.apply(
            lambda row: f"{row.get('city', '')}, {row.get('state', '')}".strip(', '),
            axis=1
        )
        
        logger.info(f"✓ Loaded {len(df_raw)} raw incidents")
        
        return df_raw
    
    # ---------------------------------------------------------
    # Generate Embeddings for Raw Incidents
    # ---------------------------------------------------------
    def embed_raw_incidents(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Generate embeddings for raw incident descriptions"""
        logger.info("Generating embeddings for raw incidents...")
        
        texts = df_raw['description'].astype(str).tolist()
        texts = [text[:EMBEDDING_MAX_LENGTH] for text in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        logger.info(f"✓ Generated embeddings for {len(embeddings)} incidents")
        
        return embeddings
    
    # ---------------------------------------------------------
    # Similarity Matching
    # ---------------------------------------------------------
    def match_to_glossary(self, embeddings: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Match incident embeddings to glossary terms"""
        logger.info("Matching incidents to glossary terms...")
        
        # Compute similarity
        sims = cosine_similarity(embeddings, self.glossary_embeddings)
        
        # Get top K matches
        top_matches = np.argsort(-sims, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(sims, top_matches, axis=1)
        
        matches = []
        for i in range(len(embeddings)):
            incident_matches = []
            for rank in range(top_k):
                glossary_idx = top_matches[i][rank]
                glossary_row = self.df_glossary.iloc[glossary_idx]
                
                incident_matches.append({
                    'term': glossary_row['term'],
                    'category': glossary_row['category'],
                    'score': float(top_scores[i][rank]),
                    'rank': rank + 1
                })
            
            matches.append(incident_matches)
        
        logger.info(f"✓ Generated matches for {len(matches)} incidents")
        
        return matches
    
    # ---------------------------------------------------------
    # Knowledge Graph Enhanced Classification
    # ---------------------------------------------------------
    def classify_with_graph(self, incident_text: str, top_terms: List[Dict]) -> Dict:
        """
        Use knowledge graph to enhance classification with contextual information
        
        Args:
            incident_text: The incident description
            top_terms: List of top matching terms from embedding similarity
            
        Returns:
            Enhanced classification with confidence and reasoning
        """
        # Get top term
        if not top_terms or len(top_terms) == 0:
            return {
                'predicted_term': 'Unknown',
                'predicted_category': 'Unknown',
                'confidence': 0.0,
                'method': 'fallback'
            }
        
        top_term = top_terms[0]
        
        # Query knowledge graph for contextual information
        with self.driver.session() as session:
            # Get term's co-occurrence patterns
            query = """
            MATCH (t:Term {term: $term})
            OPTIONAL MATCH (t)-[:CO_OCCURS_WITH]-(related:Term)
            OPTIONAL MATCH (t)-[:BELONGS_TO]->(c:Category)
            RETURN t.term AS term, 
                   t.category AS category,
                   t.incident_frequency AS frequency,
                   c.name AS category_name,
                   COLLECT(DISTINCT related.term) AS related_terms
            LIMIT 1
            """
            
            result = session.run(query, term=top_term['term'])
            record = result.single()
            
            if record:
                # Use graph context to adjust confidence
                base_confidence = top_term['score']
                
                # Boost confidence if term has high frequency in training data
                frequency_boost = min(0.1, record['frequency'] / 100) if record['frequency'] else 0
                
                # Boost if multiple top matches agree on category
                category_agreement = sum(
                    1 for t in top_terms[:3] if t['category'] == top_term['category']
                ) / min(3, len(top_terms))
                category_boost = (category_agreement - 0.33) * 0.1
                
                adjusted_confidence = min(1.0, base_confidence + frequency_boost + category_boost)
                
                return {
                    'predicted_term': top_term['term'],
                    'predicted_category': top_term['category'],
                    'confidence': adjusted_confidence,
                    'base_similarity': base_confidence,
                    'frequency_in_training': record['frequency'] or 0,
                    'related_terms': record['related_terms'][:5] if record['related_terms'] else [],
                    'method': 'graph_enhanced',
                    'secondary_matches': [
                        {'term': t['term'], 'score': t['score']} 
                        for t in top_terms[1:3]
                    ]
                }
        
        # Fallback to embedding similarity only
        return {
            'predicted_term': top_term['term'],
            'predicted_category': top_term['category'],
            'confidence': top_term['score'],
            'base_similarity': top_term['score'],
            'method': 'embedding_only'
        }
    
    # ---------------------------------------------------------
    # Process All Raw Incidents
    # ---------------------------------------------------------
    def process_incidents(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Process all raw incidents through the pipeline"""
        logger.info("Processing raw incidents through knowledge graph pipeline...")
        
        # Generate embeddings
        embeddings = self.embed_raw_incidents(df_raw)
        
        # Match to glossary
        all_matches = self.match_to_glossary(embeddings, top_k=5)
        
        # Classify with graph enhancement
        classifications = []
        tracker = ProgressTracker(len(df_raw), "Graph-enhanced classification")
        
        for idx, (_, row) in enumerate(df_raw.iterrows()):
            tracker.update()
            
            classification = self.classify_with_graph(
                row['description'],
                all_matches[idx]
            )
            
            classification['incident_id'] = row['incident_id']
            classifications.append(classification)
        
        tracker.finish()
        
        # Create results DataFrame
        df_classifications = pd.DataFrame(classifications)
        
        # Merge with original data
        df_processed = df_raw.merge(
            df_classifications,
            on='incident_id',
            how='left'
        )
        
        logger.info(f"✓ Processed {len(df_processed)} incidents")
        
        return df_processed
    
    # ---------------------------------------------------------
    # Generate Processing Report
    # ---------------------------------------------------------
    def generate_report(self, df_processed: pd.DataFrame) -> Dict:
        """Generate summary report of processing"""
        logger.info("Generating processing report...")
        
        report = {
            'total_incidents': len(df_processed),
            'confidence_distribution': {
                'high': (df_processed['confidence'] >= HIGH_CONFIDENCE_THRESHOLD).sum(),
                'medium': ((df_processed['confidence'] >= LOW_CONFIDENCE_THRESHOLD) & 
                          (df_processed['confidence'] < HIGH_CONFIDENCE_THRESHOLD)).sum(),
                'low': (df_processed['confidence'] < LOW_CONFIDENCE_THRESHOLD).sum()
            },
            'category_distribution': df_processed['predicted_category'].value_counts().to_dict(),
            'avg_confidence': float(df_processed['confidence'].mean()),
            'method_distribution': df_processed['method'].value_counts().to_dict()
        }
        
        print("\n" + "="*60)
        print("PROCESSING REPORT")
        print("="*60)
        print(f"\nTotal Incidents Processed: {report['total_incidents']:,}")
        print(f"Average Confidence: {report['avg_confidence']:.3f}")
        
        print("\nConfidence Distribution:")
        print(f"  High (≥{HIGH_CONFIDENCE_THRESHOLD}): {report['confidence_distribution']['high']:,}")
        print(f"  Medium ({LOW_CONFIDENCE_THRESHOLD}-{HIGH_CONFIDENCE_THRESHOLD}): {report['confidence_distribution']['medium']:,}")
        print(f"  Low (<{LOW_CONFIDENCE_THRESHOLD}): {report['confidence_distribution']['low']:,}")
        
        print("\nTop 10 Predicted Categories:")
        for cat, count in sorted(report['category_distribution'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / report['total_incidents']) * 100
            print(f"  {cat}: {count:,} ({pct:.1f}%)")
        
        print(f"\nClassification Methods:")
        for method, count in report['method_distribution'].items():
            print(f"  {method}: {count:,}")
        
        return report

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_phase5_pipeline():
    """Execute complete Phase 5 raw incident processing pipeline"""
    print_phase_header(5, "Raw Incident Processing")
    
    try:
        # Initialize processor
        processor = RawIncidentProcessor()
        
        # Load raw incidents
        print("\n📥 Loading raw incidents from past year...")
        df_raw = processor.load_raw_incidents(RAW_INCIDENTS_LASTYEAR_PATH)
        
        # Process incidents
        print("\n🔄 Processing incidents with knowledge graph...")
        df_processed = processor.process_incidents(df_raw)
        
        # Generate report
        print("\n📊 Generating processing report...")
        report = processor.generate_report(df_processed)
        
        # Save outputs
        print("\n💾 Saving outputs...")
        save_csv(df_processed, MACHINE_PROCESSED_OUTPUT)
        
        report_path = RESULTS_DIR / "processing_report.json"
        save_json(report, report_path)
        
        print_phase_footer(5, "Raw Incident Processing")
        
        return {
            'processor': processor,
            'processed_data': df_processed,
            'report': report
        }
        
    except Exception as e:
        logger.error(f"✗ Phase 5 failed: {e}")
        raise

if __name__ == "__main__":
    results = run_phase5_pipeline()
    print("\n✓ Phase 5 complete. Raw incidents processed and ready for comparison.")
