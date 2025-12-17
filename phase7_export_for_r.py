"""
Phase 7: Export Data for R Analysis
-----------------------------------
This module:
1. Consolidates all results from previous phases
2. Prepares data in R-friendly formats
3. Exports embeddings for UMAP visualization
4. Creates metadata files for R scripts
5. Exports graph data for network visualization
"""

import pandas as pd
from datetime import datetime
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import networkx as nx

from config import *
from utils import *

logger = setup_logging(__name__)

# ============================================================
# R DATA EXPORTER
# ============================================================

class RDataExporter:
    """Exports all analysis results for R visualization and analysis"""
    
    def __init__(self):
        self.export_data = {}
        
    def load_all_results(self):
        """Load all results from previous phases"""
        logger.info("Loading all pipeline results...")
        
        # Phase 1-2: Glossary and incidents
        self.glossary = load_csv(PROCESSED_DATA_DIR / "glossary_final.csv")
        self.historical_incidents = load_csv(PROCESSED_HISTORICAL_PATH)
        
        # Phase 3: Embeddings and matches
        self.incident_embeddings = load_numpy(INCIDENT_EMBEDDINGS_PATH)
        self.glossary_embeddings = load_numpy(GLOSSARY_EMBEDDINGS_PATH)
        self.embedding_matches = load_csv(OUTPUT_DIR / "embedding_matches.csv")
        
        # Phase 5: Machine processed
        self.machine_processed = load_csv(MACHINE_PROCESSED_OUTPUT)
        
        # Phase 6: Comparison results
        self.comparison_results = load_csv(COMPARISON_RESULTS_PATH)
        self.evaluation_metrics = load_json(EVALUATION_METRICS_PATH)
        
        # Graph statistics
        graph_stats_path = RESULTS_DIR / "knowledge_graph_stats.json"
        if graph_stats_path.exists():
            self.graph_stats = load_json(graph_stats_path)
        else:
            self.graph_stats = {}
        
        logger.info("✓ All results loaded")
    
    # ---------------------------------------------------------
    # Create Structured Incidents Dataset
    # ---------------------------------------------------------
    def create_structured_incidents(self) -> pd.DataFrame:
        """
        Create comprehensive structured incidents dataset combining:
        - Original incident data
        - Machine predictions
        - Human labels (ground truth)
        - Confidence scores
        - Embedding-based matches
        """
        logger.info("Creating structured incidents dataset for R...")
        
        # Start with comparison results (has both machine and human)
        df_structured = self.comparison_results.copy()
        
        # Ensure we have the right columns
        required_cols = {
            'incident_id': 'incident_id',
            'description': 'description',
            'date': 'date',
            'location': 'location',
            'state': 'state',
            'predicted_category_machine': 'predicted_term_machine',
            'predicted_category_human': 'manual_term',
            'confidence': 'confidence'
        }
        
        # Rename for R compatibility (no underscores at start/end, clear names)
        rename_map = {}
        for old, new in required_cols.items():
            if old in df_structured.columns and old != new:
                rename_map[old] = new
        
        if rename_map:
            df_structured = df_structured.rename(columns=rename_map)
        
        # Add match correctness indicator
        if 'predicted_term_machine' in df_structured.columns and 'manual_term' in df_structured.columns:
            df_structured['is_correct'] = (
                df_structured['predicted_term_machine'] == df_structured['manual_term']
            ).astype(int)
        
        # Add confidence categories
        if 'confidence' in df_structured.columns:
            df_structured['confidence_level'] = pd.cut(
                df_structured['confidence'],
                bins=[0, LOW_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD, 1.0],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
        
        # Add temporal features if date exists
        if 'date' in df_structured.columns:
            df_structured['date'] = pd.to_datetime(df_structured['date'], errors='coerce')
            df_structured['year'] = df_structured['date'].dt.year
            df_structured['month'] = df_structured['date'].dt.month
            df_structured['quarter'] = df_structured['date'].dt.quarter
        
        # Clean text fields (remove very long descriptions for R)
        if 'description' in df_structured.columns:
            df_structured['description_short'] = df_structured['description'].str[:500]
        
        logger.info(f"✓ Created structured dataset with {len(df_structured)} incidents")
        
        return df_structured
    
    # ---------------------------------------------------------
    # Export Embeddings for UMAP
    # ---------------------------------------------------------
    def export_embeddings_for_umap(self):
        """Export embeddings in R-friendly format"""
        logger.info("Exporting embeddings for UMAP visualization...")
        
        # Save incident embeddings as CSV (R can read this)
        df_inc_embeddings = pd.DataFrame(
            self.incident_embeddings,
            columns=[f'dim_{i}' for i in range(self.incident_embeddings.shape[1])]
        )
        
        # Add incident IDs
        if len(self.comparison_results) == len(df_inc_embeddings):
            df_inc_embeddings['incident_id'] = self.comparison_results['incident_id'].values
        
        inc_emb_path = OUTPUT_DIR / "incident_embeddings_for_r.csv"
        save_csv(df_inc_embeddings, inc_emb_path)
        
        # Save glossary embeddings
        df_gloss_embeddings = pd.DataFrame(
            self.glossary_embeddings,
            columns=[f'dim_{i}' for i in range(self.glossary_embeddings.shape[1])]
        )
        
        # Add term names
        if len(self.glossary) == len(df_gloss_embeddings):
            df_gloss_embeddings['term'] = self.glossary['term'].values
            df_gloss_embeddings['category'] = self.glossary['category'].values
        
        gloss_emb_path = OUTPUT_DIR / "glossary_embeddings_for_r.csv"
        save_csv(df_gloss_embeddings, gloss_emb_path)
        
        logger.info("✓ Embeddings exported for R")
    
    # ---------------------------------------------------------
    # Export Confusion Matrix Data
    # ---------------------------------------------------------
    def export_confusion_matrix_data(self):
        """Export confusion matrix in long format for R"""
        logger.info("Exporting confusion matrix data...")
        
        # Get unique categories
        y_true = self.comparison_results['predicted_category_human']
        y_pred = self.comparison_results['predicted_category_machine']
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        categories = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=categories)
        
        # Convert to long format (better for ggplot2)
        cm_long = []
        for i, true_cat in enumerate(categories):
            for j, pred_cat in enumerate(categories):
                cm_long.append({
                    'true_category': true_cat,
                    'predicted_category': pred_cat,
                    'count': int(cm[i, j])
                })
        
        df_cm = pd.DataFrame(cm_long)
        cm_path = OUTPUT_DIR / "confusion_matrix_long.csv"
        save_csv(df_cm, cm_path)
        
        logger.info("✓ Confusion matrix exported")
    
    # ---------------------------------------------------------
    # Export Category Performance Data
    # ---------------------------------------------------------
    def export_category_performance(self):
        """Export per-category performance metrics"""
        logger.info("Exporting category performance data...")
        
        per_class = self.evaluation_metrics.get('per_class', {})
        
        # Extract class-specific metrics (exclude aggregates)
        category_metrics = []
        for category, metrics in per_class.items():
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                category_metrics.append({
                    'category': category,
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1-score', 0),
                    'support': metrics.get('support', 0)
                })
        
        df_cat_perf = pd.DataFrame(category_metrics)
        cat_perf_path = OUTPUT_DIR / "category_performance.csv"
        save_csv(df_cat_perf, cat_perf_path)
        
        logger.info("✓ Category performance exported")
    
    # ---------------------------------------------------------
    # Export Confidence Analysis
    # ---------------------------------------------------------
    def export_confidence_analysis(self):
        """Export confidence level analysis"""
        logger.info("Exporting confidence analysis...")
        
        if 'confidence' not in self.comparison_results.columns:
            logger.warning("No confidence data available")
            return
        
        df_conf = self.comparison_results[['incident_id', 'confidence']].copy()
        
        # Add correctness
        if 'predicted_category_machine' in self.comparison_results.columns:
            df_conf['is_correct'] = (
                self.comparison_results['predicted_category_machine'] == 
                self.comparison_results['predicted_category_human']
            ).astype(int)
        
        # Add confidence bins
        df_conf['confidence_bin'] = pd.cut(
            df_conf['confidence'],
            bins=10,
            labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)]
        )
        
        conf_path = OUTPUT_DIR / "confidence_analysis.csv"
        save_csv(df_conf, conf_path)
        
        logger.info("✓ Confidence analysis exported")
    
    # ---------------------------------------------------------
    # Export Temporal Analysis
    # ---------------------------------------------------------
    def export_temporal_analysis(self):
        """Export temporal trends in incidents"""
        logger.info("Exporting temporal analysis...")
        
        if 'date' not in self.comparison_results.columns:
            logger.warning("No date data available")
            return
        
        df_temporal = self.comparison_results[
            ['incident_id', 'date', 'predicted_category_machine', 
             'predicted_category_human', 'state']
        ].copy()
        
        df_temporal['date'] = pd.to_datetime(df_temporal['date'], errors='coerce')
        df_temporal = df_temporal.dropna(subset=['date'])
        
        # Add time features
        df_temporal['year'] = df_temporal['date'].dt.year
        df_temporal['month'] = df_temporal['date'].dt.month
        df_temporal['year_month'] = df_temporal['date'].dt.to_period('M').astype(str)
        df_temporal['quarter'] = df_temporal['date'].dt.quarter
        
        temporal_path = OUTPUT_DIR / "temporal_analysis.csv"
        save_csv(df_temporal, temporal_path)
        
        logger.info("✓ Temporal analysis exported")
    
    # ---------------------------------------------------------
    # Export Geographic Analysis
    # ---------------------------------------------------------
    def export_geographic_analysis(self):
        """Export geographic distribution of incidents"""
        logger.info("Exporting geographic analysis...")
        
        if 'state' not in self.comparison_results.columns:
            logger.warning("No geographic data available")
            return
        
        df_geo = self.comparison_results[
            ['incident_id', 'state', 'location', 'predicted_category_machine']
        ].copy()
        
        # State-level aggregation
        state_counts = df_geo.groupby(['state', 'predicted_category_machine']).size().reset_index(name='count')
        
        geo_path = OUTPUT_DIR / "geographic_analysis.csv"
        save_csv(state_counts, geo_path)
        
        logger.info("✓ Geographic analysis exported")
    
    # ---------------------------------------------------------
    # Export Graph Network Data
    # ---------------------------------------------------------
    def export_graph_network_data(self):
        """Export knowledge graph network data for R visualization"""
        logger.info("Exporting graph network data...")
        
        # Load graph export if available
        if not GRAPH_JSON_PATH.exists():
            logger.warning("No graph export found")
            return
        
        graph_data = load_json(GRAPH_JSON_PATH)
        
        # Create nodes dataframe
        nodes = []
        for node in graph_data.get('nodes', [])[:500]:  # Limit for visualization
            node_props = node.get('properties', {})
            nodes.append({
                'id': node.get('id'),
                'label': ', '.join(node.get('labels', [])),
                'name': node_props.get('term', node_props.get('name', str(node.get('id')))),
                'category': node_props.get('category', 'Unknown')
            })
        
        df_nodes = pd.DataFrame(nodes)
        nodes_path = OUTPUT_DIR / "graph_nodes.csv"
        save_csv(df_nodes, nodes_path)
        
        # Create edges dataframe
        edges = []
        for rel in graph_data.get('relationships', [])[:1000]:  # Limit for visualization
            edges.append({
                'source': rel.get('source'),
                'target': rel.get('target'),
                'type': rel.get('type'),
                'score': rel.get('properties', {}).get('score', 1.0)
            })
        
        df_edges = pd.DataFrame(edges)
        
        # Filter edges to only those with nodes in our limited node set
        node_ids = set(df_nodes['id'])
        df_edges = df_edges[
            df_edges['source'].isin(node_ids) & 
            df_edges['target'].isin(node_ids)
        ]
        
        edges_path = OUTPUT_DIR / "graph_edges.csv"
        save_csv(df_edges, edges_path)
        
        logger.info(f"✓ Graph network data exported: {len(df_nodes)} nodes, {len(df_edges)} edges")
    
    # ---------------------------------------------------------
    # Create R Metadata File
    # ---------------------------------------------------------
    def create_r_metadata(self):
        """Create metadata file for R scripts"""
        logger.info("Creating R metadata file...")
        
        metadata = {
            'project_name': 'Knowledge Graph Hate Incident Classification',
            'generated_at': datetime.now().isoformat(),
            'python_version': '3.8+',
            'model_name': EMBEDDING_MODEL_NAME,
            
            # File paths (relative to output directory)
            'files': {
                'structured_incidents': 'structured_incidents.csv',
                'incident_embeddings': 'incident_embeddings_for_r.csv',
                'glossary_embeddings': 'glossary_embeddings_for_r.csv',
                'confusion_matrix': 'confusion_matrix_long.csv',
                'category_performance': 'category_performance.csv',
                'confidence_analysis': 'confidence_analysis.csv',
                'temporal_analysis': 'temporal_analysis.csv',
                'geographic_analysis': 'geographic_analysis.csv',
                'graph_nodes': 'graph_nodes.csv',
                'graph_edges': 'graph_edges.csv',
                'evaluation_report': '../results/evaluation_report.txt',
                'evaluation_metrics': '../results/evaluation_metrics.json'
            },
            
            # Dataset statistics
            'statistics': {
                'total_incidents': len(self.comparison_results),
                'total_glossary_terms': len(self.glossary),
                'n_categories': self.comparison_results['predicted_category_human'].nunique(),
                'accuracy': self.evaluation_metrics.get('accuracy', 0),
                'f1_macro': self.evaluation_metrics.get('f1_macro', 0),
                'embedding_dimensions': self.incident_embeddings.shape[1]
            },
            
            # Thresholds
            'thresholds': {
                'high_confidence': HIGH_CONFIDENCE_THRESHOLD,
                'low_confidence': LOW_CONFIDENCE_THRESHOLD,
                'similarity': SIMILARITY_THRESHOLD
            },
            
            # Graph statistics
            'graph_stats': self.graph_stats
        }
        
        save_json(metadata, R_METADATA_PATH)
        logger.info("✓ R metadata created")
        
        return metadata
    
    # ---------------------------------------------------------
    # Main Export Pipeline
    # ---------------------------------------------------------
    def export_all(self):
        """Execute complete export pipeline"""
        logger.info("Starting R export pipeline...")
        
        # Load all results
        self.load_all_results()
        
        # Create structured incidents
        df_structured = self.create_structured_incidents()
        save_csv(df_structured, STRUCTURED_INCIDENTS_OUTPUT)
        
        # Export various datasets
        self.export_embeddings_for_umap()
        self.export_confusion_matrix_data()
        self.export_category_performance()
        self.export_confidence_analysis()
        self.export_temporal_analysis()
        self.export_geographic_analysis()
        self.export_graph_network_data()
        
        # Create metadata
        metadata = self.create_r_metadata()
        
        logger.info("✓ R export pipeline complete")
        
        return metadata

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_phase7_pipeline():
    """Execute complete Phase 7 R export pipeline"""
    print_phase_header(7, "R Export and Preparation")
    
    try:
        # Initialize exporter
        exporter = RDataExporter()
        
        # Export all data
        print("\n📦 Exporting data for R analysis...")
        metadata = exporter.export_all()
        
        # Summary
        print("\n📊 Export Summary:")
        print(f"  Total incidents: {metadata['statistics']['total_incidents']:,}")
        print(f"  Categories: {metadata['statistics']['n_categories']}")
        print(f"  Overall accuracy: {metadata['statistics']['accuracy']:.3f}")
        print(f"  Files exported: {len(metadata['files'])}")
        
        print("\n📁 Exported files in output/ directory:")
        for file_key, file_name in metadata['files'].items():
            print(f"  - {file_name}")
        
        print_phase_footer(7, "R Export and Preparation")
        
        return {
            'exporter': exporter,
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"✗ Phase 7 failed: {e}")
        raise

if __name__ == "__main__":
    results = run_phase7_pipeline()
    print("\n✓ Phase 7 complete. Data ready for R analysis.")
