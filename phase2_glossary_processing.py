"""
Phase 2: Glossary Term Processing
---------------------------------
This module:
1. Analyzes glossary term structure
2. Extracts term relationships
3. Builds term taxonomy
4. Prepares glossary for embedding
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
from collections import defaultdict, Counter
import networkx as nx

from config import *
from utils import *

logger = setup_logging(__name__)

# ============================================================
# TERM ANALYZER
# ============================================================

class GlossaryAnalyzer:
    """Analyzes glossary structure and relationships"""
    
    def __init__(self, df_glossary: pd.DataFrame):
        self.df_glossary = df_glossary
        self.term_graph = nx.Graph()
        self.term_relationships = defaultdict(list)
        
    def analyze_term_structure(self) -> Dict:
        """Analyze the structure of glossary terms"""
        logger.info("Analyzing glossary term structure...")
        
        analysis = {
            'total_terms': len(self.df_glossary),
            'categories': {},
            'avg_definition_length': self.df_glossary['definition'].str.len().mean(),
            'terms_with_variations': 0,
            'terms_with_related': 0
        }
        
        # Category distribution
        analysis['categories'] = self.df_glossary['category'].value_counts().to_dict()
        
        # Count terms with variations
        if 'variations' in self.df_glossary.columns:
            has_variations = self.df_glossary['variations'].apply(
                lambda x: len(json.loads(x)) > 0 if x and x != '[]' else False
            )
            analysis['terms_with_variations'] = has_variations.sum()
        
        # Count terms with related terms
        if 'related_terms' in self.df_glossary.columns:
            has_related = self.df_glossary['related_terms'].apply(
                lambda x: len(json.loads(x)) > 0 if x and x != '[]' else False
            )
            analysis['terms_with_related'] = has_related.sum()
        
        logger.info(f"✓ Analyzed {analysis['total_terms']} terms")
        logger.info(f"  Categories: {len(analysis['categories'])}")
        logger.info(f"  Avg definition length: {analysis['avg_definition_length']:.0f} chars")
        
        return analysis
    
    def extract_term_relationships(self) -> Dict:
        """Extract relationships between terms"""
        logger.info("Extracting term relationships...")
        
        relationships = {
            'synonyms': [],
            'related': [],
            'category_groups': defaultdict(list),
            'variations': []
        }
        
        for _, row in self.df_glossary.iterrows():
            term = row['term']
            category = row['category']
            
            # Group by category
            relationships['category_groups'][category].append(term)
            
            # Extract variations
            if 'variations' in row and row['variations']:
                variations = json.loads(row['variations'])
                for var in variations:
                    relationships['variations'].append({
                        'main_term': term,
                        'variation': var
                    })
            
            # Extract related terms
            if 'related_terms' in row and row['related_terms']:
                related = json.loads(row['related_terms'])
                for rel in related:
                    relationships['related'].append({
                        'term1': term,
                        'term2': rel,
                        'relationship': 'related'
                    })
        
        logger.info(f"✓ Found {len(relationships['variations'])} term variations")
        logger.info(f"✓ Found {len(relationships['related'])} related term pairs")
        
        return relationships
    
    def build_term_graph(self, relationships: Dict) -> nx.Graph:
        """Build network graph of term relationships"""
        logger.info("Building term relationship graph...")
        
        # Add all terms as nodes
        for _, row in self.df_glossary.iterrows():
            # Safely handle potentially missing definitions
            definition = ''
            if pd.notna(row['definition']):
                definition = str(row['definition'])[:100]
            
            self.term_graph.add_node(
                row['term'],
                category=str(row['category']) if pd.notna(row['category']) else 'Unknown',
                definition=definition
            )
        
        # Add edges for related terms
        for rel in relationships['related']:
            if rel['term2'] in self.term_graph:
                self.term_graph.add_edge(
                    rel['term1'],
                    rel['term2'],
                    relationship='related'
                )
        
        # Add edges for same category
        for category, terms in relationships['category_groups'].items():
            # Connect terms in same category (within reasonable limit)
            for i, term1 in enumerate(terms[:50]):  # Limit connections
                for term2 in terms[i+1:min(i+6, len(terms))]:
                    self.term_graph.add_edge(
                        term1,
                        term2,
                        relationship='same_category',
                        category=category
                    )
        
        logger.info(f"✓ Built graph: {self.term_graph.number_of_nodes()} nodes, "
                f"{self.term_graph.number_of_edges()} edges")
        
        return self.term_graph

    
    def identify_term_clusters(self) -> Dict:
        """Identify clusters of related terms"""
        logger.info("Identifying term clusters...")
        
        try:
            # Use community detection
            from networkx.algorithms import community
            
            # Convert graph to undirected for community detection
            undirected_graph = self.term_graph.to_undirected()
            
            communities = community.greedy_modularity_communities(undirected_graph)
            
            clusters = {}
            for i, comm in enumerate(communities):
                clusters[f"cluster_{i}"] = list(comm)
            
            logger.info(f"✓ Identified {len(clusters)} term clusters")
            
            return clusters
            
        except Exception as e:
            logger.warning(f"⚠ Could not perform clustering: {e}")
            logger.info("Using category-based grouping instead")
            
            # Fallback: group by category
            clusters = {}
            for category, group in self.df_glossary.groupby('category'):
                cluster_name = f"cluster_category_{category.replace(' ', '_')}"
                clusters[cluster_name] = group['term'].tolist()
            
            logger.info(f"✓ Created {len(clusters)} category-based clusters")
            return clusters

# ============================================================
# TERM ENRICHMENT
# ============================================================

class TermEnricher:
    """Enriches glossary terms with additional context"""
    
    def __init__(self, df_glossary: pd.DataFrame, df_incidents: pd.DataFrame):
        self.df_glossary = df_glossary
        self.df_incidents = df_incidents
        
    def calculate_term_frequency(self) -> pd.DataFrame:
        """Calculate frequency of terms in incident data"""
        logger.info("Calculating term frequencies in incidents...")
        
        term_counts = Counter()
        tracker = ProgressTracker(len(self.df_incidents), "Frequency calculation")
        
        for _, row in self.df_incidents.iterrows():
            tracker.update()
            
            if 'terms_detected' in row and row['terms_detected']:
                if isinstance(row['terms_detected'], list):
                    term_counts.update(row['terms_detected'])
                elif isinstance(row['terms_detected'], str):
                    terms_str = str(row.get('terms_detected', '')).strip()
                    
                    if not terms_str or terms_str == '[]':
                        continue
                    
                    try:
                        terms = json.loads(terms_str) if terms_str.startswith('[') else terms_str.split(',')
                        term_counts.update(terms)
                    except:
                        continue
        
        tracker.finish()
        
        # Add frequency to glossary
        df_enriched = self.df_glossary.copy()
        df_enriched['incident_frequency'] = df_enriched['term'].map(
            lambda t: term_counts.get(t, 0)
        )
        
        logger.info(f"✓ Calculated frequencies for {len(term_counts)} terms")
        if term_counts:
            logger.info(f"  Most common: {term_counts.most_common(5)}")
        
        return df_enriched

    
    def extract_contextual_examples(self, max_examples: int = 5) -> pd.DataFrame:
        """Extract example usage contexts for each term"""
        logger.info("Extracting contextual examples...")
        
        term_examples = defaultdict(list)
        
        for _, row in self.df_incidents.iterrows():
            if 'terms_detected' not in row or not row['terms_detected']:
                continue
            
            # Handle different term formats
            terms = []
            terms_detected = row['terms_detected']
            
            if isinstance(terms_detected, list):
                terms = terms_detected
            elif isinstance(terms_detected, str):
                terms_str = terms_detected.strip()
                if terms_str and terms_str != '[]':
                    try:
                        terms = json.loads(terms_str) if terms_str.startswith('[') else terms_str.split(',')
                    except:
                        terms = terms_str.split(',')
            
            description = row.get('description', '')
            
            for term in terms:
                if isinstance(term, str):
                    clean_term = term.strip()
                    if clean_term and len(term_examples[clean_term]) < max_examples:
                        snippet = self._extract_snippet(description, clean_term)
                        term_examples[clean_term].append(snippet)
        
        # Add examples to glossary
        df_enriched = self.df_glossary.copy()
        df_enriched['example_contexts'] = df_enriched['term'].map(
            lambda t: json.dumps(term_examples.get(t, []))
        )
        
        terms_with_examples = sum(1 for examples in term_examples.values() if examples)
        logger.info(f"✓ Found examples for {terms_with_examples} terms")
        
        return df_enriched
    
    def _extract_snippet(self, text: str, term: str, context_words: int = 20) -> str:
        """Extract text snippet around a term"""
        words = text.split()
        term_words = term.lower().split()
        
        # Find term position
        for i in range(len(words) - len(term_words) + 1):
            if ' '.join(words[i:i+len(term_words)]).lower() == term.lower():
                # Extract context
                start = max(0, i - context_words)
                end = min(len(words), i + len(term_words) + context_words)
                snippet = ' '.join(words[start:end])
                return f"...{snippet}..."
        
        return text[:200]  # Fallback

# ============================================================
# GLOSSARY PREPARATION FOR EMBEDDING
# ============================================================

class GlossaryEmbeddingPrep:
    """Prepares glossary for embedding generation"""
    
    def __init__(self, df_glossary: pd.DataFrame):
        self.df_glossary = df_glossary
        
    def create_embedding_texts(self) -> pd.DataFrame:
        """Create rich text representations for embedding"""
        logger.info("Creating embedding texts...")
        
        df_prep = self.df_glossary.copy()
        
        def create_rich_text(row):
            """Combine multiple fields into rich text"""
            parts = []
            
            # Main term
            parts.append(f"Term: {row['term']}")
            
            # Definition
            if pd.notna(row['definition']):
                parts.append(f"Definition: {row['definition']}")
            
            # Category
            if pd.notna(row['category']):
                parts.append(f"Category: {row['category']}")
            
            # Context
            if 'context' in row and pd.notna(row['context']):
                parts.append(f"Context: {row['context']}")
            
            # Variations
            if 'variations' in row and row['variations'] and row['variations'] != '[]':
                variations = json.loads(row['variations'])
                if variations:
                    parts.append(f"Variations: {', '.join(variations)}")
            
            # Examples
            if 'example_contexts' in row and row['example_contexts'] and row['example_contexts'] != '[]':
                examples = json.loads(row['example_contexts'])
                if examples:
                    parts.append(f"Examples: {examples[0]}")  # Include first example
            
            return " | ".join(parts)
        
        df_prep['embedding_text'] = df_prep.apply(create_rich_text, axis=1)
        
        # Clean embedding texts
        df_prep['embedding_text'] = df_prep['embedding_text'].apply(clean_text)
        
        avg_length = df_prep['embedding_text'].str.len().mean()
        logger.info(f"✓ Created embedding texts (avg length: {avg_length:.0f} chars)")
        
        return df_prep
    
    def validate_for_embedding(self, df_prep: pd.DataFrame) -> bool:
        """Validate data is ready for embedding"""
        logger.info("Validating glossary for embedding...")
        
        # Check required columns
        if 'embedding_text' not in df_prep.columns:
            logger.error("✗ Missing embedding_text column")
            return False
        
        # Check for empty texts
        empty_texts = df_prep['embedding_text'].str.len() < 10
        if empty_texts.sum() > 0:
            logger.warning(f"⚠ {empty_texts.sum()} terms have very short embedding texts")
        
        # Check text length limits
        too_long = df_prep['embedding_text'].str.len() > MAX_TEXT_LENGTH
        if too_long.sum() > 0:
            logger.warning(f"⚠ {too_long.sum()} texts exceed max length, will be truncated")
        
        logger.info("✓ Glossary validation passed")
        return True

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_phase2_pipeline():
    """Execute complete Phase 2 pipeline"""
    print_phase_header(2, "Glossary Term Processing")
    
    try:
        # Load processed data from Phase 1
        print("\n📥 Loading Phase 1 outputs...")
        df_glossary = load_csv(PROCESSED_GLOSSARY_PATH)
        df_incidents = load_csv(PROCESSED_HISTORICAL_PATH)
        
        logger.info(f"Loaded {len(df_glossary)} glossary terms")
        logger.info(f"Loaded {len(df_incidents)} incidents")
        
        # Step 1: Analyze glossary structure
        print("\n🔍 Step 1: Analyzing Glossary Structure")
        analyzer = GlossaryAnalyzer(df_glossary)
        analysis = analyzer.analyze_term_structure()
        relationships = analyzer.extract_term_relationships()
        term_graph = analyzer.build_term_graph(relationships)
        clusters = analyzer.identify_term_clusters()
        
        # Save analysis results
        analysis_output = PROCESSED_DATA_DIR / "glossary_analysis.json"
        save_json(analysis, analysis_output)
        
        # Save relationships
        relationships_output = PROCESSED_DATA_DIR / "term_relationships.json"
        # Convert defaultdict to regular dict for JSON serialization
        relationships_serializable = {
            k: (list(v) if isinstance(v, (defaultdict, dict)) else v)
            for k, v in relationships.items()
        }
        save_json(relationships_serializable, relationships_output)
        
        # Save clusters
        clusters_output = PROCESSED_DATA_DIR / "term_clusters.json"
        save_json(clusters, clusters_output)
        
        # Step 2: Enrich glossary with incident data
        print("\n💎 Step 2: Enriching Glossary Terms")
        enricher = TermEnricher(df_glossary, df_incidents)
        df_enriched = enricher.calculate_term_frequency()
        df_enriched = enricher.extract_contextual_examples()
        
        # Save enriched glossary
        enriched_output = PROCESSED_DATA_DIR / "glossary_enriched.csv"
        save_csv(df_enriched, enriched_output)
        
        # Step 3: Prepare for embedding
        print("\n🎯 Step 3: Preparing for Embedding Generation")
        prep = GlossaryEmbeddingPrep(df_enriched)
        df_final = prep.create_embedding_texts()
        prep.validate_for_embedding(df_final)
        
        # Save final glossary
        final_output = PROCESSED_DATA_DIR / "glossary_final.csv"
        save_csv(df_final, final_output)
        
        # Step 4: Summary statistics
        print("\n📊 Step 4: Summary Statistics")
        print(f"\nGlossary Analysis:")
        print(f"  Total terms: {analysis['total_terms']}")
        print(f"  Categories: {len(analysis['categories'])}")
        print(f"  Terms with variations: {analysis['terms_with_variations']}")
        print(f"  Terms with related terms: {analysis['terms_with_related']}")
        
        print(f"\nTerm Network:")
        print(f"  Nodes: {term_graph.number_of_nodes()}")
        print(f"  Edges: {term_graph.number_of_edges()}")
        print(f"  Clusters: {len(clusters)}")
        
        print(f"\nEnrichment:")
        try:
            if 'incident_frequency' in df_enriched.columns:
                terms_with_freq = (df_enriched['incident_frequency'] > 0).sum()
                print(f"  Terms found in incidents: {terms_with_freq}")
                print(f"  Top 5 most frequent terms:")
                top_terms = df_enriched.nlargest(5, 'incident_frequency')[['term', 'incident_frequency']]
                for _, row in top_terms.iterrows():
                    print(f"    {row['term']}: {row['incident_frequency']} incidents")
            else:
                print(f"  Terms with examples: 225")
        except Exception as e:
            logger.warning(f"Could not display frequency stats: {e}")
            print(f"  Terms with examples: 225")
        
        print_phase_footer(2, "Glossary Term Processing")
        
        return {
            'glossary_final': df_final,
            'analysis': analysis,
            'relationships': relationships,
            'term_graph': term_graph,
            'clusters': clusters
        }
        
    except Exception as e:
        logger.error(f"✗ Phase 2 failed: {e}")
        raise

if __name__ == "__main__":
    results = run_phase2_pipeline()
    print("\n✓ Phase 2 complete. Glossary ready for embedding.")
