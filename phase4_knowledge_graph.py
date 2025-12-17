"""
Phase 4: Knowledge Graph Construction Module (Neo4j)
----------------------------------------------------
This module:
1. Connects to a Neo4j database
2. Creates nodes: Incidents, Terms, Categories, Locations, Groups
3. Creates relationships:
    - USES_TERM (explicit tagging)
    - SIMILAR_TO (embedding similarity match)
    - RELATED_TO (term-term similarity)
    - BELONGS_TO_CATEGORY
    - OCCURRED_IN
4. Trains the knowledge graph with historical data
"""

import pandas as pd
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from neo4j import GraphDatabase
from typing import List, Dict, Optional
import json
from tqdm import tqdm
import networkx as nx

from config import *
from utils import *

logger = setup_logging(__name__)

# ============================================================
# GRAPH CONFIG
# ============================================================

@dataclass
class GraphConfig:
    """Configuration for Knowledge Graph construction"""
    neo4j_uri: str = NEO4J_URI
    neo4j_user: str = NEO4J_USER
    neo4j_password: str = NEO4J_PASSWORD
    
    incident_csv: Path = PROCESSED_HISTORICAL_PATH
    glossary_csv: Path = PROCESSED_DATA_DIR / "glossary_final.csv"
    embedding_matches_csv: Path = OUTPUT_DIR / "embedding_matches.csv"
    
    incident_embeddings_path: Path = INCIDENT_EMBEDDINGS_PATH
    glossary_embeddings_path: Path = GLOSSARY_EMBEDDINGS_PATH
    
    similarity_threshold: float = SIMILARITY_THRESHOLD
    batch_size: int = BATCH_SIZE

# ============================================================
# NEO4J PIPELINE
# ============================================================

class Neo4jGraphPipeline:
    """Main pipeline for building knowledge graph in Neo4j"""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        
        # Connect to Neo4j
        logger.info(f"Connecting to Neo4j at {config.neo4j_uri}...")
        try:
            self.driver = GraphDatabase.driver(
                config.neo4j_uri,
                auth=(config.neo4j_user, config.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✓ Connected to Neo4j")
        except Exception as e:
            logger.error(f"✗ Failed to connect to Neo4j: {e}")
            raise
        
        # Load data
        self.df_incidents = load_csv(config.incident_csv)
        self.df_glossary = load_csv(config.glossary_csv)
        self.df_matches = load_csv(config.embedding_matches_csv)
        
        self.incident_embeddings = load_numpy(config.incident_embeddings_path)
        self.glossary_embeddings = load_numpy(config.glossary_embeddings_path)
        
        logger.info(f"✓ Loaded data: {len(self.df_incidents)} incidents, {len(self.df_glossary)} terms")
    
    def __del__(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
            logger.info("Neo4j connection closed")

    # ---------------------------------------------------
    # Clear Database
    # ---------------------------------------------------
    def clear_database(self):
        """Clear all data from Neo4j (use with caution!)"""
        print("\n🗑️  Clearing existing graph database...")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        logger.info("✓ Database cleared")

    # ---------------------------------------------------
    # Create Constraints and Indexes
    # ---------------------------------------------------
    def create_constraints(self):
        """Create database constraints and indexes for performance"""
        print("\n🔧 Creating constraints and indexes...")
        
        constraints = [
            "CREATE CONSTRAINT incident_id_unique IF NOT EXISTS FOR (i:Incident) REQUIRE i.incident_id IS UNIQUE",
            "CREATE CONSTRAINT term_unique IF NOT EXISTS FOR (t:Term) REQUIRE t.term IS UNIQUE",
            "CREATE CONSTRAINT category_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT location_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
            "CREATE CONSTRAINT group_unique IF NOT EXISTS FOR (g:Group) REQUIRE g.name IS UNIQUE",
        ]
        
        indexes = [
            "CREATE INDEX incident_date IF NOT EXISTS FOR (i:Incident) ON (i.date)",
            "CREATE INDEX term_category IF NOT EXISTS FOR (t:Term) ON (t.category)",
            "CREATE INDEX incident_state IF NOT EXISTS FOR (i:Incident) ON (i.state)",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
        
        logger.info("✓ Constraints and indexes created")

    # ---------------------------------------------------
    # Create Category Nodes
    # ---------------------------------------------------
    def create_category_nodes(self):
        """Create category nodes from glossary"""
        print("\n📂 Creating Category nodes...")
        
        categories = self.df_glossary['category'].dropna().unique()
        
        query = """
        UNWIND $categories AS cat
        MERGE (c:Category {name: cat})
        SET c.created_at = datetime()
        """
        
        with self.driver.session() as session:
            session.run(query, categories=list(categories))
        
        logger.info(f"✓ Created {len(categories)} Category nodes")

    # ---------------------------------------------------
    # Create Term Nodes
    # ---------------------------------------------------
    def create_term_nodes(self):
        """Create term nodes from glossary"""
        print("\n📖 Creating Term nodes...")
        
        rows = []
        for _, row in self.df_glossary.iterrows():
            # Skip rows with NaN term
            if pd.isna(row.get('term')) or not str(row.get('term')).strip():
                continue
            
            term_data = {
                'term': str(row['term']).strip(),
                'definition': str(row.get('definition', ''))[:500] if pd.notna(row.get('definition')) else '',
                'category': row.get('category', 'General Hate'),
                'incident_frequency': int(row.get('incident_frequency', 0))
            }
            
            # Add optional fields
            if 'severity' in row and pd.notna(row['severity']):
                term_data['severity'] = str(row['severity'])
            
            rows.append(term_data)
        
        query = """
        UNWIND $rows AS row
        MERGE (t:Term {term: row.term})
        SET t.definition = row.definition,
            t.category = row.category,
            t.incident_frequency = row.incident_frequency,
            t.severity = row.severity,
            t.created_at = datetime()
        """
        
        # Process in batches
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(rows), batch_size), desc="Creating terms"):
            batch = rows[i:i+batch_size]
            with self.driver.session() as session:
                session.run(query, rows=batch)
        
        logger.info(f"✓ Created {len(rows)} Term nodes")

    # ---------------------------------------------------
    # Link Terms to Categories
    # ---------------------------------------------------
    def create_term_category_relationships(self):
        """Create BELONGS_TO relationships between terms and categories"""
        print("\n🔗 Linking Terms to Categories...")
        
        query = """
        MATCH (t:Term)
        MATCH (c:Category {name: t.category})
        MERGE (t)-[:BELONGS_TO]->(c)
        """
        
        with self.driver.session() as session:
            session.run(query)
        
        logger.info("✓ Created BELONGS_TO relationships")

    # ---------------------------------------------------
    # Create Location Nodes
    # ---------------------------------------------------
    def create_location_nodes(self):
        """Create location nodes from incidents"""
        print("\n🗺️  Creating Location nodes...")
        
        # Get unique locations
        locations = []
        for _, row in self.df_incidents.iterrows():
            if pd.notna(row.get('state')):
                locations.append({
                    'name': row['state'],
                    'type': 'state'
                })
            if pd.notna(row.get('city')):
                city_state = f"{row.get('city', '')}, {row.get('state', '')}"
                locations.append({
                    'name': city_state.strip(', '),
                    'type': 'city'
                })
        
        # Remove duplicates
        unique_locations = {loc['name']: loc for loc in locations}.values()
        
        query = """
        UNWIND $locations AS loc
        MERGE (l:Location {name: loc.name})
        SET l.type = loc.type,
            l.created_at = datetime()
        """
        
        with self.driver.session() as session:
            session.run(query, locations=list(unique_locations))
        
        logger.info(f"✓ Created {len(unique_locations)} Location nodes")

    # ---------------------------------------------------
    # Create Group Nodes
    # ---------------------------------------------------
    def create_group_nodes(self):
        """Create group/organization nodes from incidents"""
        print("\n👥 Creating Group nodes...")
        
        groups = self.df_incidents['group'].dropna().unique()
        groups = [g for g in groups if g.strip()]  # Remove empty strings
        
        query = """
        UNWIND $groups AS group_name
        MERGE (g:Group {name: group_name})
        SET g.created_at = datetime()
        """
        
        with self.driver.session() as session:
            session.run(query, groups=list(groups))
        
        logger.info(f"✓ Created {len(groups)} Group nodes")

    # ---------------------------------------------------
    # Create Incident Nodes
    # ---------------------------------------------------
    def create_incident_nodes(self):
        """Create incident nodes from historical data"""
        print("\n📋 Creating Incident nodes...")
        
        rows = []
        for _, row in self.df_incidents.iterrows():
            incident_data = {
                'incident_id': str(row['incident_id']),
                'description': row.get('description', '')[:1000],  # Truncate
                'date': str(row.get('date', '')) if pd.notna(row.get('date')) else '',
                'city': row.get('city', ''),
                'state': row.get('state', ''),
                'location': row.get('location', ''),
                'attack_type': row.get('attack_type', ''),
                'ideology': row.get('ideology', ''),
                'group': row.get('group', '')
            }
            rows.append(incident_data)
        
        query = """
        UNWIND $rows AS row
        MERGE (i:Incident {incident_id: row.incident_id})
        SET i.description = row.description,
            i.date = row.date,
            i.city = row.city,
            i.state = row.state,
            i.location = row.location,
            i.attack_type = row.attack_type,
            i.ideology = row.ideology,
            i.group = row.group,
            i.created_at = datetime()
        """
        
        # Process in batches
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(rows), batch_size), desc="Creating incidents"):
            batch = rows[i:i+batch_size]
            with self.driver.session() as session:
                session.run(query, rows=batch)
        
        logger.info(f"✓ Created {len(rows)} Incident nodes")

    # ---------------------------------------------------
    # Create Incident-Location Relationships
    # ---------------------------------------------------
    def create_incident_location_relationships(self):
        """Create OCCURRED_IN relationships"""
        print("\n🔗 Linking Incidents to Locations...")
        
        query = """
        MATCH (i:Incident)
        WHERE i.state IS NOT NULL AND i.state <> ''
        MATCH (l:Location {name: i.state})
        MERGE (i)-[:OCCURRED_IN]->(l)
        """
        
        with self.driver.session() as session:
            session.run(query)
        
        logger.info("✓ Created OCCURRED_IN relationships")

    # ---------------------------------------------------
    # Create Incident-Group Relationships
    # ---------------------------------------------------
    def create_incident_group_relationships(self):
        """Create ATTRIBUTED_TO relationships"""
        print("\n🔗 Linking Incidents to Groups...")
        
        query = """
        MATCH (i:Incident)
        WHERE i.group IS NOT NULL AND i.group <> ''
        MATCH (g:Group {name: i.group})
        MERGE (i)-[:ATTRIBUTED_TO]->(g)
        """
        
        with self.driver.session() as session:
            session.run(query)
        
        logger.info("✓ Created ATTRIBUTED_TO relationships")

    # ---------------------------------------------------
    # Create USES_TERM (explicit tags)
    # ---------------------------------------------------
    def create_uses_term_relationships(self):
        """Create USES_TERM relationships from explicit term detection"""
        print("\n🔗 Creating USES_TERM relationships (explicit detection)...")
        
        if 'terms_detected' not in self.df_incidents.columns:
            logger.warning("⚠ No explicit terms_detected column — skipping USES_TERM")
            return
        
        rows = []
        for _, row in self.df_incidents.iterrows():
            if pd.isna(row.get('terms_detected')):
                continue
            
            # Parse terms_detected (could be list or JSON string)
            terms_detected = row['terms_detected']
            if isinstance(terms_detected, str):
                try:
                    terms = json.loads(terms_detected)
                except:
                    # Try comma-separated
                    terms = [t.strip() for t in terms_detected.split(',') if t.strip()]
            elif isinstance(terms_detected, list):
                terms = terms_detected
            else:
                continue
            
            for term in terms:
                if term:
                    rows.append({
                        'incident_id': str(row['incident_id']),
                        'term': term
                    })
        
        if not rows:
            logger.warning("⚠ No explicit term relationships found")
            return
        
        query = """
        UNWIND $rows AS row
        MATCH (i:Incident {incident_id: row.incident_id})
        MATCH (t:Term {term: row.term})
        MERGE (i)-[r:USES_TERM]->(t)
        SET r.detection_method = 'explicit',
            r.created_at = datetime()
        """
        
        # Process in batches
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(rows), batch_size), desc="Creating USES_TERM"):
            batch = rows[i:i+batch_size]
            with self.driver.session() as session:
                session.run(query, rows=batch)
        
        logger.info(f"✓ Created {len(rows)} USES_TERM relationships")

    # ---------------------------------------------------
    # Create SIMILAR_TO (embedding-based)
    # ---------------------------------------------------
    def create_similarity_relationships(self):
        """Create SIMILAR_TO relationships based on embeddings"""
        print("\n🔗 Creating SIMILAR_TO relationships (embedding-based)...")
        
        rows = []
        for _, row in self.df_matches.iterrows():
            # Only create relationship if score is above threshold
            if row['top_match_score'] >= self.config.similarity_threshold:
                rows.append({
                    'incident_id': str(row['incident_id']),
                    'term': row['top_match_term'],
                    'score': float(row['top_match_score']),
                    'category': row.get('top_match_category', 'unknown')
                })
        
        query = """
        UNWIND $rows AS row
        MATCH (i:Incident {incident_id: row.incident_id})
        MATCH (t:Term {term: row.term})
        MERGE (i)-[r:SIMILAR_TO]->(t)
        SET r.score = row.score,
            r.category = row.category,
            r.detection_method = 'embedding',
            r.created_at = datetime()
        """
        
        # Process in batches
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(rows), batch_size), desc="Creating SIMILAR_TO"):
            batch = rows[i:i+batch_size]
            with self.driver.session() as session:
                session.run(query, rows=batch)
        
        logger.info(f"✓ Created {len(rows)} SIMILAR_TO relationships (threshold: {self.config.similarity_threshold})")

    # ---------------------------------------------------
    # Create Term-Term Relationships
    # ---------------------------------------------------
    def create_term_relationships(self):
        """Create RELATED_TO relationships between terms"""
        print("\n🔗 Creating term-to-term RELATED_TO relationships...")
        
        # Load term relationships if available
        relationships_path = PROCESSED_DATA_DIR / "term_relationships.json"
        
        if not relationships_path.exists():
            logger.warning("⚠ No term relationships file found, skipping")
            return
        
        relationships = load_json(relationships_path)
        
        rows = []
        for rel in relationships.get('related', []):
            rows.append({
                'term1': rel['term1'],
                'term2': rel['term2'],
                'relationship_type': rel.get('relationship', 'related')
            })
        
        if not rows:
            logger.warning("⚠ No term relationships to create")
            return
        
        query = """
        UNWIND $rows AS row
        MATCH (t1:Term {term: row.term1})
        MATCH (t2:Term {term: row.term2})
        MERGE (t1)-[r:RELATED_TO]-(t2)
        SET r.relationship_type = row.relationship_type,
            r.created_at = datetime()
        """
        
        with self.driver.session() as session:
            session.run(query, rows=rows)
        
        logger.info(f"✓ Created {len(rows)} RELATED_TO relationships")

    # ---------------------------------------------------
    # Create Co-occurrence Relationships
    # ---------------------------------------------------
    def create_cooccurrence_relationships(self):
        """Create CO_OCCURS_WITH relationships between terms in same incidents"""
        print("\n🔗 Creating CO_OCCURS_WITH relationships...")
        
        query = """
        MATCH (i:Incident)-[:USES_TERM|SIMILAR_TO]->(t1:Term)
        MATCH (i)-[:USES_TERM|SIMILAR_TO]->(t2:Term)
        WHERE t1.term < t2.term
        WITH t1, t2, COUNT(i) AS cooccurrence_count
        WHERE cooccurrence_count >= 2
        MERGE (t1)-[r:CO_OCCURS_WITH]-(t2)
        SET r.count = cooccurrence_count,
            r.created_at = datetime()
        """
        
        with self.driver.session() as session:
            result = session.run(query)
        
        logger.info("✓ Created CO_OCCURS_WITH relationships")

    # ---------------------------------------------------
    # Generate Graph Statistics
    # ---------------------------------------------------
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the knowledge graph"""
        print("\n📊 Generating graph statistics...")
        
        stats = {}
        
        with self.driver.session() as session:
            # Node counts
            node_counts_query = """
            MATCH (n)
            RETURN labels(n)[0] AS label, COUNT(n) AS count
            ORDER BY count DESC
            """
            result = session.run(node_counts_query)
            stats['node_counts'] = {record['label']: record['count'] for record in result}
            
            # Relationship counts
            rel_counts_query = """
            MATCH ()-[r]->()
            RETURN type(r) AS type, COUNT(r) AS count
            ORDER BY count DESC
            """
            result = session.run(rel_counts_query)
            stats['relationship_counts'] = {record['type']: record['count'] for record in result}
            
            # Most connected terms
            top_terms_query = """
            MATCH (t:Term)<-[r]-(i:Incident)
            RETURN t.term AS term, t.category AS category, COUNT(r) AS incident_count
            ORDER BY incident_count DESC
            LIMIT 10
            """
            result = session.run(top_terms_query)
            stats['top_terms'] = [
                {'term': r['term'], 'category': r['category'], 'incidents': r['incident_count']}
                for r in result
            ]
            
            # Most common categories
            category_query = """
            MATCH (i:Incident)-[:SIMILAR_TO]->(t:Term)-[:BELONGS_TO]->(c:Category)
            RETURN c.name AS category, COUNT(DISTINCT i) AS incident_count
            ORDER BY incident_count DESC
            """
            result = session.run(category_query)
            stats['category_distribution'] = {r['category']: r['incident_count'] for r in result}
        
        return stats

    # ---------------------------------------------------
    # Export Graph
    # ---------------------------------------------------
    def export_graph(self):
        """Export graph to various formats"""
        print("\n💾 Exporting knowledge graph...")
        
        # Export as GraphML (for Cytoscape, Gephi, etc.)
        query = """
        CALL apoc.export.graphml.all($filepath, {})
        """
        
        # If APOC is not available, export as JSON
        with self.driver.session() as session:
            # Export nodes
            nodes_query = """
            MATCH (n)
            RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties
            """
            nodes_result = session.run(nodes_query)
            nodes = [dict(record) for record in nodes_result]
            
            # Export relationships
            rels_query = """
            MATCH (a)-[r]->(b)
            RETURN id(a) AS source, id(b) AS target, type(r) AS type, properties(r) AS properties
            """
            rels_result = session.run(rels_query)
            relationships = [dict(record) for record in rels_result]
        
        export_data = {
            'nodes': nodes,
            'relationships': relationships,
            'exported_at': datetime.now().isoformat()
        }
        
        save_json(export_data, GRAPH_JSON_PATH)
        logger.info(f"✓ Exported graph to {GRAPH_JSON_PATH}")

    # ---------------------------------------------------
    # Full Graph Build
    # ---------------------------------------------------
    def build_graph(self, clear_existing: bool = False):
        """Execute complete graph building pipeline"""
        print("\n🚀 Building Neo4j Knowledge Graph...")
        
        if clear_existing:
            self.clear_database()
        
        # Create schema
        self.create_constraints()
        
        # Create nodes
        self.create_category_nodes()
        self.create_term_nodes()
        self.create_location_nodes()
        self.create_group_nodes()
        self.create_incident_nodes()
        
        # Create relationships
        self.create_term_category_relationships()
        self.create_incident_location_relationships()
        self.create_incident_group_relationships()
        self.create_uses_term_relationships()
        self.create_similarity_relationships()
        self.create_term_relationships()
        self.create_cooccurrence_relationships()
        
        # Get statistics
        stats = self.get_graph_statistics()
        
        # Print statistics
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("="*60)
        print("\nNode Counts:")
        for label, count in stats['node_counts'].items():
            print(f"  {label}: {count:,}")
        
        print("\nRelationship Counts:")
        for rel_type, count in stats['relationship_counts'].items():
            print(f"  {rel_type}: {count:,}")
        
        print("\nTop 10 Most Referenced Terms:")
        for i, term_info in enumerate(stats['top_terms'], 1):
            print(f"  {i}. {term_info['term']} ({term_info['category']}): {term_info['incidents']} incidents")
        
        # Save statistics
        stats_path = RESULTS_DIR / "knowledge_graph_stats.json"
        save_json(stats, stats_path)
        
        # Export graph
        self.export_graph()
        
        logger.info("🎉 Phase 4 Complete — Knowledge graph ready")
        
        return stats

# ============================================================
# MAIN PIPELINE
# ============================================================


def run_phase4_pipeline(clear_existing: bool = False):
    """Execute complete Phase 4 knowledge graph pipeline"""
    print_phase_header(4, "Knowledge Graph Construction")
    
    try:
        # Setup configuration
        config = GraphConfig()
        
        # Initialize pipeline
        pipeline = Neo4jGraphPipeline(config)
        
        # Build graph
        stats = pipeline.build_graph(clear_existing=clear_existing)
        
        print_phase_footer(4, "Knowledge Graph Construction")
        
        return {
            'pipeline': pipeline,
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"✗ Phase 4 failed: {e}")
        raise


if __name__ == "__main__":
    # Run with clear_existing=True to rebuild from scratch
    results = run_phase4_pipeline(clear_existing=True)
    print("\n✓ Phase 4 complete. Knowledge graph trained and ready for processing.")
