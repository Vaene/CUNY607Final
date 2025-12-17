"""
Main Pipeline Runner
-------------------
Executes all phases of the hate incident classification pipeline
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import *
from utils import *

# Import all phase modules
from phase1_data_ingestion import run_phase1_pipeline
from phase2_glossary_processing import run_phase2_pipeline
from phase3_embeddings import run_phase3_pipeline
from phase4_knowledge_graph import run_phase4_pipeline
from phase5_raw_incident_processor import run_phase5_pipeline
from phase6_comparison_evaluation import run_phase6_pipeline
from phase7_export_for_r import run_phase7_pipeline

logger = setup_logging(__name__)

# ============================================================
# PIPELINE ORCHESTRATOR
# ============================================================

class PipelineOrchestrator:
    """Orchestrates execution of all pipeline phases"""
    
    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config
        self.results = {}
        
    def phase_is_complete(self, phase_num: int) -> bool:
        """Check if a phase has already been completed"""
        
        # Define what files each phase should produce
        phase_outputs = {
            1: [
                PROCESSED_DATA_DIR / "glossary_processed.csv",
                PROCESSED_DATA_DIR / "historical_processed.csv",
                PROCESSED_DATA_DIR / "training_data.csv"
            ],
            2: [
                PROCESSED_DATA_DIR / "glossary_final.csv",
                PROCESSED_DATA_DIR / "glossary_analysis.json"
            ],
            3: [
                INCIDENT_EMBEDDINGS_PATH,
                GLOSSARY_EMBEDDINGS_PATH,
                OUTPUT_DIR / "embedding_matches.csv"
            ],
            4: [
                GRAPH_JSON_PATH
            ],
            5: [
                MACHINE_PROCESSED_OUTPUT
            ],
            6: [
                COMPARISON_RESULTS_PATH,
                EVALUATION_METRICS_PATH
            ],
            7: [
                STRUCTURED_INCIDENTS_OUTPUT,
                R_METADATA_PATH
            ]
        }
        
        outputs = phase_outputs.get(phase_num, [])
        
        # Check if all output files exist
        return all(Path(f).exists() for f in outputs)

    def phase_is_complete(self, phase_num: int) -> bool:
        """Check if a phase has already been completed"""
        from pathlib import Path
        
        # Define what files each phase should produce
        phase_outputs = {
            1: [
                PROCESSED_DATA_DIR / "glossary_processed.csv",
                PROCESSED_DATA_DIR / "historical_processed.csv",
                PROCESSED_DATA_DIR / "training_data.csv"
            ],
            2: [
                PROCESSED_DATA_DIR / "glossary_final.csv",
                PROCESSED_DATA_DIR / "glossary_analysis.json"
            ],
            3: [
                INCIDENT_EMBEDDINGS_PATH,
                GLOSSARY_EMBEDDINGS_PATH,
                OUTPUT_DIR / "embedding_matches.csv"
            ],
            4: [
                GRAPH_JSON_PATH
            ],
            5: [
                MACHINE_PROCESSED_OUTPUT
            ],
            6: [
                COMPARISON_RESULTS_PATH,
                EVALUATION_METRICS_PATH
            ],
            7: [
                STRUCTURED_INCIDENTS_OUTPUT,
                R_METADATA_PATH
            ]
        }
        
        outputs = phase_outputs.get(phase_num, [])
        
        # Check if all output files exist
        return all(Path(f).exists() for f in outputs)

        
    def run_all_phases(self, start_phase: int = 1, end_phase: int = 7):
        """
        Run all phases of the pipeline
        
        Args:
            start_phase: Phase number to start from (1-7)
            end_phase: Phase number to end at (1-7)
        """
        print("\n" + "="*70)
        print("KNOWLEDGE GRAPH HATE INCIDENT CLASSIFICATION PIPELINE")
        print("="*70)
        print(f"\nStarting pipeline execution...")
        print(f"Phases: {start_phase} through {end_phase}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*70 + "\n")
        
        phases = [
            (1, "Data Ingestion", run_phase1_pipeline, self.config.run_phase1),
            (2, "Glossary Processing", run_phase2_pipeline, self.config.run_phase2),
            (3, "Embedding Generation", run_phase3_pipeline, self.config.run_phase3),
            (4, "Knowledge Graph Construction", run_phase4_pipeline, self.config.run_phase4),
            (5, "Raw Incident Processing", run_phase5_pipeline, self.config.run_phase5),
            (6, "Comparison & Evaluation", run_phase6_pipeline, self.config.run_phase6),
            (7, "R Export", run_phase7_pipeline, self.config.run_phase7),
        ]
        
        for phase_num, phase_name, phase_func, should_run in phases:
            if phase_num < start_phase or phase_num > end_phase:
                continue
                
            if not should_run:
                print(f"\n⏭️  Skipping Phase {phase_num}: {phase_name}")
                continue
            
            try:
                # Check if phase already completed
                if self.phase_is_complete(phase_num):
                    print(f"\n⏭️  Phase {phase_num} already complete, skipping...")
                    print(f"   (Use --start {phase_num} --end {phase_num} to force re-run)")
                    continue
                
                print(f"\n🚀 Executing Phase {phase_num}: {phase_name}")
                result = phase_func()
                self.results[f'phase{phase_num}'] = result
                print(f"✅ Phase {phase_num} completed successfully\n")
                
            except Exception as e:
                print(f"\n❌ Phase {phase_num} failed: {e}\n")
                logger.error(f"Phase {phase_num} failed", exc_info=True)
                
                # Ask user if they want to continue
                if self.config.verbose:
                    response = input("Continue with next phase? (y/n): ")
                    if response.lower() != 'y':
                        print("\nPipeline execution terminated by user.")
                        return False
                else:
                    print("Stopping pipeline execution due to error.")
                    return False
        
        print("\n" + "="*70)
        print("✅ PIPELINE EXECUTION COMPLETE")
        print("="*70 + "\n")
        
        self.print_summary()
        
        return True
    
    def print_summary(self):
        """Print summary of pipeline execution"""
        print("\n📊 PIPELINE EXECUTION SUMMARY")
        print("-"*70)
        print(f"Total phases executed: {len(self.results)}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Results directory: {RESULTS_DIR}")
        print("\n📁 Key Output Files:")
        
        key_files = [
            ("Structured Incidents", STRUCTURED_INCIDENTS_OUTPUT),
            ("Machine Processed", MACHINE_PROCESSED_OUTPUT),
            ("Comparison Results", COMPARISON_RESULTS_PATH),
            ("Evaluation Report", EVALUATION_REPORT_PATH),
            ("Evaluation Metrics", EVALUATION_METRICS_PATH),
            ("R Metadata", R_METADATA_PATH),
        ]
        
        for name, filepath in key_files:
            if filepath.exists():
                print(f"  ✓ {name}: {filepath}")
            else:
                print(f"  ✗ {name}: Not found")
        
        print("\n📈 Next Steps:")
        print("  1. Review evaluation report: results/evaluation_report.txt")
        print("  2. Run R analysis: Rscript -e \"rmarkdown::render('analysis_report.Rmd')\"")
        print("  3. View HTML report: analysis_report.html")
        print("-"*70 + "\n")

# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    """Main entry point for pipeline execution"""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Hate Incident Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main_pipeline.py
  
  # Run specific phases
  python main_pipeline.py --start 3 --end 5
  
  # Skip Neo4j phase (if not available)
  python main_pipeline.py --skip 4
  
  # Rebuild knowledge graph
  python main_pipeline.py --start 4 --end 4 --clear-neo4j
        """
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        choices=range(1, 8),
        help='Starting phase (1-7)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=7,
        choices=range(1, 8),
        help='Ending phase (1-7)'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        nargs='+',
        help='Phases to skip (e.g., --skip 4 to skip Neo4j)'
    )
    
    parser.add_argument(
        '--clear-neo4j',
        action='store_true',
        help='Clear Neo4j database before building graph (Phase 4 only)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig()
    config.verbose = args.verbose
    
    # Handle skipped phases
    if args.skip:
        for phase_num in args.skip:
            setattr(config, f'run_phase{phase_num}', False)
            print(f"⏭️  Will skip Phase {phase_num}")
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    # Run pipeline
    try:
        success = orchestrator.run_all_phases(args.start, args.end)
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            return 0
        else:
            print("\n⚠️  Pipeline completed with errors")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.error("Pipeline execution failed", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
