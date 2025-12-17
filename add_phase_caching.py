#!/usr/bin/env python3
"""Add smart caching to skip completed phases"""

from pathlib import Path

main_file = Path("main_pipeline.py")
content = main_file.read_text()

# Add caching logic before run_all_phases
cache_code = '''
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
'''

# Find the PipelineOrchestrator class and add the method
if 'def phase_is_complete' not in content:
    # Add after __init__ method
    pattern = 'def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):\n        self.config = config\n        self.results = {}'
    
    if pattern in content:
        content = content.replace(pattern, pattern + '\n' + cache_code)
        print("✓ Added phase_is_complete method")
    else:
        print("⚠ Could not find insertion point")

# Modify run_all_phases to check cache
old_run_logic = '''            try:
                print(f"\\n🚀 Executing Phase {phase_num}: {phase_name}")
                result = phase_func()'''

new_run_logic = '''            try:
                # Check if phase already completed
                if self.phase_is_complete(phase_num):
                    print(f"\\n⏭️  Phase {phase_num} already complete, skipping...")
                    print(f"   (Use --start {phase_num} --end {phase_num} to force re-run)")
                    continue
                
                print(f"\\n🚀 Executing Phase {phase_num}: {phase_name}")
                result = phase_func()'''

if old_run_logic in content:
    content = content.replace(old_run_logic, new_run_logic)
    print("✓ Added cache checking to run_all_phases")

main_file.write_text(content)
print("\n✓ Smart caching enabled!")
print("\nNow when you run python main_pipeline.py:")
print("  - It will skip phases that are already complete")
print("  - To force re-run: python main_pipeline.py --start 2 --end 2")
