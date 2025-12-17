#!/bin/bash

echo "=================================================="
echo "RESETTING PIPELINE - CLEARING ALL GENERATED DATA"
echo "=================================================="
echo ""
echo "⚠️  WARNING: This will delete:"
echo "   - All embeddings"
echo "   - All processed outputs"
echo "   - All evaluation results"
echo "   - Pipeline state markers"
echo "   - Neo4j knowledge graph data"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "🗑️  Clearing generated data..."

# Remove embeddings
rm -rf embeddings/*.npy
echo "  ✓ Cleared embeddings/"

# Remove outputs
rm -rf output/*.csv
rm -rf output/*.json
echo "  ✓ Cleared output/"

# Remove results
rm -rf results/*.txt
rm -rf results/*.json
rm -rf results/*.png
rm -rf results/*.csv
echo "  ✓ Cleared results/"

# Remove pipeline state markers
rm -rf .pipeline_state/
echo "  ✓ Cleared pipeline state"

# Remove processed data (keeps raw data)
rm -rf data/processed/*.csv
echo "  ✓ Cleared processed data"

# Remove logs
rm -rf logs/*.log
echo "  ✓ Cleared logs"

# Recreate directory structure
mkdir -p embeddings
mkdir -p output
mkdir -p results
mkdir -p data/processed
mkdir -p logs
mkdir -p .pipeline_state
echo "  ✓ Recreated directory structure"

echo ""
echo "=================================================="
echo "✅ PIPELINE RESET COMPLETE"
echo "=================================================="
echo ""
echo "To start fresh, run:"
echo "  python main_pipeline.py"
echo ""
echo "Or to clear Neo4j and start fresh:"
echo "  python main_pipeline.py --clear-neo4j"
echo ""

