#!/usr/bin/env bash

set -euo pipefail

GRAPH_FILE="data/processed/fraud_graph.pt"
MODEL_FILE="data/models/fraud_gnn_best.pt"

if [ -f "$GRAPH_FILE" ]; then
    echo "Step 1: Graph already exists at $GRAPH_FILE skipping"
else
    echo "Step 1: Build temporal graph (download → preprocess → graph_builder)"
    uv run python -m arctan.data.graph_builder
fi

echo ""

if [ -f "$MODEL_FILE" ]; then
    echo "Step 2: Model already trained at $MODEL_FILE skipping"
else
    echo "Step 2: Train model"
    uv run python -m arctan.train
fi

echo ""
echo "Step 3: Evaluate model"
uv run python -m arctan.evaluate

echo ""
echo "Done"
