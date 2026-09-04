#!/usr/bin/env bash

set -euo pipefail

MODEL_TYPE="${1:-temporal}"

GRAPH_FILE="data/processed/fraud_graph.pt"
TEMPORAL_FILE="data/processed/temporal_data.pt"
STATIC_MODEL="data/models/fraud_gnn_best.pt"
TEMPORAL_MODEL="data/models/temporal_gnn_best.pt"

echo "Model type: $MODEL_TYPE"
echo ""

# Step 1: Build graph artifacts
if [ "$MODEL_TYPE" = "temporal" ]; then
    if [ -f "$TEMPORAL_FILE" ]; then
        echo "Step 1: Temporal data already exists at $TEMPORAL_FILE skipping"
    else
        echo "Step 1: Build temporal data (download → preprocess → graph_builder)"
        uv run python -m arctan.data.graph_builder
    fi
else
    if [ -f "$GRAPH_FILE" ]; then
        echo "Step 1: Graph already exists at $GRAPH_FILE skipping"
    else
        echo "Step 1: Build static graph (download → preprocess → graph_builder)"
        uv run python -m arctan.data.graph_builder
    fi
fi

echo ""

# Step 2: Train model
if [ "$MODEL_TYPE" = "temporal" ]; then
    if [ -f "$TEMPORAL_MODEL" ]; then
        echo "Step 2: Temporal model already trained at $TEMPORAL_MODEL skipping"
    else
        echo "Step 2: Train temporal GNN (TGN with entity memory)"
        uv run python -m arctan.temporal_train
    fi
else
    if [ -f "$STATIC_MODEL" ]; then
        echo "Step 2: Static model already trained at $STATIC_MODEL skipping"
    else
        echo "Step 2: Train static FraudGNN"
        uv run python -m arctan.train
    fi
fi

echo ""
echo "Step 3: Evaluate model ($MODEL_TYPE)"
uv run python -m arctan.evaluate "$MODEL_TYPE"

echo ""
echo "Done ($MODEL_TYPE model)"
