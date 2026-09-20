"""
End-to-end case-level demonstration script for the Arctan fraud detection pipeline.
This script showcases the full data generation, graph building, training, evaluation,
inference, counterfactual explanations, and drift detection.
"""

import time

# Section 1 Imports
from arctan.config import get_default_config
from arctan.data.download import ensure_dataset
from arctan.data.graph_builder import build_graph, load_graph
from arctan.data.preprocess import FEATURE_COLS, preprocess

# Section 7 Imports
from arctan.drift import detect_feature_drift

# Section 3 Imports
from arctan.evaluate import evaluate_model

# Section 4 Imports
from arctan.inference import FraudScorer

# Section 6 Imports
from arctan.models.counterfactual import CounterfactualExplainer

# Section 2 Imports
from arctan.train import train_model


def main():
    print("Arctan End-to-End Case-Level Demonstration\n")

    config = get_default_config()

    print("SECTION 1: Setup — Generate Data & Build Graph")
    graph = None
    try:
        print("Ensuring dataset exists...")
        ensure_dataset(config)

        print("Preprocessing dataset (creating nodes/edges with temporal split)...")
        preprocess(config)

        print("Building PyG graph...")
        build_graph(config)

        print("Loading graph...")
        graph = load_graph(config)
        print(f"Graph loaded successfully: {graph.num_nodes} nodes, {graph.num_edges} edges")
    except Exception as e:
        print(f"Section 1 failed: {e}")

    print("\nSECTION 2: Train Static Model")
    try:
        print("Training model...")
        train_model(config)
        print("Model training completed.")
    except Exception as e:
        print(f"Section 2 failed: {e}")

    print("\nSECTION 3: Evaluate")
    try:
        print("Evaluating model...")
        metrics = evaluate_model(config)
        print("Evaluation Metrics:")
        if metrics:
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"Section 3 failed: {e}")

    print("\nSECTION 4: Batch Scoring Demo")
    scorer = None
    batch_results = []
    try:
        scorer = FraudScorer(config)
        
        if graph is not None and hasattr(graph, 'entity_id') and len(graph.entity_id) > 0:
            sample_ids = graph.entity_id[:20]  # first 20 entities
            print(f"Scoring batch of {len(sample_ids)} entities...")
            
            t0 = time.perf_counter()
            batch_results = scorer.score_batch(sample_ids)
            elapsed = time.perf_counter() - t0
            
            print(f"Batch scoring completed in {elapsed:.4f} seconds.\n")
            
            header = (
                f"{'Entity ID':<15} | {'Score':<6} | "
                f"{'Level':<10} | {'P(Fraud)':<10} | "
                f"{'Calib P'}"
            )
            print(header)
            print("-" * 70)
            for res in batch_results:
                if res is not None:
                    eid = res.get("entity_id", "N/A")[:14]
                    score = res.get("risk_score", 0)
                    level = res.get("risk_level", "N/A")
                    pfraud = res.get("fraud_probability", 0.0)
                    pcalib = res.get(
                        "calibrated_fraud_probability", 0.0
                    )
                    print(
                        f"{eid:<15} | {score:<6} | "
                        f"{level:<10} | {pfraud:<10.4f} | "
                        f"{pcalib:<10.4f}"
                    )
        else:
            print("Graph or entity IDs not available for batch scoring.")
    except Exception as e:
        print(f"Section 4 failed: {e}")

    print("\nSECTION 5: Single Entity Deep Dive")
    high_risk_entity = None
    try:
        # Pick the highest risk entity from the batch results if available
        if batch_results:
            valid_results = [r for r in batch_results if r is not None]
            if valid_results:
                highest_risk = max(valid_results, key=lambda x: x.get("risk_score", 0))
                high_risk_entity = highest_risk.get("entity_id")
                
        if high_risk_entity:
            print(f"Deep dive for high-risk entity: {high_risk_entity}")
            single_result = scorer.score_entity(high_risk_entity)
            
            print("Detailed Scoring Result:")
            for k, v in single_result.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print("No high-risk entity found to deep dive.")
    except Exception as e:
        print(f"Section 5 failed: {e}")

    print("\nSECTION 6: Counterfactual Explanation")
    try:
        if scorer and scorer.model and graph and high_risk_entity:
            # Find the node index for the high risk entity
            high_risk_idx = scorer.node_to_idx.get(high_risk_entity)
            
            if high_risk_idx is not None:
                print(f"Generating counterfactual explanation for node index {high_risk_idx}...")
                explainer = CounterfactualExplainer(scorer.model, feature_names=FEATURE_COLS)
                cf_result = explainer.find_counterfactual(node_idx=high_risk_idx, graph=graph)
                
                cf_prob = cf_result.get(
                    'counterfactual_prob', 0.0
                )
                print(f"Success: {cf_result.get('success')}")
                orig = cf_result.get('original_prob', 0.0)
                print(f"Original Probability: {orig:.4f}")
                print(f"Counterfactual Probability: {cf_prob:.4f}")
                print("Features changed:")
                
                perturbation = cf_result.get("perturbation", {})
                if perturbation:
                    for feat, change in perturbation.items():
                        print(f"  {feat}: {change:+.4f}")
                else:
                    print("  None")
            else:
                print("High-risk entity index not found in graph.")
        else:
            print("Model, graph, or high-risk entity not available for counterfactual explanation.")
    except Exception as e:
        print(f"Section 6 failed: {e}")

    print("\nSECTION 7: Drift Detection")
    try:
        if graph is not None and hasattr(graph, 'train_mask') and hasattr(graph, 'test_mask'):
            print("Detecting feature drift between train and test distributions...")
            
            train_features = graph.x[graph.train_mask].cpu().numpy()
            test_features = graph.x[graph.test_mask].cpu().numpy()
            
            drift_config = config.drift
            feature_drift = detect_feature_drift(
                train_features, test_features,
                FEATURE_COLS, drift_config,
            )
            
            print(f"Summary: {feature_drift.get('summary', 'N/A')}")
            drifted_features = [
                f for f, m in feature_drift.get(
                    "features", {}
                ).items()
                if m.get("drifted")
            ]
            
            if drifted_features:
                print("Drifted features list:")
                for f in drifted_features[:5]:
                    print(f"  - {f}")
                if len(drifted_features) > 5:
                    print(f"  ... and {len(drifted_features) - 5} more.")
            else:
                print("No significantly drifted features found.")
        else:
            print("Graph masks not available for drift detection.")
    except Exception as e:
        print(f"Section 7 failed: {e}")

    print("\nSECTION 8: Summary")
    try:
        print("Demo completed successfully. In a real-world setting, this entire pipeline")
        print("would be scheduled automatically, ingesting transactions and periodically")
        print("retraining the model and refreshing the knowledge graph while continuous")
        print("drift monitoring runs in the background.")
    except Exception as e:
        print(f"Section 8 failed: {e}")

    print("\nArctan Demo Finished")

if __name__ == '__main__':
    main()
