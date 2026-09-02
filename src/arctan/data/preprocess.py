"""Feature engineering for financial transaction graphs with temporal entity splits.

**Temporal split strategy (standard transductive GNN)**:
  • All entities get features computed from all transactions (full graph structure)
  • Entity *assignment* to train/val/test is based on temporal period of first activity
  • StandardScaler is fit on training entities only, then applied to all
  • This is the standard approach in temporal node classification literature

Temporal boundaries (on 744-hour / 30-day window):
  • Train:  entities whose first transaction is in steps [1, 445]   (~60%)
  • Val:    entities whose first transaction is in steps [446, 594]  (~20%)
  • Test:   entities whose first transaction is in steps [595, 744]  (~20%)

Graph-structural features engineered per entity (16 total):
  • `in_degree`, `out_degree`, `total_degree`
  • `in_volume`, `out_volume`, `total_volume`, `net_cashflow`
  • `unique_in_counterparties`, `unique_out_counterparties`, `total_counterparties`
  • `avg_transaction_size`, `avg_in_amount`, `avg_out_amount`
  • `out_high_risk_type_ratio`
  • `balance_depletion_ratio`: fraction of outgoing txns draining balance to zero
  • `max_single_txn_ratio`: largest single txn / total outgoing volume
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import polars as pl
from sklearn.preprocessing import StandardScaler

from arctan.config import PipelineConfig, get_default_config
from arctan.data.download import download_all

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Temporal split boundaries (on 744-step / 30-day hourly window)
TRAIN_END = 445       # steps [1, 445]   ≈ 60%
VAL_END = 594         # steps [446, 594] ≈ 20%
                      # steps [595, 744] ≈ 20%  (test)


class TemporalSplitResult(NamedTuple):
    """Output of the temporal preprocessing pipeline."""
    nodes_df: pl.DataFrame
    edges_df: pl.DataFrame
    train_entity_ids: set[str]
    val_entity_ids: set[str]
    test_entity_ids: set[str]


# The 16 features used for GNN input (order matters for consistency)
FEATURE_COLS = [
    "out_degree",
    "in_degree",
    "total_degree",
    "out_volume",
    "in_volume",
    "total_volume",
    "net_cashflow",
    "avg_out_amount",
    "avg_in_amount",
    "avg_transaction_size",
    "unique_out_counterparties",
    "unique_in_counterparties",
    "total_counterparties",
    "out_high_risk_type_ratio",
    "balance_depletion_ratio",
    "max_single_txn_ratio",
]


def _load_raw_transactions(transactions_dir: Path) -> pl.DataFrame:
    """Load transaction records from parquet or CSV files."""
    parquet_files = list(transactions_dir.glob("**/*.parquet"))
    csv_files = list(transactions_dir.glob("**/*.csv"))

    if parquet_files:
        logger.info("Scanning %d Parquet file(s) in %s...", len(parquet_files), transactions_dir)
        lf = pl.scan_parquet(parquet_files)
    elif csv_files:
        logger.info("Scanning %d CSV file(s) in %s...", len(csv_files), transactions_dir)
        lf = pl.scan_csv(csv_files, infer_schema_length=10000)
    else:
        logger.warning("No transaction files found in %s", transactions_dir)
        return pl.DataFrame()

    cols = lf.collect_schema().names()

    sender_col = (
        "sender" if "sender" in cols
        else "nameOrig" if "nameOrig" in cols
        else "from_account" if "from_account" in cols
        else None
    )
    receiver_col = (
        "receiver" if "receiver" in cols
        else "nameDest" if "nameDest" in cols
        else "to_account" if "to_account" in cols
        else None
    )
    amount_col = "amount" if "amount" in cols else "value" if "value" in cols else None
    time_col = (
        "step" if "step" in cols
        else "timestamp" if "timestamp" in cols
        else "time" if "time" in cols
        else None
    )
    fraud_col = (
        "is_fraud" if "is_fraud" in cols
        else "isFraud" if "isFraud" in cols
        else "label" if "label" in cols
        else None
    )
    type_col = (
        "txn_type" if "txn_type" in cols
        else "type" if "type" in cols
        else None
    )
    old_bal_col = (
        "old_balance_sender" if "old_balance_sender" in cols
        else "oldbalanceOrg" if "oldbalanceOrg" in cols
        else None
    )
    new_bal_col = (
        "new_balance_sender" if "new_balance_sender" in cols
        else "newbalanceOrig" if "newbalanceOrig" in cols
        else None
    )

    if not sender_col or not receiver_col:
        logger.error("Required sender/receiver columns not found.")
        return pl.DataFrame()

    select_exprs = [
        pl.col(sender_col).cast(pl.Utf8).alias("sender"),
        pl.col(receiver_col).cast(pl.Utf8).alias("receiver"),
    ]

    if amount_col:
        select_exprs.append(pl.col(amount_col).cast(pl.Float64).alias("amount"))
    else:
        select_exprs.append(pl.lit(1.0).cast(pl.Float64).alias("amount"))

    if time_col:
        select_exprs.append(pl.col(time_col).cast(pl.Float64).alias("step"))
    else:
        select_exprs.append(pl.lit(0.0).cast(pl.Float64).alias("step"))

    if fraud_col:
        select_exprs.append(pl.col(fraud_col).cast(pl.Int64).alias("is_fraud"))
    else:
        select_exprs.append(pl.lit(0).cast(pl.Int64).alias("is_fraud"))

    if type_col:
        select_exprs.append(pl.col(type_col).cast(pl.Utf8).alias("txn_type"))
    else:
        select_exprs.append(pl.lit("TRANSFER").cast(pl.Utf8).alias("txn_type"))

    if old_bal_col:
        select_exprs.append(pl.col(old_bal_col).cast(pl.Float64).alias("old_balance_sender"))
    else:
        select_exprs.append(pl.lit(0.0).cast(pl.Float64).alias("old_balance_sender"))

    if new_bal_col:
        select_exprs.append(pl.col(new_bal_col).cast(pl.Float64).alias("new_balance_sender"))
    else:
        select_exprs.append(pl.lit(0.0).cast(pl.Float64).alias("new_balance_sender"))

    df = lf.select(select_exprs).drop_nulls(subset=["sender", "receiver"]).collect()
    return df.sort("step")


def preprocess(config: PipelineConfig) -> TemporalSplitResult:
    """Run feature engineering with temporal entity assignment.

    1. Load and sort transactions by timestamp
    2. Build global entity set with features from ALL transactions
    3. Assign entities to train/val/test by first-transaction time
    4. Fit StandardScaler on training entities only
    5. Return enriched nodes_df, edges_df, and entity split sets
    """
    config.paths.ensure_dirs()
    raw_txns = _load_raw_transactions(config.paths.transactions_dir)

    if raw_txns.height == 0:
        logger.info("No transactions found. Acquiring benchmark dataset...")
        download_all(config)
        raw_txns = _load_raw_transactions(config.paths.transactions_dir)

    logger.info("Loaded %d raw financial transactions.", raw_txns.height)

    # 1. Build unique entity set
    senders = raw_txns.select(pl.col("sender").alias("entity_id"))
    receivers = raw_txns.select(pl.col("receiver").alias("entity_id"))
    all_nodes = pl.concat([senders, receivers]).unique(subset=["entity_id"])
    all_nodes = all_nodes.with_row_index("node_id")

    logger.info("Extracted %d unique account entities.", all_nodes.height)

    # 2. Entity-level fraud labels
    fraud_senders = (
        raw_txns.filter(pl.col("is_fraud") == 1)
        .select(pl.col("sender").alias("entity_id"))
        .unique()
    )
    # Only the SENDER of a fraudulent transaction is a fraud entity.
    # Receivers are victims or unknowing participants — labeling them as fraud
    # would inflate prevalence and conflate victims with bad actors.
    fraud_entities = fraud_senders.unique()
    fraud_entities = fraud_entities.with_columns(pl.lit(1).alias("is_fraud"))

    all_nodes = all_nodes.join(fraud_entities, on="entity_id", how="left")
    all_nodes = all_nodes.with_columns(pl.col("is_fraud").fill_null(0))

    fraud_count = all_nodes.filter(pl.col("is_fraud") == 1).height
    logger.info(
        "Entity fraud distribution: %d fraudulent (%.2f%%)",
        fraud_count,
        (fraud_count / max(1, all_nodes.height)) * 100,
    )

    # 3. Map edges to node indices
    edges_df = raw_txns.join(
        all_nodes.select(["entity_id", "node_id"]).rename({"node_id": "src_id"}),
        left_on="sender",
        right_on="entity_id",
        how="inner",
    ).join(
        all_nodes.select(["entity_id", "node_id"]).rename({"node_id": "dst_id"}),
        left_on="receiver",
        right_on="entity_id",
        how="inner",
    )

    # 4. Feature engineering from ALL transactions
    logger.info("Engineering graph-structural features...")

    out_agg = edges_df.group_by("src_id").agg(
        [
            pl.len().alias("out_degree"),
            pl.col("amount").sum().alias("out_volume"),
            pl.col("amount").mean().alias("avg_out_amount"),
            pl.col("amount").max().alias("max_out_amount"),
            pl.col("dst_id").n_unique().alias("unique_out_counterparties"),
            (
                pl.col("txn_type").is_in(["TRANSFER", "CASH_OUT"]).sum()
                / pl.len()
            ).alias("out_high_risk_type_ratio"),
            (
                (pl.col("new_balance_sender") == 0.0).sum()
                / pl.len()
            ).alias("balance_depletion_ratio"),
        ]
    )

    in_agg = edges_df.group_by("dst_id").agg(
        [
            pl.len().alias("in_degree"),
            pl.col("amount").sum().alias("in_volume"),
            pl.col("amount").mean().alias("avg_in_amount"),
            pl.col("src_id").n_unique().alias("unique_in_counterparties"),
        ]
    )

    all_nodes = all_nodes.join(out_agg, left_on="node_id", right_on="src_id", how="left")
    all_nodes = all_nodes.join(in_agg, left_on="node_id", right_on="dst_id", how="left")

    all_nodes = all_nodes.fill_null(0.0).with_columns(
        [
            (pl.col("out_degree") + pl.col("in_degree")).alias("total_degree"),
            (pl.col("out_volume") + pl.col("in_volume")).alias("total_volume"),
            (pl.col("in_volume") - pl.col("out_volume")).alias("net_cashflow"),
            (
                pl.col("unique_out_counterparties") + pl.col("unique_in_counterparties")
            ).alias("total_counterparties"),
        ]
    )

    all_nodes = all_nodes.with_columns(
        pl.when(pl.col("total_degree") > 0)
        .then(pl.col("total_volume") / pl.col("total_degree"))
        .otherwise(0.0)
        .alias("avg_transaction_size")
    )

    all_nodes = all_nodes.with_columns(
        pl.when(pl.col("out_volume") > 0)
        .then(pl.col("max_out_amount") / pl.col("out_volume"))
        .otherwise(0.0)
        .alias("max_single_txn_ratio")
    )

    if "max_out_amount" in all_nodes.columns:
        all_nodes = all_nodes.drop("max_out_amount")

    # 5. Temporal entity assignment
    # Non-fraud entities: assigned by first transaction time
    # Fraud entities: assigned by first FRAUDULENT transaction time
    # This ensures fraud is distributed across train/val/test splits
    # based on when the fraudulent behavior begins.

    # First transaction time for all entities
    first_txn_sender = raw_txns.group_by("sender").agg(
        pl.col("step").min().alias("first_step")
    ).rename({"sender": "entity_id"})
    first_txn_receiver = raw_txns.group_by("receiver").agg(
        pl.col("step").min().alias("first_step")
    ).rename({"receiver": "entity_id"})
    first_txn = pl.concat([first_txn_sender, first_txn_receiver])
    first_txn = first_txn.group_by("entity_id").agg(
        pl.col("first_step").min().alias("first_step")
    )

    # First FRAUD transaction time for fraud entities (senders only)
    fraud_txns_df = raw_txns.filter(pl.col("is_fraud") == 1)
    first_fraud_txn = fraud_txns_df.group_by("sender").agg(
        pl.col("step").min().alias("first_fraud_step")
    ).rename({"sender": "entity_id"})

    # Merge: fraud entities use first_fraud_step, non-fraud use first_step
    assignment = first_txn.join(first_fraud_txn, on="entity_id", how="left")
    assignment = assignment.with_columns(
        pl.when(pl.col("first_fraud_step").is_not_null())
        .then(pl.col("first_fraud_step"))
        .otherwise(pl.col("first_step"))
        .alias("split_step")
    )

    train_entities = set(
        assignment.filter(pl.col("split_step") <= TRAIN_END)["entity_id"].to_list()
    )
    val_entities = set(
        assignment.filter(
            (pl.col("split_step") > TRAIN_END) & (pl.col("split_step") <= VAL_END)
        )["entity_id"].to_list()
    )
    test_entities = set(
        assignment.filter(pl.col("split_step") > VAL_END)["entity_id"].to_list()
    )

    logger.info(
        "Temporal entity assignment: train=%d, val=%d, test=%d",
        len(train_entities), len(val_entities), len(test_entities),
    )

    # Log fraud distribution per temporal split
    split_groups = [
        ("train", train_entities),
        ("val", val_entities),
        ("test", test_entities),
    ]
    for name, entity_set in split_groups:
        fraud_in_split = len(
            entity_set & set(
                all_nodes.filter(pl.col("is_fraud") == 1)["entity_id"].to_list()
            )
        )
        total_in_split = len(entity_set)
        prevalence = (fraud_in_split / total_in_split * 100) if total_in_split > 0 else 0.0
        logger.info(
            "  %s: %d entities, %d fraud (%.2f%% prevalence)",
            name, total_in_split, fraud_in_split, prevalence,
        )

    # 6. Normalize features (fit on train entities only)
    logger.info("Fitting StandardScaler on training entities only...")
    scaler = StandardScaler()

    train_nids = set(
        all_nodes.filter(pl.col("entity_id").is_in(list(train_entities)))["node_id"].to_list()
    )
    train_feature_matrix = all_nodes.filter(
        pl.col("node_id").is_in(list(train_nids))
    ).select(FEATURE_COLS).to_numpy()

    scaler.fit(train_feature_matrix)

    # Transform ALL nodes using train-fitted scaler
    full_feature_matrix = all_nodes.select(FEATURE_COLS).to_numpy()
    scaled_matrix = scaler.transform(full_feature_matrix)

    for i, col in enumerate(FEATURE_COLS):
        all_nodes = all_nodes.with_columns(
            pl.Series(name=col, values=scaled_matrix[:, i]).cast(pl.Float32)
        )

    # 7. Build edge DataFrame
    edges_out = edges_df.select(["src_id", "dst_id", "amount", "step"])

    logger.info(
        "Preprocessing complete: %d nodes, %d edges, %d features.",
        all_nodes.height,
        edges_out.height,
        len(FEATURE_COLS),
    )

    return TemporalSplitResult(
        nodes_df=all_nodes,
        edges_df=edges_out,
        train_entity_ids=train_entities,
        val_entity_ids=val_entities,
        test_entity_ids=test_entities,
    )


if __name__ == "__main__":
    cfg = get_default_config()
    result = preprocess(cfg)
