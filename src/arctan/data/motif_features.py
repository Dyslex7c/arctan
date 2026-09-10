"""Temporal motif feature engineering for fraud detection.

Computes per-entity structural motif features that capture dynamic fraud
patterns in the transaction graph.  All features are derived from the edge
DataFrame produced by :func:`arctan.data.preprocess.preprocess`.

**Motif features (6 total)**:
  • `t2_cycle_count`: Circular money flow (A→B→A layering)
  • `t3_cycle_count`: Multi-hop structuring rings (A→B→C→A)
  • `fan_out_burst_count`: Rapid fund dispersal to many targets
  • `fan_in_burst_count`: Mule account aggregation from many sources
  • `ping_pong_ratio`: Round-trip money flow fraction
  • `temporal_clustering_coeff`: Coordinated ring activity

All motif computations respect a configurable time window (default 48 steps
for cycles, 10 steps for bursts) so that only temporally proximate patterns
are counted.
"""

from __future__ import annotations

import logging

import polars as pl

from arctan.config import MotifConfig

logger = logging.getLogger(__name__)

MOTIF_FEATURE_COLS = [
    "t2_cycle_count",
    "t3_cycle_count",
    "fan_out_burst_count",
    "fan_in_burst_count",
    "ping_pong_ratio",
    "temporal_clustering_coeff",
]


def _compute_t2_cycles(edges_df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Count temporal 2-cycles per entity: A→B then B→A within *window* steps.

    A 2-cycle indicates circular money flow — a hallmark of layering in
    money laundering where funds are moved out and returned to obfuscate
    their origin.

    Returns a DataFrame with columns ``[node_id, t2_cycle_count]``.
    """
    edges_ab = edges_df.select(
        pl.col("src_id").alias("a"),
        pl.col("dst_id").alias("b"),
        pl.col("step").alias("t1"),
    )
    edges_ba = edges_df.select(
        pl.col("src_id").alias("b_ret"),
        pl.col("dst_id").alias("a_ret"),
        pl.col("step").alias("t2"),
    )

    # Join on B and filter for A→B then B→A within window
    cycles = (
        edges_ab.join(edges_ba, left_on="b", right_on="b_ret", how="inner")
        .filter(
            (pl.col("a") == pl.col("a_ret"))
            & (pl.col("t2") > pl.col("t1"))
            & ((pl.col("t2") - pl.col("t1")) <= window)
        )
    )

    counts = cycles.group_by("a").agg(pl.len().alias("t2_cycle_count"))
    return counts.rename({"a": "node_id"})


def _compute_t3_cycles(edges_df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Count temporal 3-cycles per entity: A→B→C→A within *window* steps.

    A 3-cycle indicates multi-hop structuring — a classic pattern where
    funds pass through intermediate accounts before returning, creating
    a laundering ring.

    Returns a DataFrame with columns ``[node_id, t3_cycle_count]``.
    """
    # A→B at t1
    ab = edges_df.select(
        pl.col("src_id").alias("a"),
        pl.col("dst_id").alias("b"),
        pl.col("step").alias("t1"),
    )
    # B→C at t2
    bc = edges_df.select(
        pl.col("src_id").alias("b2"),
        pl.col("dst_id").alias("c"),
        pl.col("step").alias("t2"),
    )
    # C→A at t3
    ca = edges_df.select(
        pl.col("src_id").alias("c2"),
        pl.col("dst_id").alias("a2"),
        pl.col("step").alias("t3"),
    )

    # Chain: A→B→C (t2 > t1, within window)
    ab_bc = (
        ab.join(bc, left_on="b", right_on="b2", how="inner")
        .filter(
            (pl.col("t2") > pl.col("t1"))
            & ((pl.col("t2") - pl.col("t1")) <= window)
        )
    )

    # Extend: A→B→C→A (t3 > t2, overall within window from t1)
    cycles_3 = (
        ab_bc.join(ca, left_on="c", right_on="c2", how="inner")
        .filter(
            (pl.col("a") == pl.col("a2"))
            & (pl.col("t3") > pl.col("t2"))
            & ((pl.col("t3") - pl.col("t1")) <= window)
        )
    )

    counts = cycles_3.group_by("a").agg(pl.len().alias("t3_cycle_count"))
    return counts.rename({"a": "node_id"})


def _compute_fan_out_bursts(
    edges_df: pl.DataFrame, burst_window: int, min_targets: int
) -> pl.DataFrame:
    """Count fan-out burst episodes per entity.

    A burst is a non-overlapping time bin of ``burst_window`` steps in which
    the entity sends to ≥ ``min_targets`` unique counterparties.  This
    captures rapid fund dispersal patterns typical of layering.

    Returns a DataFrame with columns ``[node_id, fan_out_burst_count]``.
    """
    outgoing = edges_df.select(
        pl.col("src_id").alias("node_id"),
        pl.col("dst_id"),
        pl.col("step"),
    ).with_columns(
        (pl.col("step") // burst_window).alias("time_bin"),
    )

    bins = outgoing.group_by(["node_id", "time_bin"]).agg(
        pl.col("dst_id").n_unique().alias("n_unique_targets"),
    )

    bursts = (
        bins.filter(pl.col("n_unique_targets") >= min_targets)
        .group_by("node_id")
        .agg(pl.len().alias("fan_out_burst_count"))
    )
    return bursts


def _compute_fan_in_bursts(
    edges_df: pl.DataFrame, burst_window: int, min_sources: int
) -> pl.DataFrame:
    """Count fan-in burst episodes per entity.

    A burst is a non-overlapping time bin of ``burst_window`` steps in which
    the entity receives from ≥ ``min_sources`` unique counterparties.  This
    captures mule account aggregation patterns.

    Returns a DataFrame with columns ``[node_id, fan_in_burst_count]``.
    """
    incoming = edges_df.select(
        pl.col("dst_id").alias("node_id"),
        pl.col("src_id"),
        pl.col("step"),
    ).with_columns(
        (pl.col("step") // burst_window).alias("time_bin"),
    )

    bins = incoming.group_by(["node_id", "time_bin"]).agg(
        pl.col("src_id").n_unique().alias("n_unique_sources"),
    )

    bursts = (
        bins.filter(pl.col("n_unique_sources") >= min_sources)
        .group_by("node_id")
        .agg(pl.len().alias("fan_in_burst_count"))
    )
    return bursts


def _compute_ping_pong_ratio(
    edges_df: pl.DataFrame, window: int
) -> pl.DataFrame:
    """Compute the ping-pong ratio per entity.

    For each entity A, this is the fraction of its unique outgoing
    counterparties B where a reverse edge B→A exists within ``window``
    steps.  A high ping-pong ratio suggests round-trip money flow.

    Returns a DataFrame with columns ``[node_id, ping_pong_ratio]``.
    """
    # Unique outgoing edges per sender
    out_edges = edges_df.select(
        pl.col("src_id").alias("a"),
        pl.col("dst_id").alias("b"),
        pl.col("step").alias("t1"),
    )
    # Potential reverse edges
    rev_edges = edges_df.select(
        pl.col("src_id").alias("b_rev"),
        pl.col("dst_id").alias("a_rev"),
        pl.col("step").alias("t2"),
    )

    # Find reciprocated edges within window (either direction in time)
    reciprocated = (
        out_edges.join(rev_edges, left_on="b", right_on="b_rev", how="inner")
        .filter(
            (pl.col("a") == pl.col("a_rev"))
            & ((pl.col("t2") - pl.col("t1")).abs() <= window)
        )
        .select(["a", "b"])
        .unique()
    )

    recip_counts = reciprocated.group_by("a").agg(
        pl.len().alias("reciprocated_count"),
    )

    total_out_pairs = (
        out_edges.select(["a", "b"])
        .unique()
        .group_by("a")
        .agg(pl.len().alias("total_out_pairs"))
    )

    ping_pong = (
        total_out_pairs.join(recip_counts, on="a", how="left")
        .with_columns(
            (pl.col("reciprocated_count").fill_null(0) / pl.col("total_out_pairs"))
            .alias("ping_pong_ratio"),
        )
        .select(pl.col("a").alias("node_id"), "ping_pong_ratio")
    )
    return ping_pong


def _compute_temporal_clustering(edges_df: pl.DataFrame) -> pl.DataFrame:
    """Compute an undirected clustering coefficient per entity.

    For each entity A, finds all neighbours and checks what fraction of
    neighbour pairs (B, C) are themselves connected by an edge.  High
    clustering indicates participation in a tightly connected group
    (e.g. a fraud ring).

    Returns a DataFrame with columns ``[node_id, temporal_clustering_coeff]``.
    """
    # Build undirected neighbour list
    forward = edges_df.select(
        pl.col("src_id").alias("node"), pl.col("dst_id").alias("neighbor"),
    )
    backward = edges_df.select(
        pl.col("dst_id").alias("node"), pl.col("src_id").alias("neighbor"),
    )
    neighbors = pl.concat([forward, backward]).unique()

    # Self-join to get ordered neighbour pairs per node
    pairs = (
        neighbors.join(
            neighbors.rename({"neighbor": "neighbor2"}),
            on="node",
            how="inner",
        )
        .filter(pl.col("neighbor") < pl.col("neighbor2"))
    )

    if pairs.height == 0:
        return pl.DataFrame({"node_id": pl.Series([], dtype=pl.UInt32),
                             "temporal_clustering_coeff": pl.Series([], dtype=pl.Float32)})

    # Check which neighbour pairs are themselves connected
    edge_set = neighbors.select(
        pl.col("node").alias("neighbor"),
        pl.col("neighbor").alias("neighbor2"),
    )

    connected = pairs.join(
        edge_set,
        on=["neighbor", "neighbor2"],
        how="semi",  # keep rows from pairs that have a match
    )

    triangles = connected.group_by("node").agg(
        pl.len().alias("triangles"),
    )
    total_pairs = pairs.group_by("node").agg(
        pl.len().alias("total_pairs"),
    )

    clustering = (
        total_pairs.join(triangles, on="node", how="left")
        .with_columns(
            (pl.col("triangles").fill_null(0) / pl.col("total_pairs"))
            .cast(pl.Float32)
            .alias("temporal_clustering_coeff"),
        )
        .select(pl.col("node").alias("node_id"), "temporal_clustering_coeff")
    )
    return clustering


def compute_motif_features(
    edges_df: pl.DataFrame,
    config: MotifConfig,
) -> pl.DataFrame:
    """Compute all 6 temporal motif features per entity.

    Args:
        edges_df: Edge DataFrame with columns ``src_id``, ``dst_id``,
            ``step``, ``amount`` (integer node IDs and numeric step/amount).
        config: Motif configuration controlling time windows and thresholds.

    Returns:
        DataFrame with columns ``[node_id, t2_cycle_count, t3_cycle_count,
        fan_out_burst_count, fan_in_burst_count, ping_pong_ratio,
        temporal_clustering_coeff]``.  One row per entity.
    """
    if edges_df.height == 0:
        return pl.DataFrame({
            "node_id": pl.Series([], dtype=pl.UInt32),
            **{col: pl.Series([], dtype=pl.Float32) for col in MOTIF_FEATURE_COLS},
        })

    # Collect all unique node IDs
    all_node_ids = pl.concat([
        edges_df.select(pl.col("src_id").alias("node_id")),
        edges_df.select(pl.col("dst_id").alias("node_id")),
    ]).unique()

    logger.info("Computing temporal motif features for %d entities...", all_node_ids.height)

    # 1. Temporal 2-cycles
    logger.info("  Computing 2-cycle counts (window=%d)...", config.cycle_window)
    t2 = _compute_t2_cycles(edges_df, config.cycle_window)

    # 2. Temporal 3-cycles
    logger.info("  Computing 3-cycle counts (window=%d)...", config.cycle_window)
    t3 = _compute_t3_cycles(edges_df, config.cycle_window)

    # 3. Fan-out bursts
    logger.info(
        "  Computing fan-out bursts (window=%d, min_targets=%d)...",
        config.burst_window, config.burst_min_targets,
    )
    fan_out = _compute_fan_out_bursts(
        edges_df, config.burst_window, config.burst_min_targets,
    )

    # 4. Fan-in bursts
    logger.info(
        "  Computing fan-in bursts (window=%d, min_sources=%d)...",
        config.burst_window, config.burst_min_targets,
    )
    fan_in = _compute_fan_in_bursts(
        edges_df, config.burst_window, config.burst_min_targets,
    )

    # 5. Ping-pong ratio
    logger.info("  Computing ping-pong ratio (window=%d)...", config.cycle_window)
    pp = _compute_ping_pong_ratio(edges_df, config.cycle_window)

    # 6. Temporal clustering coefficient
    logger.info("  Computing temporal clustering coefficient...")
    tc = _compute_temporal_clustering(edges_df)

    # Join all features onto the full node set
    result = all_node_ids
    for df, _ in [
        (t2, "t2_cycle_count"),
        (t3, "t3_cycle_count"),
        (fan_out, "fan_out_burst_count"),
        (fan_in, "fan_in_burst_count"),
        (pp, "ping_pong_ratio"),
        (tc, "temporal_clustering_coeff"),
    ]:
        result = result.join(df, on="node_id", how="left")

    # Fill nulls (entities without the pattern get 0)
    result = result.fill_null(0.0)

    logger.info("Temporal motif features computed: %d entities, %d features.",
                result.height, len(MOTIF_FEATURE_COLS))

    return result
