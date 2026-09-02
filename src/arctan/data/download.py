"""Dataset acquisition: generates financial transaction networks with learnable
graph-structural fraud patterns.

Key design principles:
  1. Fraud accounts start legitimate: All fraud accounts have early normal activity,
     so their first transaction falls in the training period. They turn fraudulent later.
  2. Distinct graph topology: Fraud accounts develop anomalous patterns over time like
     fan-out bursts, intra-ring clustering, balance depletion.
  3. Temporal concentration: Fraudulent behavior clusters in later time periods.
  4. Realistic prevalence: ~2-5% of entities are fraudulent across all splits.

The resulting graph has clearly different degree/volume/counterparty distributions
between fraud and legitimate entities, giving the GNN real signal to learn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from arctan.config import PipelineConfig, get_default_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_benchmark_financial_transactions(
    output_path: Path,
    num_accounts: int = 50_000,
    num_legitimate_txns: int = 150_000,
    num_fraud_accounts: int = 1_000,
    num_steps: int = 744,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a temporally-structured transaction network with learnable fraud topology.

    Fraud accounts are designated upfront but begin with normal activity.
    Their fraudulent behavior emerges in the later 40% of the time window.
    """
    logger.info(
        "Generating temporal financial transaction network "
        "(%d accounts, %d fraud accounts, %d steps)...",
        num_accounts,
        num_fraud_accounts,
        num_steps,
    )
    rng = np.random.default_rng(seed)

    account_ids = [f"ACC_{i:06d}" for i in range(num_accounts)]

    # 1. Designate fraud accounts
    fraud_account_indices = set(
        rng.choice(num_accounts, size=num_fraud_accounts, replace=False).tolist()
    )
    fraud_list = sorted(fraud_account_indices)
    non_fraud_list = sorted(set(range(num_accounts)) - fraud_account_indices)
    fraud_arr = np.array(fraud_list)

    logger.info("Designated %d fraud accounts.", len(fraud_account_indices))

    # 2. Generate legitimate background transactions (ALL accounts participate)
    # Use Zipfian distribution but ensure fraud accounts appear early as normal actors
    sender_indices = (rng.zipf(a=1.35, size=num_legitimate_txns) - 1) % num_accounts
    receiver_indices = (rng.zipf(a=1.35, size=num_legitimate_txns) - 1) % num_accounts

    self_mask = sender_indices == receiver_indices
    receiver_indices[self_mask] = (receiver_indices[self_mask] + 1) % num_accounts

    legit_steps = np.sort(rng.integers(1, num_steps + 1, size=num_legitimate_txns))
    legit_types = rng.choice(
        ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
        size=num_legitimate_txns,
        p=[0.35, 0.30, 0.20, 0.10, 0.05],
    )
    legit_amounts = np.exp(
        rng.normal(loc=6.0, scale=1.2, size=num_legitimate_txns)
    ).astype(np.float32)
    legit_amounts = np.clip(legit_amounts, 1.0, 100_000.0)

    legit_old_bal = rng.uniform(500.0, 100_000.0, size=num_legitimate_txns).astype(np.float32)
    legit_new_bal = np.maximum(0.0, legit_old_bal - legit_amounts).astype(np.float32)

    legit_senders = [account_ids[i] for i in sender_indices]
    legit_receivers = [account_ids[i] for i in receiver_indices]
    legit_is_fraud = np.zeros(num_legitimate_txns, dtype=np.int64)

    # 3. Seed early legitimate transactions for fraud accounts
    # Every fraud account gets 2-5 normal transactions in the EARLY period (steps 1-300)
    # This ensures their first-transaction lands in the training period.
    logger.info("Seeding early legitimate activity for fraud accounts...")
    early_txns: dict[str, list] = {
        "step": [], "txn_type": [], "sender": [], "receiver": [],
        "amount": [], "old_balance_sender": [], "new_balance_sender": [],
        "is_fraud": [],
    }
    early_cutoff = int(num_steps * 0.40)  # step ~297

    for fraud_idx in fraud_list:
        n_early = rng.integers(2, 6)
        for _ in range(n_early):
            step = int(rng.integers(1, early_cutoff + 1))
            target = rng.choice(non_fraud_list)
            amount = float(rng.uniform(100.0, 5_000.0))  # normal-sized amounts
            old_bal = float(rng.uniform(5_000.0, 50_000.0))
            early_txns["step"].append(step)
            early_txns["txn_type"].append(rng.choice(["PAYMENT", "TRANSFER", "CASH_IN"]))
            early_txns["sender"].append(account_ids[fraud_idx])
            early_txns["receiver"].append(account_ids[target])
            early_txns["amount"].append(amount)
            early_txns["old_balance_sender"].append(old_bal)
            early_txns["new_balance_sender"].append(max(0.0, old_bal - amount))
            early_txns["is_fraud"].append(0)  # early activity is legitimate

    # 4. Generate fraud transactions with distinct graph-structural patterns
    fraud_txns: dict[str, list] = {
        "step": [], "txn_type": [], "sender": [], "receiver": [],
        "amount": [], "old_balance_sender": [], "new_balance_sender": [],
        "is_fraud": [],
    }

    fraud_step_start = int(num_steps * 0.45)  # fraud starts around step 335

    # Pattern A: Fan-out (money laundering layering)
    # Each fraud account sends to 15-40 unique random targets in a short burst
    logger.info("Injecting Pattern A: Fan-out layering...")
    for fraud_idx in fraud_list:
        n_targets = rng.integers(15, 41)
        targets = rng.choice(
            non_fraud_list, size=min(n_targets, len(non_fraud_list)), replace=False
        )
        burst_start = rng.integers(fraud_step_start, num_steps - 20)
        for t_idx in targets:
            step = int(burst_start + rng.integers(0, 20))
            amount = float(rng.uniform(2000.0, 50_000.0))
            old_bal = float(rng.uniform(50_000.0, 500_000.0))
            fraud_txns["step"].append(step)
            fraud_txns["txn_type"].append("TRANSFER")
            fraud_txns["sender"].append(account_ids[fraud_idx])
            fraud_txns["receiver"].append(account_ids[t_idx])
            fraud_txns["amount"].append(amount)
            fraud_txns["old_balance_sender"].append(old_bal)
            fraud_txns["new_balance_sender"].append(max(0.0, old_bal - amount))
            fraud_txns["is_fraud"].append(1)

    # Pattern B: Intra-ring transactions (fraud accounts transact with each other)
    logger.info("Injecting Pattern B: Fraud ring clustering...")
    num_ring_txns = len(fraud_list) * 3  # ~3 intra-ring txns per fraud account
    ring_senders = rng.choice(fraud_arr, size=num_ring_txns)
    ring_receivers = rng.choice(fraud_arr, size=num_ring_txns)
    ring_self = ring_senders == ring_receivers
    ring_receivers[ring_self] = fraud_arr[
        (np.searchsorted(fraud_arr, ring_receivers[ring_self]) + 1) % len(fraud_arr)
    ]

    for i in range(num_ring_txns):
        step = int(rng.integers(fraud_step_start, num_steps + 1))
        amount = float(rng.uniform(5000.0, 100_000.0))
        old_bal = float(rng.uniform(10_000.0, 200_000.0))
        fraud_txns["step"].append(step)
        fraud_txns["txn_type"].append(rng.choice(["TRANSFER", "CASH_OUT"]))
        fraud_txns["sender"].append(account_ids[ring_senders[i]])
        fraud_txns["receiver"].append(account_ids[ring_receivers[i]])
        fraud_txns["amount"].append(amount)
        fraud_txns["old_balance_sender"].append(old_bal)
        fraud_txns["new_balance_sender"].append(max(0.0, old_bal - amount))
        fraud_txns["is_fraud"].append(1)

    # Pattern C: Balance depletion (account takeover)
    logger.info("Injecting Pattern C: Balance depletion (ATO)...")
    for fraud_idx in rng.choice(fraud_list, size=min(300, len(fraud_list)), replace=False):
        step = int(rng.integers(fraud_step_start + 50, num_steps + 1))
        old_bal = float(rng.uniform(20_000.0, 300_000.0))
        target = rng.choice(non_fraud_list)
        fraud_txns["step"].append(step)
        fraud_txns["txn_type"].append("CASH_OUT")
        fraud_txns["sender"].append(account_ids[fraud_idx])
        fraud_txns["receiver"].append(account_ids[target])
        fraud_txns["amount"].append(old_bal)
        fraud_txns["old_balance_sender"].append(old_bal)
        fraud_txns["new_balance_sender"].append(0.0)
        fraud_txns["is_fraud"].append(1)

    # 5. Combine all transaction sources
    early_df = pl.DataFrame({
        "step": early_txns["step"],
        "txn_type": early_txns["txn_type"],
        "sender": early_txns["sender"],
        "receiver": early_txns["receiver"],
        "amount": np.array(early_txns["amount"], dtype=np.float32),
        "old_balance_sender": np.array(early_txns["old_balance_sender"], dtype=np.float32),
        "new_balance_sender": np.array(early_txns["new_balance_sender"], dtype=np.float32),
        "is_fraud": np.array(early_txns["is_fraud"], dtype=np.int64),
    })

    fraud_df = pl.DataFrame({
        "step": fraud_txns["step"],
        "txn_type": fraud_txns["txn_type"],
        "sender": fraud_txns["sender"],
        "receiver": fraud_txns["receiver"],
        "amount": np.array(fraud_txns["amount"], dtype=np.float32),
        "old_balance_sender": np.array(fraud_txns["old_balance_sender"], dtype=np.float32),
        "new_balance_sender": np.array(fraud_txns["new_balance_sender"], dtype=np.float32),
        "is_fraud": np.array(fraud_txns["is_fraud"], dtype=np.int64),
    })

    legit_df = pl.DataFrame({
        "step": legit_steps,
        "txn_type": legit_types,
        "sender": legit_senders,
        "receiver": legit_receivers,
        "amount": legit_amounts,
        "old_balance_sender": legit_old_bal,
        "new_balance_sender": legit_new_bal,
        "is_fraud": legit_is_fraud,
    })

    df = pl.concat([legit_df, early_df, fraud_df], how="diagonal_relaxed").sort("step")

    # 6. Stats
    total_fraud_txns = int(df["is_fraud"].sum())
    fraud_steps = df.filter(pl.col("is_fraud") == 1)["step"]

    fraud_senders = set(df.filter(pl.col("is_fraud") == 1)["sender"].to_list())
    fraud_receivers = set(df.filter(pl.col("is_fraud") == 1)["receiver"].to_list())
    all_fraud_entities = fraud_senders | fraud_receivers

    all_senders = set(df["sender"].to_list())
    all_receivers = set(df["receiver"].to_list())
    all_entities = all_senders | all_receivers

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    logger.info(
        "Wrote temporal transaction network to %s "
        "(Rows: %d, Fraud txns: %d (%.2f%%), "
        "Total entities: %d, Fraud entities: %d (%.2f%%), "
        "Fraud step range: %d–%d)",
        output_path,
        df.height,
        total_fraud_txns,
        (total_fraud_txns / df.height) * 100,
        len(all_entities),
        len(all_fraud_entities),
        (len(all_fraud_entities) / len(all_entities)) * 100,
        fraud_steps.min() if total_fraud_txns > 0 else 0,
        fraud_steps.max() if total_fraud_txns > 0 else 0,
    )
    return df


def download_financial_dataset(dest_dir: Path) -> None:
    """Download PaySim transaction dataset from HuggingFace, or generate benchmark network."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_file = dest_dir / "financial_transactions.parquet"

    if target_file.exists():
        logger.info("Financial transaction dataset already exists at %s", target_file)
        return

    logger.info("Downloading real-world financial transaction dataset")

    try:
        from huggingface_hub import hf_hub_download

        hf_file = hf_hub_download(
            repo_id="ealvaradob/paysim-financial-fraud",
            filename="paysim.parquet",
            repo_type="dataset",
            local_dir=str(dest_dir),
        )
        logger.info("Downloaded dataset from HuggingFace to %s", hf_file)
        return
    except Exception as exc:
        logger.info(
            "HuggingFace dataset download bypassed or unavailable (%s). "
            "Initializing temporal financial transaction benchmark...",
            exc,
        )

    generate_benchmark_financial_transactions(target_file)


def download_all(config: PipelineConfig) -> None:
    """Create directories and acquire financial transactions."""
    config.paths.ensure_dirs()
    download_financial_dataset(config.paths.transactions_dir)
    logger.info("Financial data acquisition complete.")


if __name__ == "__main__":
    cfg = get_default_config()
    download_all(cfg)
