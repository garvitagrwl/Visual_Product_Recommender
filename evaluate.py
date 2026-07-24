
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_embeddings import STAGE_CONFIG, load_stage_model
from src.evaluation.evaluator import RetrievalEvaluator
from src.evaluation.report import build_comparison_table, save_comparison_chart, save_comparison_table
from src.feature_extraction.embedder import load_embeddings
from src.retrieval.cosine_index import CosineSimilarityIndex
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main(config_path: str, sample_size: int | None) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    logger = get_logger(
        "evaluate",
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
    )

    logger.info("=== Phase 8: Evaluation ===")

    backbone_name = config["model"]["backbone"]
    image_size = tuple(config["preprocessing"]["image_size"])
    category_column = config["subset"]["category_column"]
    k_values = config["evaluation"]["k_values"]

    metadata_df = pd.read_csv(config["data"]["full_subset_csv"])
    metadata_df["id"] = metadata_df["id"].astype(str)

    query_df = pd.read_csv(config["data"]["val_csv"])
    if sample_size is not None and sample_size < len(query_df):
        query_df = query_df.sample(n=sample_size, random_state=config["seed"])
        logger.info("Using a sample of %d validation queries", sample_size)
    else:
        logger.info("Using all %d validation queries", len(query_df))

    results = []
    embedding_generation_times = {}

    for stage, keys in STAGE_CONFIG.items():
        embeddings_path = config["embeddings"][keys["output_path"]]
        metadata_path = config["embeddings"][keys["metadata_path"]]

        if not Path(embeddings_path).exists():
            logger.warning("Skipping stage '%s' - no embeddings found at %s", stage, embeddings_path)
            continue

        logger.info("--- Evaluating stage: %s ---", stage)

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                stage_metadata = json.load(f)
            embedding_generation_times[stage] = stage_metadata.get("generation_time_seconds")
        except FileNotFoundError:
            logger.warning(
                "No metadata file at %s for stage '%s' - embedding_generation_time_s "
                "will be blank for this stage, but evaluation will proceed normally.",
                metadata_path,
                stage,
            )
            embedding_generation_times[stage] = None

        ids, embeddings = load_embeddings(embeddings_path)
        index = CosineSimilarityIndex()
        index.build(ids, embeddings)

        model = load_stage_model(config, stage, backbone_name, image_size, logger)

        evaluator = RetrievalEvaluator(
            model=model,
            index=index,
            metadata_df=metadata_df,
            images_dir=config["dataset"]["images_dir"],
            image_size=image_size,
            backbone=backbone_name,
            category_column=category_column,
            logger=logger,
        )
        result = evaluator.evaluate(query_df, k_values)
        result.stage = stage
        results.append(result)

        logger.info(
            "Stage '%s': Precision@%d=%.4f, Recall@%d=%.4f, mean_latency=%.2fms",
            stage,
            k_values[0],
            result.precision_at_k[k_values[0]],
            k_values[0],
            result.recall_at_k[k_values[0]],
            result.mean_latency_ms,
        )

    if not results:
        logger.error("No stages had embeddings available - nothing to evaluate.")
        return

    table = build_comparison_table(results, embedding_generation_times, k_values)
    save_comparison_table(table, config["evaluation"]["results_csv"])
    save_comparison_chart(results, k_values, config["evaluation"]["chart_path"])

    print("\n=== Evaluation Comparison ===\n")
    print(table.to_string(index=False))
    print(f"\nSaved table to {config['evaluation']['results_csv']}")
    print(f"Saved chart to {config['evaluation']['chart_path']}")

    logger.info("=== Phase 8 complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8: Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Evaluate on a random sample of this many validation queries instead of all of them (faster)",
    )
    args = parser.parse_args()
    main(args.config, args.sample_size)
