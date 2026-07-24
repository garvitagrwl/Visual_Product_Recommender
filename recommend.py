"""Similarity Search entry point - query against baseline,
transfer_learning, or siamese embeddings, selected via --stage.

Critical detail: the query image must be embedded with the SAME
backbone weights that generated the catalog embeddings it's being
compared against. --stage controls both which embeddings file loads
AND (via build_embeddings.load_stage_model) which backbone weights
embed the query - the two scripts share this logic so they can never
drift apart on which weights a given stage means.

Usage:
    python recommend.py --image path/to/query.jpg --stage baseline
    python recommend.py --image path/to/query.jpg --stage transfer_learning
    python recommend.py --image path/to/query.jpg --stage siamese
"""

from __future__ import annotations

import argparse

import pandas as pd

from build_embeddings import STAGE_CONFIG, load_stage_model
from src.feature_extraction.embedder import load_embeddings
from src.retrieval.cosine_index import CosineSimilarityIndex
from src.retrieval.query_pipeline import RecommendationPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main(config_path: str, image_path: str, stage: str, top_k: int | None) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    logger = get_logger(
        "recommend",
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
    )

    top_k = top_k or config["retrieval"]["top_k"]
    backbone_name = config["model"]["backbone"]
    image_size = tuple(config["preprocessing"]["image_size"])

    keys = STAGE_CONFIG[stage]
    embeddings_path = config["embeddings"][keys["output_path"]]

    logger.info("Stage: %s | Loading catalog embeddings from %s", stage, embeddings_path)
    ids, embeddings = load_embeddings(embeddings_path)

    logger.info("Building cosine similarity index (%d items)", len(ids))
    index = CosineSimilarityIndex()
    index.build(ids, embeddings)

    logger.info("Loading backbone: %s", backbone_name)
    model = load_stage_model(config, stage, backbone_name, image_size, logger)

    metadata_df = pd.read_csv(config["data"]["full_subset_csv"])
    metadata_df["id"] = metadata_df["id"].astype(str)

    pipeline = RecommendationPipeline(
        model=model,
        index=index,
        metadata_df=metadata_df,
        image_size=image_size,
        backbone=backbone_name,
    )

    logger.info("Querying with image: %s (top_k=%d)", image_path, top_k)
    results = pipeline.recommend(image_path, top_k)

    print(f"\n[{stage}] Top {len(results)} visually similar products for: {image_path}\n")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. [{result.similarity_score:.4f}] {result.product_display_name} "
            f"({result.article_type}, {result.base_colour}) - id={result.product_id}, "
            f"file={result.image_filename}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Similarity Search (baseline / transfer_learning / siamese)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to query image")
    parser.add_argument(
        "--stage",
        type=str,
        default="baseline",
        choices=list(STAGE_CONFIG.keys()),
        help="Which embeddings/weights to use for the query",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override config's retrieval.top_k")
    args = parser.parse_args()
    main(args.config, args.image, args.stage, args.top_k)