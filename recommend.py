"""Phase 4 entry point: Similarity Search.

Given an uploaded image, finds and prints the top-K visually similar
products from the catalog, using the baseline embeddings from Phase 3.

Usage:
    python recommend.py --image path/to/query.jpg
    python recommend.py --image path/to/query.jpg --top-k 10
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.feature_extraction.backbone import build_backbone
from src.feature_extraction.embedder import load_embeddings
from src.retrieval.cosine_index import CosineSimilarityIndex
from src.retrieval.query_pipeline import RecommendationPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main(config_path: str, image_path: str, top_k: int | None) -> None:
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

    logger.info("Loading catalog embeddings from %s", config["embeddings"]["baseline_path"])
    ids, embeddings = load_embeddings(config["embeddings"]["baseline_path"])

    logger.info("Building cosine similarity index (%d items)", len(ids))
    index = CosineSimilarityIndex()
    index.build(ids, embeddings)

    logger.info("Loading backbone: %s", backbone_name)
    model = build_backbone(backbone_name, image_size)

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

    print(f"\nTop {len(results)} visually similar products for: {image_path}\n")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. [{result.similarity_score:.4f}] {result.product_display_name} "
            f"({result.article_type}, {result.base_colour}) - id={result.product_id}, "
            f"file={result.image_filename}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Similarity Search")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to query image")
    parser.add_argument("--top-k", type=int, default=None, help="Override config's retrieval.top_k")
    args = parser.parse_args()
    main(args.config, args.image, args.top_k)
