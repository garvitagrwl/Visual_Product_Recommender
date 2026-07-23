"""Phase 5 entry point: Streamlit UI.

Upload an image, preview it, and see the top-K visually similar
products from the catalog - same pipeline as recommend.py (Phase 4),
just wrapped in a UI instead of the CLI.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.feature_extraction.backbone import build_backbone
from src.feature_extraction.embedder import load_embeddings
from src.retrieval.cosine_index import CosineSimilarityIndex
from src.retrieval.query_pipeline import RecommendationPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

CONFIG_PATH = "configs/config.yaml"


@st.cache_resource(show_spinner="Loading model and building similarity index...")
def load_pipeline(config_path: str) -> tuple[RecommendationPipeline, dict]:
    """Load everything the app needs once, cached across reruns.

    Returns:
        (pipeline, config) - config is returned alongside so the UI
        can read display settings (e.g. default top_k) without
        reloading the file.
    """
    config = load_config(config_path)
    set_seed(config["seed"])

    logger = get_logger(
        "app",
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
    )

    backbone_name = config["model"]["backbone"]
    image_size = tuple(config["preprocessing"]["image_size"])

    ids, embeddings = load_embeddings(config["embeddings"]["baseline_path"])
    index = CosineSimilarityIndex()
    index.build(ids, embeddings)

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
    logger.info("Pipeline loaded and cached: %d items in catalog", len(ids))
    return pipeline, config


def main() -> None:
    st.set_page_config(page_title="Visual Product Recommendation", layout="wide")
    st.title("Visual Product Recommendation Engine")
    st.caption(
        "Upload a product photo to find visually similar items in the catalog. "
        "Ranking is by visual similarity (learned image embeddings), not category tags."
    )

    pipeline, config = load_pipeline(CONFIG_PATH)
    images_dir = Path(config["dataset"]["images_dir"])

    top_k = st.sidebar.slider(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=config["retrieval"]["top_k"],
    )
    st.sidebar.caption(f"Backbone: {config['model']['backbone']} (baseline, not fine-tuned)")

    uploaded_file = st.file_uploader(
        "Upload a product image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload an image to get recommendations.")
        return

    col_query, col_spacer = st.columns([1, 3])
    with col_query:
        st.image(uploaded_file, caption="Query image", use_container_width=True)

    # Write to a temp file so we can reuse the exact same file-path-based
    # preprocessing function the CLI (recommend.py) uses - one code path,
    # not a duplicated in-memory variant.
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.spinner("Finding visually similar products..."):
            results = pipeline.recommend(tmp_path, top_k)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not results:
        st.warning("No matches found.")
        return

    st.subheader(f"Top {len(results)} visually similar products")
    columns = st.columns(min(len(results), 5))

    for i, result in enumerate(results):
        col = columns[i % len(columns)]
        with col:
            image_path = images_dir / result.image_filename
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning(f"Image file missing: {result.image_filename}")

            st.markdown(f"**{result.product_display_name}**")
            st.caption(f"{result.article_type} · {result.base_colour}")
            st.progress(
                min(max(result.similarity_score, 0.0), 1.0),
                text=f"Similarity: {result.similarity_score:.3f}",
            )


if __name__ == "__main__":
    main()
