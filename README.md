# Visual Product Recommendation Engine

Image retrieval (not classification) system: upload a fashion product image,
get back visually similar products from the Kaggle Fashion Product Images
Dataset, using deep visual embeddings.

## Status: Phase 1 complete (Dataset Preparation)

## Setup
```bash
pip install -r requirements.txt
```

Download the Kaggle "Fashion Product Images Dataset" and extract it so
`dataset/images/`, `dataset/styles.csv`, and `dataset/images.csv` exist
(see configs/config.yaml for exact expected paths).

## Try it without the real dataset first
```bash
python tests/make_dummy_dataset.py   # generates a small synthetic dataset
python prepare_dataset.py            # runs Phase 1 against it
```

## Run Phase 1 on the real dataset
```bash
python prepare_dataset.py --config configs/config.yaml
```
Outputs: `data/subset_full.csv`, `data/train.csv`, `data/val.csv`,
`data/validation_report.json`.
