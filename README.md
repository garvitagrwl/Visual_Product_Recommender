# Visual Product Recommendation Engine

An image-based recommendation system: upload a fashion product photo,
get back visually similar items from a catalog — using deep visual
embeddings, not category/keyword matching.

Three approaches are implemented and compared on the same evaluation
pipeline:
1. **Baseline** - pretrained ResNet50 (ImageNet), no fine-tuning
2. **Transfer Learning** - ResNet50 fine-tuned via a temporary classification head
3. **Siamese Network** - trained directly on triplet (anchor/positive/negative) loss

## Results (full validation set, 480 queries)

| Stage | Precision@5 | Precision@10 | Mean Latency |
|---|---|---|---|
| Baseline | 0.9225 | 0.9123 | 796ms |
| Transfer Learning | 0.9400 | 0.9321 | 786ms |
| Siamese | **0.9475** | **0.9431** | 762ms |

Siamese > Transfer Learning > Baseline, consistently, across the full
validation set. (Recall@K is small - 0.015-0.032 - by construction:
each category has ~200-300 catalog images, so finding even a perfect
top-10 caps recall around 3-4%. Precision@K is the metric that
reflects retrieval quality here.) Full breakdown in
`reports/evaluation_results.csv` and `reports/evaluation_comparison.png`.

---

## Quick Start (using the already-trained model)

This repo ships with pre-computed embeddings, trained model weights,
and evaluation results already committed — **you do NOT need to
retrain anything** to use the app. You only need the raw dataset
images, since the app needs real photos to embed a query and to
display results.

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd visual-product-recommendation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the dataset (NOT included in git - see why below)
This project uses the Kaggle "Fashion Product Images Dataset"
(~44,000 images, 13.5GB). It's excluded from git deliberately - it's
a large, public dataset anyone can re-download, and committing it
would make every clone/pull download 13.5GB for no reason.

**Fastest option - Kaggle API:**
```bash
pip install kaggle
# Get an API token: Kaggle account -> Settings -> API -> Create New Token
# This downloads kaggle.json - place it at ~/.kaggle/kaggle.json (Linux/Mac)
# or C:\Users\<you>\.kaggle\kaggle.json (Windows)

kaggle datasets download -d paramaggarwal/fashion-product-images-dataset -p dataset --unzip
```
Confirm afterward that `dataset/images/`, `dataset/styles.csv`, and
`dataset/images.csv` exist (folder names inside the Kaggle zip may
need a quick rename/move to match this layout - check
`configs/config.yaml`'s `dataset:` section if paths don't line up).

### 4. Run the app
```bash
streamlit run app.py
```
Upload any product photo and get recommendations - no training step needed.

**Or use the CLI directly:**
```bash
python recommend.py --image path/to/your/image.jpg --stage siamese
```
(`--stage` can be `baseline`, `transfer_learning`, or `siamese` - see Results above)

---

## Full Reproduction (retraining everything from scratch)

Only needed if you want to regenerate the committed artifacts
yourself (e.g. with different hyperparameters, a different backbone,
or to verify the pipeline end-to-end). This repeats all the
training this repo's committed artifacts already represent.

Note: `train.py`'s two stages involve real backprop and are slow on
CPU - a GPU (e.g. Google Colab's free tier) is recommended for these
two steps specifically. Everything else runs fine locally/CPU.

```bash
# 1. Rebuild the dataset subset + train/val split (fast, CPU-only)
python prepare_dataset.py --config configs/config.yaml

# 2. Generate baseline embeddings (fast, CPU-only, no training)
python build_embeddings.py --stage baseline

# 3. Fine-tune via classification head (slow - use GPU/Colab)
python train.py --stage transfer_learning
python build_embeddings.py --stage transfer_learning

# 4. Train the Siamese network (slow - use GPU/Colab)
python train.py --stage siamese
python build_embeddings.py --stage siamese

# 5. Run the full evaluation (Precision@K, Recall@K, latency, comparison table+chart)
python evaluate.py
```

If running steps 3-4 on Colab: clone this repo there too, run steps
1-2 first (fast, establishes `data/*.csv`), then `train.py`, then
pull `models/*.weights.h5`, `models/*_history.json`, and
`embeddings/*_embeddings.npz` + their `*_metadata.json` files back
into your local repo before committing.

---

## Project Structure
```
visual-product-recommendation/
├── configs/config.yaml        # all paths, hyperparameters, category subset
├── dataset/                   # raw Kaggle data (gitignored - see Quick Start step 3)
├── data/                      # generated splits/triplets (committed)
├── embeddings/                # generated embeddings, one .npz + metadata per stage (committed)
├── models/                    # trained weights + training history per stage (committed)
├── reports/                   # evaluation comparison table (CSV) + chart (PNG)
├── src/
│   ├── preprocessing/         # metadata loading, validation, subsetting, splitting, image transforms
│   ├── feature_extraction/    # backbone loading, embedding generation
│   ├── retrieval/             # similarity search index, query pipeline
│   ├── training/              # fine-tuning + Siamese model builders, trainers, losses
│   ├── evaluation/             # Precision@K/Recall@K metrics, evaluator, reporting
│   └── utils/                 # config, logging, seeding
├── prepare_dataset.py         # Phase 1
├── build_embeddings.py        # Phases 3/6/7 (--stage baseline/transfer_learning/siamese)
├── train.py                   # Phases 6/7 (--stage transfer_learning/siamese)
├── recommend.py                # Phase 4 CLI (--stage to pick which model)
├── app.py                     # Phase 5 Streamlit UI
└── evaluate.py                 # Phase 8
```

## Evaluation Methodology Note
Precision@K/Recall@K use **same category as the query** as the
relevance criterion, since no manually-labeled visual-similarity
ground truth exists for this dataset. This is a standard
approximation for category-subsetted retrieval evaluation, but it's
worth stating plainly: a same-category result isn't guaranteed to be
visually similar, and vice versa.