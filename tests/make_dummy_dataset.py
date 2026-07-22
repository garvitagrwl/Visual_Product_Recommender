"""Generates a small synthetic dataset mimicking the Kaggle Fashion
Product Images Dataset's structure, so Phase 1 can be verified end to
end without downloading the full ~44k-image dataset first.

Usage:
    python tests/make_dummy_dataset.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image

CATEGORIES = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Handbags",
    "Watches",
    "Sunglasses",
    "Dresses",
]

# Deliberately uneven and slightly below/above the configured min/max
# so the validator's and sampler's edge-case handling gets exercised.
COUNTS_PER_CATEGORY = {
    "Tshirts": 260,
    "Shirts": 240,
    "Casual Shoes": 310,  # above max (300) -> should get capped
    "Sports Shoes": 220,
    "Handbags": 205,
    "Watches": 250,
    "Sunglasses": 230,
    "Dresses": 215,
}

N_MISSING_IMAGE_ROWS = 15  # rows with metadata but no image file on disk
N_NULL_FIELD_ROWS = 8      # rows with a null required field
N_DUPLICATE_ROWS = 5       # duplicate ids


def main() -> None:
    random.seed(42)
    root = Path("dataset")
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    next_id = 10000

    for category in CATEGORIES:
        for _ in range(COUNTS_PER_CATEGORY[category]):
            pid = next_id
            next_id += 1
            color = tuple(random.randint(0, 255) for _ in range(3))
            Image.new("RGB", (224, 224), color=color).save(images_dir / f"{pid}.jpg")
            rows.append(
                {
                    "id": pid,
                    "gender": random.choice(["Men", "Women", "Unisex"]),
                    "masterCategory": "Apparel",
                    "subCategory": category,
                    "articleType": category,
                    "baseColour": random.choice(["Black", "White", "Blue", "Red"]),
                    "season": random.choice(["Summer", "Winter", "Fall"]),
                    "usage": "Casual",
                    "productDisplayName": f"Dummy {category} {pid}",
                }
            )

    df = pd.DataFrame(rows)

    # Inject rows referencing images that don't exist on disk.
    for i in range(N_MISSING_IMAGE_ROWS):
        pid = next_id
        next_id += 1
        row = df.iloc[0].copy()
        row["id"] = pid
        row["productDisplayName"] = f"Missing image {pid}"
        df = pd.concat([df, row.to_frame().T], ignore_index=True)
        # Note: intentionally NOT writing an image file for this id.

    # Inject rows with a null required field.
    null_indices = df.sample(n=N_NULL_FIELD_ROWS, random_state=1).index
    df.loc[null_indices, "articleType"] = None

    # Inject duplicate ids.
    dup_rows = df.sample(n=N_DUPLICATE_ROWS, random_state=2)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df.to_csv(root / "styles.csv", index=False)

    images_csv_rows = [
        {"filename": f"{pid}.jpg", "link": f"http://example.com/{pid}.jpg"}
        for pid in df["id"].unique()
    ]
    pd.DataFrame(images_csv_rows).to_csv(root / "images.csv", index=False)

    print(f"Dummy dataset created under {root.resolve()}")
    print(f"  styles.csv: {len(df)} rows (incl. injected invalid rows)")
    print(f"  images/: {len(list(images_dir.glob('*.jpg')))} image files")


if __name__ == "__main__":
    main()
