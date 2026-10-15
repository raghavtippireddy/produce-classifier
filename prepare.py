"""Turn the raw Hugging Face parquet files into resized numpy arrays.

Source: huggingface.co/datasets/Nattakarn/fruit-and-vegetable-image-recognition
(a mirror of Kritik Seth's Kaggle "Fruits and Vegetables Image Recognition").
"""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
SIZE = 224

# The source labels have near-duplicates that are the same produce under
# different names. Merging them stops the model being graded on a coin flip.
MERGE = {
    "capsicum": "bell pepper",
    "paprika": "bell pepper",
    "sweetcorn": "corn",
    "jalepeno": "jalapeno",
    "raddish": "radish",
}


def to_square(img: Image.Image) -> Image.Image:
    """Same preprocessing the Lambda uses: fix rotation, RGB, center-crop to 224."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    return ImageOps.fit(img, (SIZE, SIZE), Image.Resampling.BILINEAR)


def load(split: str):
    files = sorted(RAW.glob(f"{split}-*.parquet"))
    tables = [pq.read_table(f) for f in files]
    names = json.loads(tables[0].schema.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    images, labels, hashes = [], [], []
    for t in tables:
        for row in t.to_pylist():
            raw = row["image"]["bytes"]
            try:
                img = to_square(Image.open(io.BytesIO(raw)))
            except Exception as e:  # a few web-scraped files are broken
                print(f"  skip unreadable image in {split}: {e}")
                continue
            images.append(np.asarray(img, dtype=np.uint8))
            labels.append(MERGE.get(names[row["label"]], names[row["label"]]))
            hashes.append(hashlib.md5(images[-1].tobytes()).hexdigest())
    return np.stack(images), labels, hashes


def main():
    x_tr, y_tr, h_tr = load("train")
    x_te, y_te, h_te = load("test")

    # Drop test images that also appear in train, so the test score is honest.
    seen = set(h_tr)
    keep = [i for i, h in enumerate(h_te) if h not in seen]
    print(f"test images also in train (removed): {len(h_te) - len(keep)} of {len(h_te)}")
    x_te, y_te = x_te[keep], [y_te[i] for i in keep]

    classes = sorted(set(y_tr))
    idx = {c: i for i, c in enumerate(classes)}
    np.savez(OUT / "train.npz", x=x_tr, y=np.array([idx[c] for c in y_tr]))
    np.savez(OUT / "test.npz", x=x_te, y=np.array([idx[c] for c in y_te]))
    (OUT / "labels.json").write_text(json.dumps(classes, indent=1))

    print(f"train: {len(x_tr)}  test: {len(x_te)}  classes: {len(classes)}")
    print("smallest classes in train:", Counter(y_tr).most_common()[-5:])


if __name__ == "__main__":
    main()
