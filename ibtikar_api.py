from pathlib import Path
from typing import List
import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="IbtikarAI Toxicity API")

# ---------- Models ----------

class TextsIn(BaseModel):
    texts: List[str]


def is_lfs_pointer(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as f:
            head = f.read(80)
        return b"git-lfs" in head
    except Exception:
        return False


def should_use_local_model(model_dir: Path) -> bool:
    expected = [
        model_dir / "config.json",
        model_dir / "tokenizer.json",
        model_dir / "tokenizer_config.json",
        model_dir / "special_tokens_map.json",
        model_dir / "model.safetensors",
    ]
    for p in expected:
        if not p.exists() or p.stat().st_size < 1024 or is_lfs_pointer(p):
            return False
    return True


LOCAL_MODEL_DIR = Path(__file__).parent / "arabert_toxic_classifier"

# Load tokenizer + model once on startup
if should_use_local_model(LOCAL_MODEL_DIR):
    model_source = str(LOCAL_MODEL_DIR)
else:
    # Fallback to HF model for development if local files are missing/LFS pointers
    # You can override via environment variable HF_MODEL_ID
    model_source = os.getenv("HF_MODEL_ID", "unitary/toxic-bert")

tokenizer = AutoTokenizer.from_pretrained(model_source)
model = AutoModelForSequenceClassification.from_pretrained(model_source)
model.eval()  # disable dropout etc.

# Try to find which output index is "toxic"
id2label = model.config.id2label or {}
toxic_index = None
for i, name in id2label.items():
    if "toxic" in str(name).lower():
        toxic_index = int(i)
        break

# Fallback: assume index 1 is toxic
if toxic_index is None:
    toxic_index = 1


@app.post("/predict")
def predict(inp: TextsIn):
    """
    Input:  { "texts": ["...", "..."] }
    Output: { "preds": [ {"label": "harmful"/"safe", "score": float}, ... ] }
    """

    if not inp.texts:
        return {"preds": []}

    enc = tokenizer(
        inp.texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**enc)
        probs = outputs.logits.softmax(dim=-1)

    preds = []
    for p in probs:
        p = p.cpu()
        toxic_prob = float(p[toxic_index])
        label = "harmful" if toxic_prob >= 0.5 else "safe"
        preds.append({"label": label, "score": toxic_prob})

    return {"preds": preds}
