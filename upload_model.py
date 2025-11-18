"""
Upload model to HuggingFace
This script will upload all model files to HuggingFace.
"""
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "Bisharababish/arabert-toxic-classifier"
folder_path = "./arabert_toxic_classifier"

print(f"Uploading model from {folder_path} to {repo_id}...")
print("This may take a while for large files...")

try:
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"\n✅ Success! Model uploaded to: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    if "model.safetensors" in str(e) or "137" in str(e):
        print("\n⚠️  The model.safetensors file appears to be a Git LFS placeholder (137 bytes).")
        print("You need to download the actual model file first.")
        print("\nOptions:")
        print("1. If you have the actual model file elsewhere, copy it to arabert_toxic_classifier/model.safetensors")
        print("2. Download it from your original source")
        print("3. If it's on GitHub, you may need to use Git LFS to pull it")

