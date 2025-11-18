# Upload Model to HuggingFace - Step by Step Guide

## Why HuggingFace?
- ✅ No Git LFS issues
- ✅ Automatic model caching
- ✅ Works perfectly with Transformers library
- ✅ Free hosting for public models
- ✅ Easy to update model versions

## Step 1: Get Your HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "IbtikarAI")
4. Select "Write" permissions
5. Copy the token (starts with `hf_...`)

## Step 2: Install HuggingFace Hub

```bash
pip install huggingface_hub
```

## Step 3: Login to HuggingFace

```bash
huggingface-cli login
```

Paste your token when prompted.

## Step 4: Create Model Repository on HuggingFace

1. Go to: https://huggingface.co/new
2. Choose "Model"
3. Repository name: `arabert-toxic-classifier` (or any name you prefer)
4. Visibility: **Public** (free) or Private (requires paid plan)
5. Click "Create repository"

## Step 5: Upload Your Model

From your IbtikarAI directory:

```bash
cd C:\Users\Leo\Desktop\ibtikar-backend-main\IbtikarAI
huggingface-cli upload YOUR_USERNAME/arabert-toxic-classifier ./arabert_toxic_classifier
```

Replace `YOUR_USERNAME` with your HuggingFace username.

**Example:**
```bash
huggingface-cli upload bisharababish/arabert-toxic-classifier ./arabert_toxic_classifier
```

## Step 6: Update Code (Already Done!)

The code is already configured to use HuggingFace. Just set the environment variable in Render:

**In Render Dashboard:**
- Go to your IbtikarAI service
- Environment → Add Environment Variable
- Key: `HUGGINGFACE_MODEL_ID`
- Value: `YOUR_USERNAME/arabert-toxic-classifier`

**Or** edit `ibtikar_api.py` line 25 and change:
```python
model_path = os.getenv("HUGGINGFACE_MODEL_ID", "bisharababish/arabert-toxic-classifier")
```

## Alternative: Python Script Upload

If CLI doesn't work, create `upload_model.py`:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./arabert_toxic_classifier",
    repo_id="YOUR_USERNAME/arabert-toxic-classifier",
    repo_type="model"
)
```

Then run:
```bash
python upload_model.py
```

## After Upload

1. Your model will be available at: `https://huggingface.co/YOUR_USERNAME/arabert-toxic-classifier`
2. The code will automatically download it on first request
3. No more Git LFS issues! 🎉
