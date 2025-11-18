from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
import torch

app = FastAPI()

# Lazy loading: Model and tokenizer loaded only on first request
_model = None
_tokenizer = None
_model_loaded = False

def get_model():
    """Load model only when first needed (lazy loading)"""
    global _model, _tokenizer, _model_loaded
    
    if not _model_loaded:
        print("Loading model (first request)...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_path = "./arabert_toxic_classifier"
        
        # Check if model file exists and is not empty (not just a placeholder)
        model_file = f"{model_path}/model.safetensors"
        use_huggingface_fallback = False
        
        # Check if model file is missing or empty (Git LFS placeholder)
        if not os.path.exists(model_file) or (os.path.exists(model_file) and os.path.getsize(model_file) < 1000):
            print("Model file missing or empty (Git LFS issue)")
            print("Attempting to download model files from GitHub...")
            
            # Download model files from GitHub raw URLs
            import requests
            base_url = "https://raw.githubusercontent.com/bisharababish/IbtikarAI/main/arabert_toxic_classifier"
            
            os.makedirs(model_path, exist_ok=True)
            
            files_to_download = [
                "config.json",
                "model.safetensors", 
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.txt"
            ]
            
            for filename in files_to_download:
                file_path = f"{model_path}/{filename}"
                if not os.path.exists(file_path) or os.path.getsize(file_path) < 100:
                    print(f"Downloading {filename}...")
                    try:
                        response = requests.get(f"{base_url}/{filename}", stream=True)
                        if response.status_code == 200:
                            with open(file_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print(f"✓ Downloaded {filename}")
                        else:
                            print(f"⚠ Could not download {filename} (status: {response.status_code})")
                    except Exception as e:
                        print(f"⚠ Error downloading {filename}: {e}")
            
            use_huggingface_fallback = True
        
        try:
            # Try loading from local path
            print(f"Attempting to load model from: {model_path}")
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            _model = AutoModelForSequenceClassification.from_pretrained(model_path)
            _model.eval()  # Set to evaluation mode
            _model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model from local path: {e}")
            
            # If local loading fails and it's a file/JSON error, try HuggingFace fallback
            if ("Expecting value" in error_msg or "No such file" in error_msg or 
                "empty" in error_msg.lower() or "not found" in error_msg.lower()):
                
                if use_huggingface_fallback:
                    print("\n" + "="*60)
                    print("Local model files are missing/corrupted (Git LFS issue)")
                    print("="*60)
                    print("\nSOLUTION: You need to either:")
                    print("1. Upload the actual model files to GitHub (not via Git LFS)")
                    print("   - Remove model from Git LFS")
                    print("   - Commit the actual 541MB model file directly")
                    print("\n2. OR use a HuggingFace model:")
                    print("   - Upload your model to HuggingFace")
                    print("   - Update model_path in code to: 'your-username/model-name'")
                    print("\n3. OR download model at startup from cloud storage")
                    print("="*60 + "\n")
                
                raise ValueError(
                    f"Model files are missing or corrupted. "
                    f"The model file is empty (Git LFS placeholder). "
                    f"Please upload the actual model files to the repository or configure HuggingFace model path. "
                    f"Error: {error_msg}"
                )
            else:
                raise
    
    return _model, _tokenizer

class TextRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    toxic: bool
    confidence: float
    message: str

@app.get("/")
def root():
    return {"message": "IbtikarAI API - Model will load on first prediction request"}

@app.get("/health")
def health():
    """Health check endpoint - doesn't load model"""
    return {"status": "healthy", "model_loaded": _model_loaded}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest):
    """
    Analyze text for toxicity.
    Model is loaded lazily on first request to save memory at startup.
    """
    try:
        # Lazy load model on first request
        model, tokenizer = get_model()
        
        # Tokenize input
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get toxic probability (assuming binary classification)
        toxic_prob = predictions[0][1].item() if predictions.shape[1] > 1 else predictions[0][0].item()
        is_toxic = toxic_prob > 0.5
        
        return AnalysisResponse(
            toxic=is_toxic,
            confidence=toxic_prob,
            message="Analysis complete"
        )
    except Exception as e:
        return AnalysisResponse(
            toxic=False,
            confidence=0.0,
            message=f"Error: {str(e)}"
        )

