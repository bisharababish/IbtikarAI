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
        use_huggingface = False
        
        if not os.path.exists(model_file) or (os.path.exists(model_file) and os.path.getsize(model_file) == 0):
            print("Model file missing or empty (Git LFS issue), will download from HuggingFace...")
            # If local model is missing/empty, transformers will automatically download from HuggingFace
            # if we provide a valid model identifier. For now, try local first.
            use_huggingface = True
        
        try:
            # Try loading from local path first
            if not use_huggingface:
                _tokenizer = AutoTokenizer.from_pretrained(model_path)
                _model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                # If local fails, transformers will raise an error
                # You can catch it and use HuggingFace model name here
                print("Attempting to load from local path...")
                _tokenizer = AutoTokenizer.from_pretrained(model_path)
                _model = AutoModelForSequenceClassification.from_pretrained(model_path)
            _model.eval()  # Set to evaluation mode
            _model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model from local path: {e}")
            
            # If local loading fails and it's a file/JSON error, try HuggingFace
            if "Expecting value" in error_msg or "No such file" in error_msg or "empty" in error_msg.lower():
                print("Local model files are corrupted/missing. You need to:")
                print("1. Upload the model files to the repository, OR")
                print("2. Update the code with your HuggingFace model identifier")
                print("   Example: model_path = 'your-username/arabert-toxic-classifier'")
                raise ValueError(f"Model files are missing or corrupted. {error_msg}")
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

