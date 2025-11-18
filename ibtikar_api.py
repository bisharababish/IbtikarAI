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
        
        # Check if model directory exists, if not, download from HuggingFace
        if not os.path.exists(model_path) or not os.path.exists(f"{model_path}/model.safetensors"):
            print("Model not found locally, this is expected on Render (Git LFS issue)")
            print("You may need to download from HuggingFace or use a different approach")
            # Option: Download from HuggingFace if needed
            # model_path = "aubmindlab/bert-base-arabertv2"  # Example
        
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            _model = AutoModelForSequenceClassification.from_pretrained(model_path)
            _model.eval()  # Set to evaluation mode
            _model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
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

