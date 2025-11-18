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
        
        # Use HuggingFace model - UPDATE THIS with your HuggingFace model ID after uploading
        # Format: "your-username/model-name"
        # Example: "bisharababish/arabert-toxic-classifier"
        model_path = os.getenv("HUGGINGFACE_MODEL_ID", "bisharababish/arabert-toxic-classifier")
        
        # Try local path first (for development), then fallback to HuggingFace
        local_path = "./arabert_toxic_classifier"
        model_file = f"{local_path}/model.safetensors"
        
        # Check if local model exists and is valid
        if os.path.exists(model_file) and os.path.getsize(model_file) > 1000:
            print(f"Loading model from local path: {local_path}")
            model_path = local_path
        else:
            print(f"Local model not found, loading from HuggingFace: {model_path}")
            print("Note: Model will be downloaded and cached on first load")
        
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            _model = AutoModelForSequenceClassification.from_pretrained(model_path)
            _model.eval()  # Set to evaluation mode
            _model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model: {e}")
            
            if "404" in error_msg or "not found" in error_msg.lower():
                raise ValueError(
                    f"Model not found on HuggingFace: {model_path}\n"
                    f"Please:\n"
                    f"1. Upload your model to HuggingFace (see HUGGINGFACE-SETUP.md)\n"
                    f"2. Update HUGGINGFACE_MODEL_ID environment variable or model_path in code\n"
                    f"3. Or ensure local model files exist in ./arabert_toxic_classifier/"
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

