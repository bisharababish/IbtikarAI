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
        
        # Use HuggingFace model - can be overridden with HUGGINGFACE_MODEL_ID env var
        # Using base AraBERT model and configuring it for binary classification
        model_path = os.getenv("HUGGINGFACE_MODEL_ID", "aubmindlab/bert-base-arabertv2")
        
        # Try local path first (for development), then fallback to HuggingFace
        local_path = "./arabert_toxic_classifier"
        model_file = f"{local_path}/model.safetensors"
        
        # Check if local model exists and is valid (not a Git LFS pointer)
        if os.path.exists(model_file) and os.path.getsize(model_file) > 1000000:  # > 1MB
            print(f"Loading model from local path: {local_path}")
            model_path = local_path
        else:
            print(f"Local model not found or invalid, loading from HuggingFace: {model_path}")
            print("Note: Model will be downloaded and cached on first load")
        
        try:
            print("Loading tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("Tokenizer loaded successfully!")
            
            # Load model with proper configuration for binary classification
            print("Loading model...")
            from transformers import AutoConfig
            
            try:
                # Try loading as-is first
                _model = AutoModelForSequenceClassification.from_pretrained(model_path)
                print("Model loaded as sequence classification model")
            except Exception as e:
                # If it fails, configure base model for classification
                print(f"Configuring base model for binary classification...")
                config = AutoConfig.from_pretrained(model_path)
                config.num_labels = 2
                config.id2label = {0: "safe", 1: "toxic"}
                config.label2id = {"safe": 0, "toxic": 1}
                _model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, 
                    config=config,
                    ignore_mismatched_sizes=True  # In case of size mismatches
                )
                print("Model configured and loaded successfully!")
            
            # Set to evaluation mode and disable gradients to save memory
            _model.eval()
            for param in _model.parameters():
                param.requires_grad = False
            
            # Move to CPU (Render free tier doesn't have GPU)
            if torch.cuda.is_available():
                _model = _model.cuda()
            else:
                _model = _model.cpu()
            
            _model_loaded = True
            print("Model loaded and ready!")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model: {error_msg}")
            import traceback
            print(traceback.format_exc())
            raise ValueError(f"Failed to load model: {error_msg}")
    
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
        try:
            model, tokenizer = get_model()
        except Exception as model_error:
            error_msg = str(model_error)
            # Return proper error response instead of crashing
            return AnalysisResponse(
                toxic=False,
                confidence=0.0,
                message=f"Model loading error: {error_msg[:200]}"  # Truncate long errors
            )
        
        # Tokenize input
        inputs = tokenizer(
            request.text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get toxic probability (binary classification: 0=safe, 1=toxic)
        if predictions.shape[1] >= 2:
            toxic_prob = predictions[0][1].item()  # Probability of class 1 (toxic)
        else:
            # Single output, treat as toxic probability
            toxic_prob = predictions[0][0].item()
        
        is_toxic = toxic_prob > 0.5
        
        return AnalysisResponse(
            toxic=is_toxic,
            confidence=toxic_prob,
            message="Analysis complete"
        )
    except Exception as e:
        # Catch any other errors and return proper response
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in analyze_text: {error_details}")
        return AnalysisResponse(
            toxic=False,
            confidence=0.0,
            message=f"Error: {str(e)[:200]}"
        )

