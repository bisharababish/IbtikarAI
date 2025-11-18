# Smaller Arabic Models for 512MB Free Tier

## Recommended Models (Smaller & Faster)

### 1. **BERT Multilingual Toxicity Classifier** ⭐ RECOMMENDED
- **Model ID**: `textdetox/bert-multilingual-toxicity-classifier`
- **Size**: ~400-500 MB (smaller than AraBERT)
- **Pros**: 
  - Pre-trained for toxicity classification
  - Supports Arabic + 14 other languages
  - Already fine-tuned for toxic/non-toxic
  - Works well on limited resources
- **Cons**: Multilingual (may be slightly less accurate than Arabic-specific)

### 2. **DistilBERT Multilingual**
- **Model ID**: `distilbert-base-multilingual-cased`
- **Size**: ~250-300 MB (much smaller!)
- **Pros**: 
  - 50% smaller than BERT
  - Faster inference
  - Supports Arabic
- **Cons**: 
  - Not pre-trained for toxicity
  - Needs configuration for classification

### 3. **AraBERTv0.2-Twitter-base**
- **Model ID**: `aubmindlab/bert-base-arabertv02-twitter`
- **Size**: ~400-500 MB
- **Pros**: 
  - Arabic-specific
  - Fine-tuned on Twitter data
  - Good for social media toxicity
- **Cons**: Still relatively large

### 4. **MARBERTv2** (if available)
- **Model ID**: `UBC-NLP/MARBERTv2`
- **Size**: ~400-500 MB
- **Pros**: 
  - Arabic-specific
  - Good performance
- **Cons**: May not be pre-trained for toxicity

## Size Comparison

| Model | Size | Memory Usage | Speed | Accuracy |
|-------|------|--------------|-------|----------|
| `aubmindlab/bert-base-arabertv2` | ~500-600 MB | High | Slow | High |
| `textdetox/bert-multilingual-toxicity-classifier` | ~400-500 MB | Medium | Medium | High |
| `distilbert-base-multilingual-cased` | ~250-300 MB | Low | Fast | Medium |
| `aubmindlab/bert-base-arabertv02-twitter` | ~400-500 MB | Medium | Medium | High |

## Best Choice for 512MB Free Tier

**`textdetox/bert-multilingual-toxicity-classifier`** is the best option because:
1. ✅ Pre-trained for toxicity (no fine-tuning needed)
2. ✅ Smaller than full AraBERT
3. ✅ Supports Arabic
4. ✅ Should fit in 512MB with PyTorch

## How to Switch

Update line 34 in `ibtikar_api.py`:
```python
model_path = os.getenv("HUGGINGFACE_MODEL_ID", "textdetox/bert-multilingual-toxicity-classifier")
```

Or set environment variable in Render:
- Key: `HUGGINGFACE_MODEL_ID`
- Value: `textdetox/bert-multilingual-toxicity-classifier`

