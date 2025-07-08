import logging
import torch
import numpy as np

from src.core.models import init_classifier_model, init_semantic_model
from src.utils.config import DEVICE

logger = logging.getLogger(__name__)

semantic_model, semantic_index = init_semantic_model()
tokenizer, classifier_model = init_classifier_model()

def classify(text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = classifier_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        return {
            "label": classifier_model.config.id2label.get(idx, str(idx)),
            "score": float(probs[0][idx].item())
        }
    except Exception as e:
        logger.exception("Classification error: %s", e)
        return { "label": "ERROR", "score": 0.0 }

def semantic_score(text: str):
    try:
        vec = semantic_model.encode([text], normalize_embeddings=True)
        D, _ = semantic_index.search(vec, k=5)
        return { "score": float(np.mean(D[0])) }
    except Exception as e:
        logger.exception("Semantic search error: %s", e)
        return { "score": 0.0 }

def is_safe(text: str):
    classification = classify(text)
    semantic = semantic_score(text)
    score = classification["score"]
    label = classification["label"]
    sim = semantic["score"]
    status = not (label == 'TOXIC' or score <= 0.2 or sim >= 0.45)
    return {
        "status": status,
        "classification_result": classification,
        "semantic_result": semantic
    }
