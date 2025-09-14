import logging
import numpy as np
import torch
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.config import DEVICE, HF_MODEL, HF_TOKEN, SEMANTIC_MODEL
from src.utils.data import load_toxic_texts

logger = logging.getLogger(__name__)


def init_semantic_model():
    """
    Initializes the semantic model and FAISS index for similarity search.
    """
    try:
        model = SentenceTransformer(SEMANTIC_MODEL)
        texts = load_toxic_texts()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return model, index
    except Exception as e:
        logger.exception("Semantic model init failed: %s", e)
        raise

def init_classifier_model(model_name: str = HF_MODEL, token: str = HF_TOKEN):
    """
    Initializes any Hugging Face sequence classification model and tokenizer.

    Args:
        model_name: Hugging Face model name.
        token: Optional authentication token for private HF models.

    Returns:
        tokenizer, model
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token).to(DEVICE)
        
        if torch.cuda.is_available():
            model = model.half()
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.exception("Classifier model init failed: %s", e)
        raise