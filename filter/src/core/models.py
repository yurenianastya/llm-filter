import logging
import numpy as np
import torch
import scann

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.config import DEVICE, HF_MODEL, HF_TOKEN, SEMANTIC_MODEL
from src.utils.data import load_toxic_texts

logger = logging.getLogger(__name__)


def init_semantic_model():
    """
    Initializes the semantic model and ScaNN index for similarity search.
    Returns the SentenceTransformer model and a ScaNN searcher.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(SEMANTIC_MODEL)
        texts = load_toxic_texts()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device
        ).astype(np.float32)  # ScaNN requires float32

        # Build ScaNN index
        searcher = scann.scann_ops_pybind.builder(
            vectors, 5, "dot_product"  # k=5 nearest neighbors
        ).tree(
            num_leaves=200, num_leaves_to_search=10, training_sample_size=min(250000, len(vectors))
        ).score_ah(
            2, anisotropic_quantization_threshold=0.2
        ).build()

        return model, searcher
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