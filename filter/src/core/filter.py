from typing import Dict
from collections import Counter
import re, unicodedata

import logging
import torch
import numpy as np

from src.core.models import init_classifier_model, init_semantic_model
from src.utils.config import DEVICE, KEYS

logger = logging.getLogger(__name__)

semantic_model, semantic_index = init_semantic_model()
tokenizer, classifier_model = init_classifier_model()

def classification_score(
    text: str,
    hf_tokenizer,
    hf_classifier_model,
    device: str,
    selected_keys: set
) -> Dict[str, float]:
    """
    Returns classification scores for a given text using a Hugging Face multi-label model.
    
    Args:
        text: Input text to classify.
        hf_tokenizer: HF tokenizer.
        hf_classifier_model: HF multi-label classification model.
        device: 'cpu' or 'cuda'.
        selected_keys: Optional subset of labels to include in output.
    
    Returns:
        Dictionary mapping labels to probabilities (0–1).
    """
    if selected_keys is None:
        selected_keys = KEYS

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = classifier_model(**inputs)

        # Multi-label: sigmoid activation
        probs = torch.sigmoid(outputs.logits)[0]
        labels = classifier_model.config.id2label

        full_scores = {labels[i]: float(probs[i].item()) for i in range(len(probs))}
        filtered_scores = {k: float(f"{full_scores[k]:.6f}") for k in selected_keys if k in full_scores}

        return filtered_scores

    except Exception as e:
        logger.exception("Classification error: %s", e)
        return {k: 0.0 for k in selected_keys}


def semantic_score(text: str) -> Dict[str, float]:
    try:
        vec = semantic_model.encode([text], normalize_embeddings=True)
        D, _ = semantic_index.search(vec, k=5)
        return { "score": float(np.mean(D[0])) }
    except Exception as e:
        logger.exception("Semantic search error: %s", e)
        return { "score": 0.0 }


def is_recurrent(text: str) -> bool:
    """
    for test purposes we won't allow text
    where single word appears more that 20% of the time
    """
    max_repetition_ratio = 0.2
    words = text.lower().split()
    if not words:
        return False
    word_counts = Counter(words)
    _, freq = word_counts.most_common(1)[0]
    ratio = freq / len(words)
    if ratio > max_repetition_ratio:
        return True
    return False

def character_anomalies(text: str) -> float:
    """
    Returns a single anomaly score (0–1) based on character-level irregularities.
    Higher values mean more anomalous content.
    """
    length = len(text) or 1

    punctuation_ratio = sum(c in "!?." for c in text) / length
    caps_ratio = sum(c.isupper() for c in text) / length
    repeat_chars = 1.0 if re.search(r"(.)\1{3,}", text) else 0.0
    non_printable = 1.0 if any(unicodedata.category(c).startswith("C") for c in text) else 0.0
    symbol_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / length

    # Weighted aggregation
    score = (
        0.2 * punctuation_ratio +
        0.2 * caps_ratio +
        0.2 * symbol_ratio +
        0.2 * repeat_chars +
        0.2 * non_printable
    )

    return min(score, 1.0)

def mixed_script_ratio(text: str) -> float:
    """
    Returns the ratio of tokens containing mixed scripts (e.g., Latin + Cyrillic).
    Higher values indicate more mixed-script content.
    """
    if not text:
        return 0.0

    def script_of_char(c):
        if 'LATIN' in unicodedata.name(c, ''):
            return 'LATIN'
        if 'CYRILLIC' in unicodedata.name(c, ''):
            return 'CYRILLIC'
        return 'OTHER'

    tokens = text.split()
    mixed_count = 0

    for token in tokens:
        scripts = set(script_of_char(c) for c in token if c.isalpha())
        if len(scripts) > 1:
            mixed_count += 1

    return mixed_count / max(len(tokens), 1)


def is_safe(text: str) -> Dict[str, float]:
    """
    Returns overall safety status for the input text combining classification,
    semantic similarity, and repetition checks.
    """
    classification = classification_score(
        text,
        hf_tokenizer=tokenizer,
        hf_classifier_model=classifier_model,
        device=DEVICE,
        selected_keys={"toxic", "severe_toxic", "obscene", "insult"}
    ) if text else {}
    semantic = semantic_score(text)
    recurrent = is_recurrent(text)
    anomalies = character_anomalies(text)
    mixed_text = mixed_script_ratio(text)
    toxic_flag = any(classification.get(label, 0) > 0.5 for label in KEYS)

    status = not (
        toxic_flag
        or semantic["score"] >= 0.45
        or recurrent 
        or anomalies > 0.4
        or mixed_text > 0.35
    )

    return {
        "status": status,
        "classification_result": classification,
        "semantic_result": semantic["score"],
        "is_recurrent_result": recurrent,
        "anomaly_result": anomalies,
        "mixed_language_result": mixed_text
    }
