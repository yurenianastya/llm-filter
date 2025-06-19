import os
import torch
import numpy as np
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.environ.get('HF_TOKEN')
# Used model for classification: Intel/toxic-prompt-roberta
HF_MODEL = os.environ.get('HF_MODEL')

torch.set_float32_matmul_precision('high')

def load_toxic_texts():
    try:
        logger.info("Loading Jigsaw toxicity dataset")
        dataset = load_dataset("jigsaw_toxicity_pred", data_dir='./datasets', split="train", trust_remote_code=True)
        toxic_texts = [
            x['comment_text'] for x in dataset
            if max(
                x.get('toxic', 0),
                x.get('severe_toxic', 0),
                x.get('obscene', 0),
                x.get('threat', 0),
                x.get('insult', 0),
                x.get('identity_hate', 0)
            ) > 0.7
        ]
        logger.info("Loaded %d toxic texts", len(toxic_texts))
        return toxic_texts
    except Exception as e:
        logger.exception("Failed to load toxic dataset: %s", e)
        return []


def load_semantic_model():
    try:
        logger.info("Loading semantic embedding model")
        model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
        toxic_texts = load_toxic_texts()
        vectors = model.encode(toxic_texts, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        logger.info("Semantic model and FAISS index ready")
        return model, index
    except Exception as e:
        logger.exception("Failed to initialize semantic model or index: %s", e)
        raise


def load_classifier_model():
    try:
        logger.info("Loading classifier model: %s", HF_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, token=HF_TOKEN)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL, token=HF_TOKEN).to(DEVICE)

        if torch.cuda.is_available():
            logger.info("Running in CUDA, converting model to half precision")
            model = model.half()

        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.exception("Failed to load classification model: %s", e)
        raise


semantic_model, semantic_index = load_semantic_model()
tokenizer, classifier_model = load_classifier_model()


def classify_text(text: str):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(DEVICE)

        with torch.no_grad():
            outputs = classifier_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label_idx = torch.argmax(probs, dim=1).item()
        score = probs[0][label_idx].item()
        label = classifier_model.config.id2label.get(label_idx, str(label_idx))

        logger.debug("Classification result: label=%s, score=%.4f", label, score)
        return { "label": label, "score": float(score) }
    except Exception as e:
        logger.exception("Error during classification: %s", e)
        return { "label": "ERROR", "score": 0.0 }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def check_toxic_text_semantic(text: str):
    try:
        input_vec = semantic_model.encode([text], normalize_embeddings=True)
        D, _ = semantic_index.search(input_vec, k=5)
        similarity = float(np.mean(D[0]))
        logger.debug("Semantic similarity score: %.4f", similarity)
        return { "score": similarity }
    except Exception as e:
        logger.exception("Semantic search failed: %s", e)
        return { "score": 0.0 }


def is_message_safe(text: str):
    logger.info("Evaluating message for safety")
    semantic_result = check_toxic_text_semantic(text)
    classification_result = classify_text(text)

    score = classification_result["score"]
    label = classification_result["label"]
    similarity = semantic_result["score"]
    
    if label == 'TOXIC':
        status = False
    elif score <= 0.2:
        status = False
    elif similarity >= 0.45:
        status = False
    else:
        status = True
    
    logger.info("Final filtering decision: status=%s, label=%s, score=%.3f, similarity=%.3f",
                status, label, score, similarity)

    return {
        "status": status,
        "classification_result": {
            "label": label,
            "score": score,
        },
        "semantic_result": {
            "score": similarity
        }
    }
