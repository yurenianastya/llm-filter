import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv(find_dotenv())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.environ.get('HF_TOKEN')
# Used model for classification: Intel/toxic-prompt-roberta
HF_MODEL = os.environ.get('HF_MODEL')

# Set precision for float32 operations
torch.set_float32_matmul_precision('high')

def load_toxic_texts():
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
    return toxic_texts


def load_semantic_model():
    model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
    toxic_texts = load_toxic_texts()
    vectors = model.encode(toxic_texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return model, index


def load_classifier_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL, token=HF_TOKEN).to(DEVICE)
    if torch.cuda.is_available():
        model = model.half()
    
    model.eval()
    return tokenizer, model


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
        return { "label": label, "score": float(score) }
    except Exception as e:
        print(f"Error during classification: {e}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def check_toxic_text_semantic(text: str):
    input_vec = semantic_model.encode([text], normalize_embeddings=True)
    D, _ = semantic_index.search(input_vec, k=5)
    similarity = float(np.mean(D[0]))
    return { "score": similarity }


def is_message_safe(text: str):
    semantic_result = check_toxic_text_semantic(text)
    classification_result = classify_text(text)

    if classification_result["label"] == 'TOXIC':
        status = False
    elif classification_result["score"] <= 0.2:
        status = False
    elif semantic_result["score"] >= 0.45:
        status = False
    else:
        status = True

    response = {
        "status": status,
        "classification_result": {
            "label": classification_result["label"],
            "score": classification_result["score"],
        },
        "semantic_result": {
            "score": semantic_result["score"]
        }
    }

    return response