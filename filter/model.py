import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_token = os.environ.get('HF_TOKEN')

tokenizer = AutoTokenizer.from_pretrained(
    os.environ.get('HF_MODEL'), use_auth_token=hf_token
    )
model = AutoModelForSequenceClassification.from_pretrained(
    os.environ.get('HF_MODEL'), use_auth_token=hf_token
    ).to(DEVICE)

if torch.cuda.is_available():
    model = model.half()

model.eval()

def print_gpu_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def classify_message(message: str):
    try:
        print_gpu_usage()
        inputs = tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label_idx = torch.argmax(probs, dim=1).item()
        score = probs[0][label_idx].item()
        label = model.config.id2label[label_idx] if hasattr(model.config, 'id2label') else str(label_idx)

        print_gpu_usage()
        return [{"label": label, "score": score}]
    except Exception as e:
        print(f"Error during classification: {e}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
