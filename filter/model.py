from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import HF_MODEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL).to(DEVICE)

if hasattr(torch, "compile"):
    model = torch.compile(model)

model.eval()

def classify_message(message: str):
    try:
        inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label_index = torch.argmax(probs, dim=1).item()
        score = probs[0][label_index].item()
        label = model.config.id2label[label_index] if hasattr(model.config, 'id2label') else str(label_index)

        return [{"label": label, "score": score}]
    
    except Exception as e:
        print(f"Error during classification: {e}")
        return None