import os
import torch
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_MODEL = os.environ.get('HF_MODEL')
SEMANTIC_MODEL = os.environ.get('SEMANTIC_MODEL')

torch.set_float32_matmul_precision('high')

keys = {
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit"
}