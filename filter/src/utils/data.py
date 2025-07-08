import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

def load_toxic_texts(threshold=0.7, data_path='/data'):
    try:
        dataset = load_dataset("jigsaw_toxicity_pred", data_dir=data_path, split="train", trust_remote_code=True)
        texts = [
            x['comment_text'] for x in dataset
            if max(
                x.get('toxic', 0),
                x.get('severe_toxic', 0),
                x.get('obscene', 0),
                x.get('threat', 0),
                x.get('insult', 0),
                x.get('identity_hate', 0)
            ) > threshold
        ]
        logger.info("Loaded %d toxic texts", len(texts))
        return texts
    except Exception as e:
        logger.exception("Failed to load toxic dataset: %s", e)
        return []
