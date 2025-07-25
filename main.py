from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader

# === Init de l'app FastAPI ===
app = FastAPI()

# === Chargement du modèle fine-tuné ===
MODEL_PATH = "export_model"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# === Map des labels (à adapter si besoin) ===
label_map = {0: "négatif", 1: "neutre", 2: "positif", 3: "mitigé"}

# === Entrée API ===
class BatchRequest(BaseModel):
    texts: List[str]

# === Dataset pour prédiction par batch ===
class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=50):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }

# === Route de prédiction par lot ===
@app.post("/predict_batch")
def predict_batch(data: BatchRequest):
    dataset = SentimentDataset(data.texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32)

    preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1)
            preds.extend(batch_preds.tolist())

    labels = [label_map[p] for p in preds]
    return {"labels": labels}
