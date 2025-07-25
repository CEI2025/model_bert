from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader

app = FastAPI()

# === Chargement modèle/tokenizer depuis Hugging Face ===
MODEL_NAME = "CEI2025/sentiment_cei"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

label_map = {0: "négatif", 1: "neutre", 2: "positif", 3: "mitigé"}

class BatchRequest(BaseModel):
    texts: List[str]

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

@app.get("/")
def root():
    return {"message": "API de classification de sentiment en ligne !"}

@app.post("/predict_batch")
def predict_batch(data: BatchRequest):
    if not data.texts:
        return {"error": "Aucun texte fourni."}

    dataset = SentimentDataset(data.texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32)

    preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1)
            preds.extend(batch_preds.tolist())

    labels = [label_map[p] for p in preds]
    return {"labels": labels}
