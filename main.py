from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# Chargement du modèle et tokenizer depuis export_model
model = BertForSequenceClassification.from_pretrained("export_model")
tokenizer = BertTokenizer.from_pretrained("export_model")
model.eval()

label_map = {0: "négatif", 1: "neutre", 2: "positif", 3: "mitigé"}

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextIn):
    inputs = tokenizer(data.text, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
    return {
        "label": label_map[prediction],
        "confidence": round(probs[0][prediction].item(), 3)
    }
