import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import torch
from transformers import pipeline


# Define GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ", device)

# Define check point
checkpoint = "hoangph3003/qqp_v2"

pipe = pipeline("text-classification", model=checkpoint, tokenizer=checkpoint, device=device)

input_pairs = [
    {"text": "How can I be successful in Kaggle Competitions?", "text_pair": "How can I be successful in life?"},
    {"text": "What is the best place to eat a pizza in Italy?", "text_pair": "What is the best restaurant in Italy?"},
    {"text": "What are the good courses to learn pytorch?", "text_pair": "Are there good courses to learn pytorch?"}
]

predictions = pipe(input_pairs)

# Extract predicted labels
preds = [int(pred['label'].split('_')[-1]) for pred in predictions]
print(preds)
