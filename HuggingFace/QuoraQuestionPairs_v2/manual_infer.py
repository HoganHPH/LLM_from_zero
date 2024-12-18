import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Define GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ", device)

# Define check point
checkpoint = "hoangph3003/qqp_v2"

# Define model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model = model.to(device)

tokens = tokenizer([
    ['How can I be successful in Kaggle Competitions?', 'How can I be successful in life?'],
    ['What is the best place to eat a pizza in Italy?','What is the best restaurant in Italy?'],
    ['What are the good courses to learn pytorch?','Are there good courses to learn pytorch?']],
    truncation=True, padding=True, return_tensors='pt')
tokens = tokens.to(device)

logits = model(**tokens).logits
logits = logits.cpu().detach().numpy()
preds = np.argmax(logits, axis=-1)
print(preds)
