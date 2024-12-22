import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification


text = "The Golden State Warriors are an American professional basketball team based in San Francisco."

checkpoint = 'hoangph3003/my_first_ner'

# Define tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(text, return_tensors="pt")

# Define model
model = AutoModelForTokenClassification.from_pretrained(checkpoint)
with torch.no_grad():
    logits = model(**inputs).logits
    
predictions = torch.argmax(logits, dim=2)
predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
print(predicted_token_class)