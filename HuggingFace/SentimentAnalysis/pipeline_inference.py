import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
from transformers import pipeline

text = "I hate this film"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier = pipeline("sentiment-analysis", model="hoangph3003/my_first_model", device=device)
result = classifier(text)
print(result)