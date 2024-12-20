import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from transformers import pipeline


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prompt = "Somatic hypermutation allows the immune system to"

# Define checkpoint
checkpoint = "hoangph3003/my_first_clm_model"

# Define model
generator = pipeline("text-generation", model=checkpoint, device=device)
result = generator(prompt)
print(result)
