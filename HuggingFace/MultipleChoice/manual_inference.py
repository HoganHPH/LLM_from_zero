import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch

from transformers import AutoTokenizer, AutoModelForMultipleChoice


# Define text and candidate answers
prompt = "His name is Bob. He has a friend. Her name is Anna. She has a black dog"
candidate1 = "Bob's friend has a white dog"
candidate2 = "Bob has a white dog"
candidate3 = "Bob's friend has a black dog"
candidate4 = "Bob has a black dog"

# Define checkpoint
checkpoint = 'hoangph3003/my_first_multiplechoice_model'

# Define tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2], [prompt, candidate3], [prompt, candidate4]], 
                   return_tensors="pt", padding=True)
labels = torch.tensor(0).unsqueeze(0)

# Define model
model = AutoModelForMultipleChoice.from_pretrained(checkpoint)
outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
logits = outputs.logits

pridicted_class = logits.argmax().item()
print(pridicted_class)