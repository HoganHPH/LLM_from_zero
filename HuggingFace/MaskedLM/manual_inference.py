import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

checkpoint = 'hoangph3003/my_first_maskedlm'

text = "A dog has<mask> eyes"

# Define tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

# Define model
model = AutoModelForMaskedLM.from_pretrained(checkpoint)

logits = model(**inputs).logits
mask_token_logits = logits[0, mask_token_index, :]

# Results
top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0]

for token in top_3_tokens:
    print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))