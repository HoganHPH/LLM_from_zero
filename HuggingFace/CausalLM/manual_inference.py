import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prompt = "Somatic hypermutation allows the immune system to"

# Define checkpoint
checkpoint = "hoangph3003/my_first_clm_model"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt").input_ids

model = AutoModelForCausalLM.from_pretrained(checkpoint)
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

# Decode the generated token ids back into text
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)