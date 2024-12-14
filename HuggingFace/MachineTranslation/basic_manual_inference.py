import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


text = "translate English to French: Good afternoon. Nice to meet you!"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("hoangph3003/my_first_translation_model")
inputs = tokenizer(text, return_tensors='pt').input_ids

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("hoangph3003/my_first_translation_model")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

# Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
