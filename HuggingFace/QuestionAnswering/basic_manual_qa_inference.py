import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering



question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

checkpoint = "hoangph3003/my_first_qa_model"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(question, context, return_tensors="pt")

model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
with torch.no_grad():
    outputs = model(**inputs)
    
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
result = tokenizer.decode(predict_answer_tokens)

print(result)
    