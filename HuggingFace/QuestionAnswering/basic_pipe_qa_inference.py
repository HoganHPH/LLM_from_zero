import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import torch
from transformers import pipeline


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

checkpoint = "hoangph3003/my_first_qa_model"

question_answerer = pipeline("question-answering", model=checkpoint, device=device)
result = question_answerer(question=question, context=context)

print(result)