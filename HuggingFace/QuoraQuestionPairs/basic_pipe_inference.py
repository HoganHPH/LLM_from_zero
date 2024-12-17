import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch

from transformers import pipeline


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A pair of questions
question1: "What can make Physics easy to learn?"
question2: "How can you make physics easy to learn?"
query = "Which city is the capital of France?, Where is the capital of France?"


# Define checkpoint
checkpoint = "hoangph3003/my_first_qqp_model"

classifier = pipeline("text-classification", 
                      model=checkpoint, 
                      device=device, 
                      tokenizer=checkpoint, 
                      padding=True, 
                      truncation=True)

result = classifier(query)
print(result)
