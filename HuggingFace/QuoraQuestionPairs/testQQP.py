import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch

from transformers import pipeline

# A pair of questions
question_1 = "HOW OLD ARE YOU?"
question_2 = "how old are you"
input_text = "{} [SEP] {}".format(question_1, question_2)

# Define checkpoint
checkpoint = "google-bert/bert-base-cased"

# 1.
# Test inference by pipeline
#

# classifier = pipeline("text-classification", model=checkpoint)
# result = classifier(input_text)
# print(result)




classifier = pipeline("text-classification", model = "textattack/bert-base-uncased-QQP")
result = classifier("Which city is the capital of France?, Where is the capital of France?")
print(result)










