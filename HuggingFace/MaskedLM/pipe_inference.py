import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import pipeline

checkpoint = 'hoangph3003/my_first_maskedlm'

text = "A dog has <mask> eyes"

# Define pipeline
mask_filler = pipeline("fill-mask", checkpoint)
result = mask_filler(text, top_k=3)
print(result)