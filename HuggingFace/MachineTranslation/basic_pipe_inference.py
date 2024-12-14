import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from transformers import pipeline


text = "translate English to French: Good afternoon. Nice to meet you!"

translator = pipeline("translation_xx_to_yy", model="hoangph3003/my_first_translation_model")
result = translator(text)
print(result)