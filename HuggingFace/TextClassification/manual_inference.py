import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


id2label = {
    0: "NEGATIVE", 
    1: "POSITIVE"
}

text = "This film is so boring"

tokenizer = AutoTokenizer.from_pretrained("hoangph3003/my_first_model")
inputs = tokenizer(text, return_tensors='pt')


model = AutoModelForSequenceClassification.from_pretrained("hoangph3003/my_first_model")
with torch.no_grad():
    logits = model(**inputs).logits
    
predicted_class_id = logits.argmax().item()
result = model.config.id2label[predicted_class_id]
print(result)

