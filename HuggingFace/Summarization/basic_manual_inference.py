import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

text = "summarize this text: Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.[1] Advances in the field of deep learning have allowed neural networks to surpass many previous approaches in performance.[2]ML finds application in many fields, including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine.[3][4] The application of ML to business problems is known as predictive analytics. Statistics and mathematical optimization (mathematical programming) methods comprise the foundations of machine learning. Data mining is a related field of study, focusing on exploratory data analysis (EDA) via unsupervised learning.[6][7] From a theoretical viewpoint, probably approximately correct (PAC) learning provides a framework for describing machine learning."

checkpoint = "hoangph3003/my_first_summarization_model"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(text, return_tensors='pt').input_ids

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
