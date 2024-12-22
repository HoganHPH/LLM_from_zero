from transformers import pipeline


text = "The Golden State Warriors are an American professional basketball team based in San Francisco."

checkpoint = 'hoangph3003/my_first_ner'

classifier = pipeline("ner", model=checkpoint)
results = classifier(text)
print(results)
