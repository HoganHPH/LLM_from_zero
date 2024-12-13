import numpy as np

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load data in the first time
imdb = load_dataset("imdb")

print(imdb)
print(imdb['train'][0])

# Preprocess

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


# Processing by tokenize and truncate
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)


# Map the tokenizer on entire dataset using batch to speed up
tokenized_imdb = imdb.map(preprocess_function, batched=True)


# Padding to the longest length of the BATCH (not the max length of the whole dataset)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Define metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {
    0: "NEGATIVE", 
    1: "POSITIVE"
}

label2id = {
    "NEGATIVE": 0, 
    "POSITIVE": 1
}


# Define model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)


# Define training arguments
training_args = TrainingArguments(
    output_dir="my_first_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.push_to_hub()

