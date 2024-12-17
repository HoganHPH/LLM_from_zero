import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load dataset with the subset of 10000 samples
glue = load_dataset("nyu-mll/glue", "qqp", split="train[:5000]")

# Train-Test split
glue = glue.train_test_split(test_size=0.2)

print(glue)
print(glue['train'][0])

# Define model checkpoint
checkpoint = "google-bert/bert-base-cased"

# Pre-processing

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Preprocess function
def preprocess_function(examples):
    return tokenizer(examples['question1'], examples['question2'], truncation=True, padding=True, max_length=512)

# Apply preprocessing to entire dataset
tokenized_glue = glue.map(preprocess_function, batched=True)

# Padding to the longest length of the BATCH (not the max length of the whole dataset)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Define Label-ClassIdx
id2label = {
    0: "not_duplicate", 
    1: "duplicate"
}

label2id = {
    "not_duplicate": 0, 
    "duplicate": 1
}

# Define model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                            num_labels=2,
                                                            id2label=id2label,
                                                            label2id=label2id)

# Define training arguments
training_args = TrainingArguments(
    output_dir='my_first_qqp_model',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_glue['train'],
    eval_dataset=tokenized_glue['test'],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.push_to_hub()
