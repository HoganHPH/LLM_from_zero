import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import torch
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, AdamW
import evaluate


# Define GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ", device)

# Define check point
checkpoint = "bert-base-uncased"

# Define model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Load dataset
quora = load_dataset("quora", trust_remote_code=True)
print(quora)
print(quora["train"][0])

# Pre-process data
def tokenize_function(example):
    questions = example["questions"]
    t1 = []
    t2 = []
    for t in questions:
        t1.append(t["text"][0])
        t2.append(t["text"][1])
    return tokenizer(t1, t2, truncation=True)

# Apply pre-processing for entire dataset
tokenized_quora = quora["train"].map(tokenize_function, batched=True)

# Drop original columns, cast the boolean type to ClassLabel, 
# Rename is_duplicate to labels and 
# Split data into train (80%) and test (20%)
new_features = tokenized_quora.features.copy()
new_features["is_duplicate"] = ClassLabel(num_classes=2, names=['not_duplicate', 'duplicate'], names_file=None, id=None)
tokenized_quora = tokenized_quora.cast(new_features)
tokenized_quora = tokenized_quora.remove_columns("questions")
tokenized_quora = tokenized_quora.rename_column("is_duplicate", "labels")
tokenized_quora = tokenized_quora.train_test_split(test_size=0.2)
print(tokenized_quora)


# Dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define metric
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="qqp_v2",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    fp16=True,
    push_to_hub=True,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_quora["train"],
    eval_dataset=tokenized_quora["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Run training and push model to HF hub
print("\n====> Training:\n")
trainer.train()
trainer.push_to_hub()
print("\n====> TRAINING SUCCESSUFFULLY!!!")
