import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import evaluate

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer

# Load dataset
wnut = load_dataset("wnut_17", trust_remote_code=True)
# print(wnut)
# print(wnut["train"][0])

# Convert "ner_tags" from label index to label names
label_list = wnut["train"].features[f"ner_tags"].feature.names
# print(label_list)

# Define checkpoint
checkpoint = 'distilbert/distilbert-base-uncased'

# Define tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Pre-process for tokenizer
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply function over entire dataset
tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
        
# Dynamically padding with DataCollator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Define metric
seqeval = evaluate.load("seqeval")

# Compute accuracy score
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    
# Define 2 maps: id2label and label2id
id2label = {
    0: "O",
    1: "B-corporation",
    2: "I-corporation",
    3: "B-creative-work",
    4: "I-creative-work",
    5: "B-group",
    6: "I-group",
    7: "B-location",
    8: "I-location",
    9: "B-person",
    10: "I-person",
    11: "B-product",
    12: "I-product",
}
label2id = {
    "O": 0,
    "B-corporation": 1,
    "I-corporation": 2,
    "B-creative-work": 3,
    "I-creative-work": 4,
    "B-group": 5,
    "I-group": 6,
    "B-location": 7,
    "I-location": 8,
    "B-person": 9,
    "I-person": 10,
    "B-product": 11,
    "I-product": 12,
}

# Define model
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint, num_labels=13, id2label=id2label, label2id=label2id
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="my_first_ner",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()

print("TRAINING SUCCESSFULLY!")