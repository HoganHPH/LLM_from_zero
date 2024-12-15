import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


# Load the dataset, choose the subset "ca_test"
billsum = load_dataset("billsum", split='ca_test')

# Train-Test split
billsum = billsum.train_test_split(test_size=0.2)

print(billsum)
for key, value in billsum['train'][0].items():
    print("\n====> ", key)
    print(value)
    

# Load tokenizer    
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Create prefix as a prompt defining the task for LLM
prefix = "summrize this text: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Transform entire dataset
tokenized_billsum = billsum.map(preprocess_function, batched=True)

# Create batch and padding dynamically
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# Define metric
rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


# Define model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="my_first_summarization_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

# Define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
    
# Run training and push model to HF hub
print("\n====> Training:\n")
trainer.train()
trainer.push_to_hub()