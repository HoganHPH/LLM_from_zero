import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, TrainingArguments, Trainer


# Load dataset
eli5 = load_dataset("eli5_category", split="train[:5000]", trust_remote_code=True)

# Train-Test split
eli5 = eli5.train_test_split(test_size=0.2)
print(eli5)
print(eli5['train'][0])

# Flatten to remove nest
eli5 = eli5.flatten()

# Define checkpoint
checkpoint = "distilbert/distilroberta-base"

# Define tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Pre-process

# Convert the list of strings to 1 string 
def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

# Apply preprocessing for entire dataset
tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)

# Second pre-processing

# Concat all sequences
# Split concatenated seq into shorter CHUNKS

block_size = 128

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder, we could add 
    # padding if the model supported it instead of this 
    # drop, you can customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    } 
    
    return result

# Apply second pre-processing for entire dataset
lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

# Data collator for dynamically padding and random mask token
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Training

# Define model
model = AutoModelForMaskedLM.from_pretrained(checkpoint)

# Define training arguments
training_args = TrainingArguments(
    output_dir="my_first_maskedlm",
    eval_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
print("Training succesffully!")

# Evaluate model
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.push_to_hub()
print("DONE!")