import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


# Load dataset
books = load_dataset("opus_books", "en-fr")
books = books['train'].train_test_split(test_size=0.2)
print(books)
print(books['train'][0])


# Load tokenizer from pretrain
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Pre-process
source_lang = "en"
target_lang = "fr"
prefix = "translate English to French:" # used as a promt to define the specified task for model

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples['translation']]
    targets = [example[target_lang] for example in examples['translation']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


tokenized_books = books.map(preprocess_function, batched=True)


# Padding sentence with dynamic padding (max_length of each batch instead of the whole dataset)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# Define metric
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decode_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# Define model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="my_first_translation_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True
)


# Define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


print("====> START TRAINING!")
trainer.train()
trainer.push_to_hub()
print("====> MODEL IS SAVED!")





    
    


    

