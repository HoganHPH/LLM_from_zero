import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize

import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

###
### Load dataset
###
train = pd.read_csv('./DATA/train.csv')
test = pd.read_csv('./DATA/test.csv')
val = pd.read_csv('./DATA/val.csv')

data = [train, test, val]

# print(data[0])

###
### Pre-processing
###

# Handle sentiment value

for i in range(len(data)):
    data[i]['sentiment'] = data[i]['sentiment'].apply(lambda x: 'positive' if x not in ['neutral', 'negative'] else x)
    

# Encode category to class id
candidate_encoder = LabelEncoder()
party_encoder = LabelEncoder()
sentiment_encoder = LabelEncoder()

for i in range(len(data)):
    data[i] = data[i].drop(columns = ['tweet_id', 'user_handle', 'timestamp'])
    data[i]['candidate'] = candidate_encoder.fit_transform(data[i]['candidate'])
    data[i]['party'] = party_encoder.fit_transform(data[i]['party'])
    data[i]['sentiment'] = sentiment_encoder.fit_transform(data[i]['sentiment'])
    
# print(data[0])

# print(f'Candidate Encoder Labels: {candidate_encoder.classes_}')
# print(f'Party Encoder Labels: {party_encoder.classes_}')
# print(f'Sentiment Encoder Labels: {sentiment_encoder.classes_}')


# Preprocess text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text.strip().lower()

for i in range(len(data)):
    data[i]['tweet_text'] = data[i]['tweet_text'].apply(clean_text)
    
# print(data[0])


###
### Define tokenizer
###
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

###
### Create Dataset instance
###
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation = True, padding = 'max_length', max_length = self.max_len, return_tensors = 'pt')
        return { 'input_ids': encoding['input_ids'].flatten(),
                 'attention_mask': encoding['attention_mask'].flatten(),
                 'labels': torch.tensor(label, dtype = torch.long) }
        
datasets = []

for i in range(len(data)):
    datasets.append(TweetDataset(data[i]['tweet_text'].tolist(), data[i]['sentiment'].tolist(), tokenizer))
    
train_dataset = datasets[0]
test_dataset = datasets[1]
val_dataset = datasets[2]

###
### Define DataLoader
###
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 16)

###
### Define Model
###
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

###
### Define optimizer
###
optimizer = AdamW(model.parameters(), lr = 5e-5)


###
### Training
###
for epoch in range(3):
    print(f"\nEpoch {epoch + 1}")
    model.train()
    total_loss = 0.
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    loss_avg = total_loss / len(train_loader)
    print("Loss : ", loss_avg)


###
### Validation
###
def evaluate(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels']
            outputs = model(input_ids, attention_mask = attention_mask)
            preds = torch.argmax(F.softmax(outputs.logits, dim = -1), dim = -1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.numpy())
    print(classification_report(true_labels, predictions, target_names = sentiment_encoder.classes_))

print("Validation Results:")
evaluate(model, val_loader)

print("Test Results:")
evaluate(model, DataLoader(test_dataset, batch_size = 16))


###
### Confution Matrix
###
all_preds = []
all_labels = []
all_pred_probs = []

dataloader = DataLoader(train_dataset + test_dataset + val_dataset, batch_size = 16)

for batch in dataloader:
    input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
    outputs = model(input_ids, attention_mask = attention_mask)
    
    pred_probs = torch.softmax(outputs.logits, dim = -1).detach().cpu().numpy()
    preds = torch.argmax(F.softmax(outputs.logits, dim = -1), dim = -1).cpu().numpy()
    
    all_pred_probs.extend(pred_probs)
    all_preds.extend(preds)
    all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Negative", "Neutral", "Positive"])
disp.plot(cmap = "Blues")
plt.xlabel("Predicted Sentiment")
plt.ylabel("True Sentiment")
plt.title("Confusion Matrix")
plt.show()
plt.savefig("Confusion Matrix")

print(classification_report(all_labels, all_preds, target_names = ["Negative", "Neutral", "Positive"]))