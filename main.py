import pandas as pd
import logging
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Load the Real and Fake News dataset from Kaggle
real_news_df = pd.read_csv("True.csv")
fake_news_df = pd.read_csv("Fake.csv")

# "text" column contains the news articles
real_news_texts = real_news_df["text"]
real_news_labels = [0] * len(real_news_texts)  # Label real news as 0

fake_news_texts = fake_news_df["text"]
fake_news_labels = [1] * len(fake_news_texts)  # Label fake news as 1

# Combine data from both sources
texts = pd.concat([real_news_texts, fake_news_texts]).tolist()
labels = real_news_labels + fake_news_labels

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize and encode the training and testing texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

# Create PyTorch datasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

# Define DataLoader for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3  # Can be adjusted

# Setup logging
logging.basicConfig(level=logging.INFO)

# Training loop with logging
for epoch in range(num_epochs):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:  # Log every 100 batches
            logging.info(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

logging.info("Training complete.")

# Testing loop
model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(test_labels, all_preds)
print(f"Accuracy: {accuracy * 100:.2f}%")

# TODO: test for training data leaks or any other indicators of overfitting
# TODO: add a confusion matrix and perform cross-validation