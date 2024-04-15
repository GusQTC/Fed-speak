from transformers import BertModel, BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


import os

import random


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('google/bert_uncased_L-4_H-256_A-4', num_labels=1)


speeches_data = {}
speaches_month = {}

train_data = []
test_data = []
for file in os.listdir('speeches/reduced/'):
    if file.endswith('.txt'):

        month = file.split('.')[0][-5:-3]
        year = file.split('.')[0][-9:-5]
        yearmonth = year + month
        with open(os.path.join('speeches/reduced/', file), 'r', encoding= 'utf8') as f:
            speech = f.read()


        if yearmonth not in speeches_data:
            speeches_data[yearmonth] = []

        
        speeches_data[yearmonth].append(speech)

# Load the economic data and create the labels tensor
economic_df = pd.read_csv('economic_data.csv')
change_interest_rate = economic_df['Interest Rate'].diff()
labels = change_interest_rate.values

#Max sequence for BERT is 512
max_length = 512


# Tokenize and process the speeches
for yearmonth, speeches in speeches_data.items():
    # Tokenize and process each speech in the month
    for i, speech in enumerate(speeches):
        # Tokenize the speech text
        tokens = tokenizer.tokenize(speech)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

         # Truncate or pad the token_ids to the maximum sequence length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids = token_ids + [0] * (max_length - len(token_ids))
        
        # Convert tokens to a 1D tensor
        input_ids = torch.tensor(token_ids)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        # Get the label for the current speech
        label = labels[i]

        # Add the input_ids and label to the train_data or test_data
        if random.random() < 0.8:  # 80% of the data goes to the training set
            train_data.append((input_ids, label))
        else:  # 20% of the data goes to the test set
            test_data.append((input_ids, label))

# Separate the inputs and labels, and pad the inputs
train_inputs, train_labels = zip(*train_data)
train_inputs = pad_sequence(train_inputs, batch_first=True)
train_labels = torch.tensor(train_labels)

test_inputs, test_labels = zip(*test_data)
test_inputs = pad_sequence(test_inputs, batch_first=True)
test_labels = torch.tensor(test_labels)



# Separate the inputs and labels, and pad the inputs
train_inputs, train_labels = zip(*train_data)
train_inputs = pad_sequence(train_inputs, batch_first=True)
train_labels = torch.tensor(train_labels)

test_inputs, test_labels = zip(*test_data)
test_inputs = pad_sequence(test_inputs, batch_first=True)
test_labels = torch.tensor(test_labels)

# Set the learning rate and initialize the optimizer
learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Create TensorDatasets for training and testing data
train_dataset = TensorDataset(train_inputs, train_labels)
test_dataset = TensorDataset(test_inputs, test_labels)

# Define the batch size
batch_size = 16

# Create DataLoaders for training and testing data
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Define the loss functions
classification_loss_function = nn.CrossEntropyLoss()
regression_loss_function = nn.MSELoss()


# Train the model
max_epochs = 10
for epoch in range(max_epochs):
    print("Epoch: ", epoch)

    model.train()
    for batch in train_dataloader:
        batch_inputs, batch_labels = batch

        batch_inputs = batch_inputs.long() # input_ids must be of type long or Int
        batch_labels = batch_labels.float()

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=batch_inputs, labels=batch_labels)
        print(outputs)
        loss = outputs.loss
        # Compute the classification loss
        #classification_loss = classification_loss_function(outputs.logits, batch_labels)

        # Compute the regression loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
        print(f"Epoch {epoch+1}/{max_epochs}: Loss = {loss.item()}")

# Test the model
model.eval()
with torch.no_grad():
    # Forward pass for testing
    outputs = model(input_ids=test_inputs, labels=test_labels)
    test_loss = outputs.loss
    preds = outputs.logits
    test_accuracy = (preds == test_labels).float().mean()
    
print(f"Test Loss: {test_loss.item()}")
print(f"Test Accuracy: {test_accuracy.item()}")