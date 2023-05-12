import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from data_manipulation import preproc
from time import perf_counter, strftime, gmtime
from sklearn.metrics import classification_report

# a headline bag
# tokenize with tok_fn
# where too long for the model, take LAST headlines 
# up to 1000 headliens, max_embed padded 500, so take LAST 500 tokens
# return tokenization
def tokenize_from_end(sample, tok_fn, max_embed=500):
    max_embed = max_embed -25 # offset
    toks = sample.split()
    if len(toks) >= max_embed:
        toks = toks[-(max_embed-2):]
        sample = " ".join(toks)
    r = tok_fn( sample, 
                max_length=max_embed+25, 
                padding='max_length',
                truncation=True, 
                return_tensors='pt')
    return r

# A Dataset for serving up tokenized headlines and winner columns 
# returning input_ids, attention_mask, labels
class ClassificationDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_seq_len: int=500):
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        X = row['headlines']
        y = row['winner'].astype(int)
        
        # Tokenize the input sequence
        encoding= tokenize_from_end(X, self.tokenizer, max_embed=self.max_seq_len)
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Convert the label to a tensor
        label = torch.tensor(y, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }


# Define the model architecture, just BERT and forward its logits
class BERTBinaryClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits


def train_bert(train, dev, test, epochs=8):
    '''predefined bert retraining setup with train/dev/test df intake'''
    batch_size      = 9
    learning_rate   = 0.0007
    num_epochs      = epochs
    max_seq_len     = 510
    device          = 'cuda'
    print('BERT model retraining...')
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # turn train dev test into Datasets
    train_dataset   = ClassificationDataset(train, tokenizer, max_seq_len=max_seq_len)
    val_dataset     = ClassificationDataset(dev  , tokenizer, max_seq_len=max_seq_len)
    test_dataset    = ClassificationDataset(test , tokenizer, max_seq_len=max_seq_len)

    # Define the dataloaders
    train_dataloader    = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader      = torch.utils.data.DataLoader(val_dataset  , batch_size=batch_size, shuffle=False)
    test_dataloader     = torch.utils.data.DataLoader(test_dataset , batch_size=batch_size, shuffle=False)

    # Initialize the model and optimizer
    model       = BERTBinaryClassifier(num_classes=2).to(device)
    optimizer   = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005)

    print("-------------------------")
    # Train the model
    for epoch in range(num_epochs):
        t0 = perf_counter()
        # Training loop
        for i,batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)

            loss.backward()
            optimizer.step()

            print(round(float(i) / len(train_dataloader), 3), end='\r')
        
        print('validation...',end='\r')
        # Validation loop
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(logits, axis=1)
                total_correct += torch.sum(predictions == labels)
                total_samples += labels.shape[0]

            accuracy = float(total_correct) / float(total_samples)
        
        t_epoch = perf_counter() - t0
        formatted_time = strftime('%M:%S', gmtime(t_epoch))

        print(f"Epoch {epoch+1}: Validation Accuracy: {accuracy} -- TIME: {formatted_time}")

    # EVAL TEST LOOP
    print('....testing...')
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        y_pred = []
        y_true = []
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, axis=1)

            total_correct += torch.sum(predictions == labels)
            total_samples += labels.shape[0]

            y_pred = y_pred + list(predictions.cpu())
            y_true = y_true + list(labels.cpu())
        accuracy = float(total_correct) / float(total_samples)

    print(classification_report(y_true,y_pred))
    return {'model': model, 'optimizer': optimizer}
