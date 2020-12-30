#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien
"""

from transformers import DistilBertModel, DistilBertConfig,DistilBertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from bpemb import BPEmb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


num_warmup_steps=0
batch_size=32
device = torch.device('cuda')
trainSetlength=None
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',do_lower_case=True)
bpemb_en = BPEmb(lang="en",add_pad_emb=True)

def bert_text_to_ids(sentence):
  return torch.tensor(tokenizer.encode(sentence, add_special_tokens=True))

def prepare_textsBert(texts, labels,maxlen):
  X = torch.LongTensor(len(texts), maxlen).fill_(tokenizer.pad_token_id)
  for i, text in enumerate(texts):
    indexed_tokens = bert_text_to_ids(text)
    length = min([maxlen, len(indexed_tokens)])
    X[i,:length] = indexed_tokens[:length]
  
  Y = torch.tensor(labels).long()
  return X.to(device), Y.to(device)

def prepare_textsBpemb(texts,labels):
  X = torch.LongTensor(len(texts), maxlen).fill_(bpemb_en.vocab_size) #padding is last
  for i, text in enumerate(texts):
    indexed_tokens = torch.tensor(bpemb_en.encode_ids(text))
    length = min([maxlen, len(indexed_tokens)])
    X[i,:length] = indexed_tokens[:length]
  Y = torch.tensor(labels).long()
  return X.to(device), Y.to(device)

def getLoader(X_train,Y_train,X_valid,Y_valid,X_test,Y_test):
    train_set = TensorDataset(X_train, Y_train)
    valid_set = TensorDataset(X_valid, Y_valid)
    test_set = TensorDataset(X_test, Y_test)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    global trainSetlength
    trainSetlength=len(train_set)
    return train_loader,valid_loader,test_loader


def perf(model, loader,avg='micro'):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = num = correct = 0
    Y_PREDS=np.asarray([])
    Yall=np.asarray([])
    for x, y in loader:
      with torch.no_grad():
        Yall=np.concatenate((Yall,y.cpu().numpy()))
        y_scores = model(x)
        loss = criterion(y_scores, y)
        y_pred = torch.max(y_scores, 1)[1]
        Y_PREDS=np.concatenate((Y_PREDS,y_pred.cpu().numpy()))
        correct += torch.sum(y_pred == y).item()
        total_loss += loss.item()
        num += len(y)
    return total_loss / num, correct / num, f1_score(Yall,Y_PREDS,average=avg)


def fit(model, epochs,train_loader,valid_loader, lr=1e-3,class_weights=None):
    nbBatch=trainSetlength/batch_size
    num_training_steps = nbBatch * epochs
    criterion = nn.CrossEntropyLoss(weight=class_weights) # voir quel loss utilisé pour du multilabel
    #optimizer = optim.Adam(model.parameters(), lr=lr) #BertAdam(model.parameters(), lr=lr, schedule='warmup_linear', warmup=warmup_proportion, t_total=num_total_steps)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, 
    num_training_steps=num_training_steps
  )
    for epoch in range(epochs):
        model.train()
        total_loss = num = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_scores = model(x)
            loss = criterion(y_scores, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num += len(y)
        print(epoch, total_loss / num, *perf(model, valid_loader))

def predict(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    Y_PREDS=np.asarray([])
    for x, y in loader:
      with torch.no_grad():
        y_scores = model(x)
        loss = criterion(y_scores, y)
        y_pred = torch.max(y_scores, 1)[1]
        total_loss += loss.item()
        Y_PREDS=np.concatenate((Y_PREDS,y_pred.cpu().numpy()))
    return Y_PREDS


class BertClassifierSequence(nn.Module):
  def __init__(self,pretrainString='distilbert-base-uncased-finetuned-sst-2-english'):
    super().__init__()
    configuration = DistilBertConfig(dropout=0.25,num_labels=7)
    self.bert = DistilBertModel(configuration).from_pretrained(pretrainString)
    self.pre_classifier = nn.Linear(configuration.dim, configuration.dim)
    self.classifier = nn.Linear(configuration.dim, configuration.num_labels)
    self.dropout = nn.Dropout(configuration.seq_classif_dropout)
    self.to(device)

  def forward(self, x):
    distilbert_output=self.bert(x, attention_mask = (x != tokenizer.pad_token_id))
    hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
    pooled_output = hidden_state[:, 0]  # (bs, dim)
    pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
    pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
    pooled_output = self.dropout(pooled_output)  # (bs, dim)
    logits = self.classifier(pooled_output)  # (bs, num_labels)
    return F.softmax(logits,dim=1)

class
