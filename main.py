#!/usr/bin/env python
# coding: utf-8

import json
import logging
import re

import numpy as np
import pandas as pd
import torch
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertForTokenClassification
from sklearn.model_selection import train_test_split
from torch.optim import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

model_name = input("Enter model name: ")

# HYPER
optim = Adadelta # Adadelta, Adagrad, Adam, Adamax, AdamW,ASGD,LBFGS, NAdam  Rprop, SGD, RMSprop, RAdam, SparseAdam
learn_rate = 1e-4 # 1e-2, 1e-3, 1e-4, 1e-5, 1e-6
batch_size = 8 # 2, 4, 8, 16, 32, 64, 128

# CONST
data_file_address = "data.json"
validation_ratio = 0.1
epochs = 10

# Reading data
df_data = pd.read_json(data_file_address, lines=True)



# Removing New Line characters
for i in range(len(df_data)):
    df_data["content"][i] = df_data["content"][i].replace("\n", " ")




# JSON formatting functions
def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines=[]
        with open(dataturks_JSON_FilePath, 'r', errors="ignore") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content'].replace("\n", " ")
            entities = []
            data_annotations = data['annotation']
            if data_annotations is not None:
                for annotation in data_annotations:
                    #only a single point in text annotation.
                    point = annotation['points'][0]
                    labels = annotation['label']
                    # handle both list of labels or a single label.
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        point_start = point['start']
                        point_end = point['end']
                        point_text = point['text']

                        lstrip_diff = len(point_text) - len(point_text.lstrip())
                        rstrip_diff = len(point_text) - len(point_text.rstrip())
                        if lstrip_diff != 0:
                            point_start = point_start + lstrip_diff
                        if rstrip_diff != 0:
                            point_end = point_end - rstrip_diff
                        entities.append((point_start, point_end + 1 , label))
            training_data.append((text, {"entities" : entities}))
        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None

def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data


# In[8]:


data = trim_entity_spans(convert_dataturks_to_spacy(data_file_address))


cleanedDF = pd.DataFrame(columns=["setences_cleaned"])
sum1 = 0
for i in range(len(data)):
    start = 0
    emptyList = ["Empty"] * len(data[i][0].split())
    numberOfWords = 0
    lenOfString = len(data[i][0])
    strData = data[i][0]
    strDictData = data[i][1]
    lastIndexOfSpace = strData.rfind(' ')
    for i in range(lenOfString):
        if (strData[i]==" " and strData[i+1]!=" "):
            for k,v in strDictData.items():
                for j in range(len(v)):
                    entList = v[len(v)-j-1]
                    if (start>=int(entList[0]) and i<=int(entList[1])):
                        emptyList[numberOfWords] = entList[2]
                        break
                    else:
                        continue
            start = i + 1
            numberOfWords += 1
        if (i == lastIndexOfSpace):
            for j in range(len(v)):
                    entList = v[len(v)-j-1]
                    if (lastIndexOfSpace>=int(entList[0]) and lenOfString<=int(entList[1])):
                        emptyList[numberOfWords] = entList[2]
                        numberOfWords += 1
    cleanedDF = cleanedDF.append(pd.Series([emptyList],  index=cleanedDF.columns ), ignore_index=True )
    sum1 = sum1 + numberOfWords


# In[12]:


MAX_LEN = 300



# In[13]:


device = torch.device("cpu")


# In[14]:


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(df_data["content"])


# In[15]:


tokenized_texts = tokenizer.texts_to_sequences(df_data["content"])
# tokenized_texts = [tokenizer.tokenize(sent) for sent in df_data["content"]]


# In[16]:


input_ids = pad_sequences(tokenized_texts,
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


# In[17]:


tags_vals = ["UNKNOWN", "Name", "Degree","Skills","College Name","Email Address","Designation","Companies worked at","Empty","Graduation Year","Years of Experience","Location"]
tag2idx = {t: i for i, t in enumerate(tags_vals)}


# In[18]:


labels = cleanedDF["setences_cleaned"].tolist()


# In[19]:



# In[20]:


tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["Empty"], padding="post",
                     dtype="long", truncating="post")


# In[21]:


attention_masks = [[float(i>0) for i in ii] for ii in input_ids]


# In[22]:


tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=validation_ratio)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=validation_ratio)


# In[23]:


tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)


# In[24]:


train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)


# In[25]:


model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))


# In[26]:


# In[27]:


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = optim(optimizer_grouped_parameters, lr=learn_rate)



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[29]:



max_grad_norm = 1.0
graph_data = []
for _ in range(epochs):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids.long(), token_type_ids=None,
                     attention_mask=b_input_mask.long(), labels=b_labels.long())

        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
        print("Evaluating batch: {}/{}".format(step, len(train_dataloader)))
    # print train loss per epoch
    train_loss= tr_loss/nb_tr_steps
    print("Epoch Train loss: {}".format(train_loss))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids.long(), token_type_ids=None,
                                  attention_mask=b_input_mask.long(), labels=b_labels.long())
            logits = model(b_input_ids.long(), token_type_ids=None,
                           attention_mask=b_input_mask.long())
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    validation_acuracy = eval_accuracy / nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(validation_acuracy))
    graph_data.append({
        'epoch': _,
        'val_loss': eval_loss,
        'val_acc': validation_acuracy,
        'train_loss': train_loss
    })
    print('---------------------------------------------------------------')


# In[ ]:
torch.save(model, f'models\{model_name}.torch')
import csv
with open(f'models\{model_name}.csv', 'w') as file:
    writer = csv.DictWriter(file, graph_data[0].keys())
    writer.writeheader()
    writer.writerows(graph_data)