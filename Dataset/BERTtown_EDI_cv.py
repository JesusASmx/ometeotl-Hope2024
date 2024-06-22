#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd

train = pd.read_csv("..//Dataset//EDI_train.csv")
val = pd.read_csv("..//Dataset//EDI_val.csv")
train_df = pd.concat([train, val])

print(train_df["category"].value_counts())

test_df = pd.read_csv("..//Dataset//Hope_test/EDI_test.csv")


import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from transformers import (set_seed,
                          BertConfig,
                          BertTokenizer,
                          get_linear_schedule_with_warmup,
                          BertForSequenceClassification)


set_seed(42)

epochs = 5 # Best performance in eval stage.
batch_size = 32
max_length = 128
labels_column = "category"
labels_ids = {'nhs': 0, 'hs': 1}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = 'nlptown/bert-base-multilingual-uncased-sentiment'

token_path = 'DeepESP/gpt2-spanish'
n_labels=2

#######

from tqdm import tqdm
import numpy as np

class HopeDataset(Dataset):
    def __init__(self, df, text, target, labels_ids):

        self.target = target

        self.texts = []

        if self.target:
            self.labels = []

        for x in df.index:
            self.texts.append(df[text][x])
            if self.target:
                self.labels.append(df[target][x])
        
        self.n_examples = len(self.texts)
        return

    def __len__(self):
        return self.n_examples

    def __getitem__(self, item):
        if self.target:
            return {'text':self.texts[item], 'label':self.labels[item]}
        else:
            return {'text':self.texts[item]}
    



################
    


class Gpt2ClassificationCollator(object):
    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder
        return

    def __call__(self, sequences):
        lbels = [sequence['label'] for sequence in sequences]
        labels = [self.labels_encoder[label] for label in lbels]

        texts = [sequence['text'] for sequence in sequences]

        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        inputs.update({'labels':torch.tensor(labels)})
        return inputs

class TestCollator(object):
    def __init__(self, use_tokenizer, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        return

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        return inputs
    

#################
    
def train(dataloader, optimizer_, scheduler_, device_):
    global model

    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.train()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()
        scheduler_.step()
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss

#########

def validation(dataloader, device_):
    global model

    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_epoch_loss


def inference(dataloader, device_):
    global model

    predictions_labels = []

    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

    return predictions_labels



#######################################################################################

model_config = BertConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model.to(device)
print('Model loaded to `%s`'%device)

###################



import numpy as np

def cv(csv, label_col, labels_ids, folds): #category
    final_csvs = []
    lista_csv = []
    for x in labels_ids:
        lab_csv = csv[csv[label_col]==x].copy().sample(frac=1).reset_index(drop=True)
        try:
            lista_csv.append(np.array_split(lab_csv, folds))
        except:
            print("CV detected as 0, we return training concat with validation")
            final_csvs.append([train_df])
            break


    # Ensemble the complete csv's:
    final_folds = []
    for i in range(folds):
        a = pd.concat([x[i].copy() for x in lista_csv]).sample(frac=1).reset_index(drop=True)
        final_folds.append(a)

    for i in range(folds):
        try:
            final_csvs.append(
                [
                    pd.DataFrame(pd.concat([final_folds[j].copy() for j in range(folds) if j != i])).sample(frac=1).reset_index(drop=True), 
                    pd.DataFrame(final_folds[i].copy()).sample(frac=1).reset_index(drop=True)
                ]
            ) #[train_df, val_df]
        except:
            if folds == 1:
                print("CV detected as 1, we return training and validation sets instead.")
                entrena = pd.read_csv("..//Dataset//EDI_train.csv")
                validac = pd.read_csv("..//Dataset//EDI_val.csv")
                final_csvs.append([entrena, validac])
            break

    return final_csvs


################################


optimizer = torch.optim.AdamW(model.parameters(),
                  lr = 5e-5, # default is 5e-5, 2e-5 first one
                  eps = 1e-8
                  )


all_loss = {'train_loss':[], 'val_loss':[]}
all_f1 = {'train_f1':[], 'val_f1':[]}


#############

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

crossval_dfs = cv(train_df, labels_column, labels_ids, 5)

collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                    labels_encoder=labels_ids,
                                    max_sequence_len=max_length)

learning_curve = []
for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch+1}")
    
    trucar = False

    for x in tqdm(crossval_dfs, desc='Fold'):
        if len(crossval_dfs) > 1500:
            trucar = True
            print("To truncate after the first iteration")

        if len(x) > 1:
            train_cv_dataset = HopeDataset(df=x[0],
                                        text="text",
                                        target=labels_column, #"category"
                                        labels_ids=labels_ids)
            train_cv_dataloader = DataLoader(train_cv_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

            valid_cv_dataset = HopeDataset(df=x[1],
                                        text="text",
                                        target=labels_column,
                                        labels_ids=labels_ids) 
            valid_cv_dataloader = DataLoader(valid_cv_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)



        total_steps = len(train_cv_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = 0,
                                                    num_training_steps = total_steps)

        train_labels, train_predict, train_loss = train(train_cv_dataloader, optimizer, scheduler, device)
        train_f1 = f1_score(train_labels, train_predict, average='macro')

        if len(x) > 1:
            valid_labels, valid_predict, val_loss = validation(valid_cv_dataloader, device)
            reporte = classification_report(valid_labels, valid_predict, target_names=list(labels_ids.keys()))
            print(reporte)

            # Print loss and accuracy values to see how training evolves.
            print("  train_loss: %.5f - val_loss: %.5f - train_f1: %.5f"%(train_loss, val_loss, train_f1))
            print()

            all_loss['val_loss'].append(val_loss)
            all_f1['val_f1'].append(f1_score(valid_labels, valid_predict))

        # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_f1['train_f1'].append(train_f1)

        if len(x) > 1:
            learning_curve.append([train_loss, val_loss])
        else:
            learning_curve.append([train_loss, 0])

        if trucar:
            break

import json

with open("lcurve.json", "w") as json_file:
    json.dump(learning_curve, json_file)


def relabel(x):
    if x == 0:
        return 'nhs'
    else:
        return 'hs'

def EDI_submission(valid_predict):
    dip = {"id":test_df.id.values.tolist(), "category":[relabel(x) for x in valid_predict]}
    return pd.DataFrame(dip)


test_ids = test_df.id.values.tolist()
test_texts = test_df["text"].tolist()
nhs_logits = {}
hs_logits = {}

model.eval()
for x in tqdm(range(len(test_ids)), desc='###PREDICTING OVER THE TEST SET###'):
    test_input = tokenizer(text=test_texts[x], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    test_output = model(**test_input)
    logits = test_output.logits
    logits = logits.detach().cpu().numpy()

    nhs_logits[test_ids[x]] = logits[0][0]
    hs_logits[test_ids[x]] = logits[0][1]


print(f"NO-HOPE LOGITS: {len(nhs_logits.keys())}, YES-HOPE LOGITS: {len(hs_logits.keys())}")

sorted_nhs = {k: v for k, v in sorted(nhs_logits.items(), key=lambda item: item[1])}
sorted_hs = {k: v for k, v in sorted(hs_logits.items(), key=lambda item: item[1])}

max_nhs = max(sorted_nhs.values())
min_nhs = min(sorted_nhs.values())

max_hs = max(sorted_hs.values())
min_hs = min(sorted_hs.values())

print(f"Max logit nhs: {max_nhs}, min logit nhs: {min_nhs}")
print(f"Max logit hs: {max_hs}, min logit hs: {min_hs}")

elegir = [max_nhs-min_nhs, max_hs-min_hs] #0 for nhs, 1 for hs
elegi = max(elegir) # The major the gap, the least the uncertainty (??? rly???)
elegido = elegir.index(elegi)

ids_els = {0: sorted_nhs, 1:sorted_hs}
ids_labels = {0: 'nhs', 1:'hs'}

elected = ids_els[elegido]
print(f"Higher logit ratio: {ids_labels[elegido]}")

#Los 200 con mayor probabilidad de pertenecer a la clase:
elegidos = list(elected)[-200:]
no_elegidos = [x for x in test_ids if x not in elegidos]

elected_finals = {k: ids_labels[elegido] for k in elegidos}
elected_finals.update({k: ids_labels[1-elegido] for k in no_elegidos})


#a = pd.DataFrame({"a":list(elected_finals.keys()), "b":list(elected_finals.values())})
#print(a["b"].value_counts())
#print({k: elected_finals[k] for k in list(elected_finals)[:5]})
#print("\n",{k: elected_finals[k] for k in list(elected_finals)[-5:]})

final_preds = {}
for x in range(len(test_ids)):
    ident = test_ids[x]
    final_preds[ident] = elected_finals[ident]

dip = {"id":test_df.id.values.tolist(), "category":[final_preds[x] for x in final_preds]}
resuls = pd.DataFrame(dip)

print(resuls["category"].value_counts())

resuls.to_csv("predictions.csv")