# train.py
import os
import pandas as pd
import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, set_seed
from sklearn.metrics import accuracy_score, f1_score
from model import BertForSimilarity
from utils import load_synonym_dict, synonym_replace, FreeLB
import random 
from tqdm import tqdm 
set_seed(42)

class SimilarityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row['query1'],
            row['query2'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs['logits'], dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSimilarity().to(device)

    # 加载数据
    train_df = pd.read_csv('data/train.csv')
    dev_df = pd.read_csv('data/dev.csv')


    print("data loaded")
    # 数据增强：对训练集部分样本进行同义词替换
    print("Loading synonym dictionary...")
    # syn_dict = load_synonym_dict('syn_dict/synonym.txt')
    # augmented_rows = []
    # for _, row in train_df.iterrows():
    #     if random.random() < 0.5:  # 50% 概率增强
    #         new_row = row.copy()
    #         if random.random() < 0.5:
    #             new_row['Query1'] = synonym_replace(row['Query1'], syn_dict)
    #         else:
    #             new_row['Query2'] = synonym_replace(row['Query2'], syn_dict)
    #         augmented_rows.append(new_row)
    # train_df_aug = pd.concat([train_df, pd.DataFrame(augmented_rows)], ignore_index=True)
    train_df_aug = train_df
    print(f"Original train size: {len(train_df)}, Augmented: {len(train_df_aug)}")

    train_dataset = SimilarityDataset(train_df_aug, tokenizer)
    dev_dataset = SimilarityDataset(dev_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    frelb = FreeLB(adv_K=3, adv_lr=1e-2, adv_max_norm=1e-1)

    best_f1 = 0.0
    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            optimizer.zero_grad()
            loss_adv = frelb.attack(model, inputs, labels, nn.CrossEntropyLoss())
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # loss_adv = outputs['loss']
            # loss_adv.backward()
            optimizer.step()
            # total_loss += loss_adv.item()
            total_loss += loss_adv

        avg_loss = total_loss / len(train_loader)
        dev_acc, dev_f1 = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), 'best_model.bin')
            print("✅ Saved best model.")

    print(f"Best Dev F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()