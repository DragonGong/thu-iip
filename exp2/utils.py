# utils.py
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ----------------------------
# 同义词词林加载（哈工大扩展版）
# 下载地址：https://github.com/liuhuanyong/ChineseDictionary
# 放在 syn_dict/synonym.txt
# 格式示例：
# Aa01A01=人 人类 人物 人士 个体...
# ----------------------------
def load_synonym_dict(path='syn_dict/synonym.txt'):
    syn_dict = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                _, words = line.strip().split('=', 1)
                word_list = words.split(' ')
                for w in word_list:
                    syn_dict[w].extend([x for x in word_list if x != w])
    return dict(syn_dict)

# 同义词替换（仅对 Query1 或 Query2 中的一个做增强）
def synonym_replace(sentence, syn_dict, replace_prob=0.3):
    words = list(sentence)
    new_words = words.copy()
    for i, w in enumerate(words):
        if w in syn_dict and random.random() < replace_prob:
            candidates = syn_dict[w]
            if candidates:
                new_words[i] = random.choice(candidates)
    return ''.join(new_words)

# ----------------------------
# FreeLB 对抗训练类
# ----------------------------
class FreeLB:
    def __init__(self, adv_K=3, adv_lr=1e-2, adv_init_mag=2e-2, adv_max_norm=1e-1):
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_init_mag = adv_init_mag
        self.adv_max_norm = adv_max_norm

    def attack(self, model, inputs, labels, loss_fn):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        embeds_init = model.bert.embeddings.word_embeddings(input_ids)
        batch_size, seq_len, hidden_size = embeds_init.size()

        delta = torch.zeros(batch_size, seq_len, hidden_size, device=embeds_init.device)
        delta.uniform_(-1, 1)
        mask = (input_ids != 0).unsqueeze(2).float()
        dims = mask.sum(dim=1, keepdim=True)
        mag = self.adv_init_mag / torch.sqrt(dims)
        delta = delta * mag * mask
        delta.requires_grad_()

        total_loss = 0.0
        for k in range(self.adv_K):
            outputs = model(inputs_embeds=embeds_init + delta, attention_mask=attention_mask)
            logits = outputs['logits']
            loss = loss_fn(logits, labels)
            loss = loss / self.adv_K
            total_loss += loss.item()

            loss.backward(retain_graph=(k < self.adv_K - 1))

            delta_grad = delta.grad.clone().detach()
            grad_norm = torch.norm(delta_grad.view(batch_size, -1), dim=1).view(-1, 1, 1)
            delta = delta + self.adv_lr * delta_grad / (grad_norm + 1e-8)
            delta = delta.detach()

            delta_norm = torch.norm(delta.view(batch_size, -1), dim=1).view(-1, 1, 1)
            delta = delta * torch.min(self.adv_max_norm / delta_norm, torch.ones_like(delta_norm))
            delta = delta * mask

            if k < self.adv_K - 1:
                delta.requires_grad_()
                # 注意：不清空模型梯度！让梯度累积

        return total_loss