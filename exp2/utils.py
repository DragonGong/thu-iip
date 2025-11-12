# utils.py
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

try:
    import OpenHowNet
    hownet_dict = OpenHowNet.HowNetDict()
    HOWNET_AVAILABLE = True
except ImportError:
    print("Warning: OpenHowNet not installed. Synonym replacement will be disabled.")
    HOWNET_AVAILABLE = False

def get_synonyms(word, topk=5):
    """从 HowNet 获取同义词（近义词）"""
    if not HOWNET_AVAILABLE:
        return []
    try:
        # 获取该词的所有 sense（义项）
        senses = hownet_dict.get_sense(word)
        synonyms = set()
        for sense in senses:
            # 获取与当前义项语义相似的其他词
            similar_words = hownet_dict.get_related_words(sense["word"], relation="antonym", merge=True)
            # 注意：HowNet 的 "similar" 关系不稳定，更推荐用 "same_sememe" 或直接查同 sememe 的词
            # 更稳妥方式：查找具有相同核心语义（sememe）的词
            sememes = sense["sememes"]
            for w in hownet_dict.get_all_words():
                if w == word:
                    continue
                wsenses = hownet_dict.get_sense(w)
                for ws in wsenses:
                    if set(sememes).issubset(set(ws["sememes"])):
                        synonyms.add(w)
                        if len(synonyms) >= topk:
                            break
                if len(synonyms) >= topk:
                    break
        return list(synonyms)[:topk]
    except Exception as e:
        # 某些词可能不在 HowNet 中，或解析失败
        return []

def synonym_replace(sentence, replace_prob=0.3, topk=5):
    """
    对句子中的每个字/词（按字处理，也可改为分词）尝试同义替换。
    注意：这里按「字」处理以兼容原始代码（Query1/Query2 是字符串）。
    若需更高精度，建议先分词（如 jieba），但会增加依赖。
    """
    if not HOWNET_AVAILABLE:
        return sentence  # 无法替换，原样返回

    words = list(sentence)  # 按字切分（简单处理）
    new_words = words.copy()
    for i, char in enumerate(words):
        if random.random() < replace_prob:
            syns = get_synonyms(char, topk=topk)
            if syns:
                new_words[i] = random.choice(syns)
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