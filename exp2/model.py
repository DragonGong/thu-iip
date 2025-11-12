# model.py
import torch
import torch.nn as nn
from transformers import BertModel

class BertForSimilarity(nn.Module):
    def __init__(self, model_name='bert-base-chinese', num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None):
        if inputs_embeds is not None:
            outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}
    