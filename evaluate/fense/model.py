import torch
import numpy as np
from torch import nn, optim, threshold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel


class BERTFlatClassifier(nn.Module):
    def __init__(self, model_type, num_classes=5) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_classes)
    
    def forward(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        logits = self.clf(x)
        return logits

