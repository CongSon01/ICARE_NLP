import torch
import torch.nn as nn
from transformers import BertModel

class BaseModel(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.n_class = n_class
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768, n_class)
        )

    def forward(self, x):
        x, _ = self.bert(x)
        x = torch.mean(x, 1)
        logits = self.classifier(x)
        return logits