import torch
import torch.nn as nn
from transformers import XLNetModel

def get_tokens_id(x):
    tokens_id = x.new_ones(x.shape)
    for i in range(tokens_id.size(0)):
        ids = x[i].tolist()
        first_seq = ids.index(102)  # Assuming 102 is the end-of-sentence token
        for j in range(first_seq + 1):
            tokens_id[i, j] = 0
    return tokens_id

class Predictor(nn.Module):
    def __init__(self, num_class, hidden_size):
        super(Predictor, self).__init__()

        self.num_class = num_class

        self.dis = nn.Sequential(
            nn.Linear(hidden_size, self.num_class)
        )

    def forward(self, z):
        return self.dis(z)

class ILModel(nn.Module):
    def __init__(self, n_tasks, n_class, hidden_size):
        super().__init__()

        self.n_class = n_class
        self.n_tasks = n_tasks

        # Load XLNet model
        self.Bert = XLNetModel.from_pretrained("xlnet-base-cased")
        self.config = self.Bert.config
        self.vocab_size = self.config.vocab_size

        self.hidden_size = hidden_size

        # Encoders
        self.Text_Encoder = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.Tanh()
        )

        self.Distill_Encoder = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.Tanh()
        )

        # Classifier layer
        self.cls_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, n_class)
        )

    def forward(self, x, mask):
        # XLNet does not use token_type_ids
        output = self.Bert(input_ids=x, attention_mask=mask, return_dict=True)

        sequence_output = output.last_hidden_state
        xlnet_embedding = sequence_output[:, 0, :]  # Extract [CLS] token embedding

        text_features = self.Text_Encoder(xlnet_embedding)
        distill_features = self.Distill_Encoder(xlnet_embedding)

        distill_pred = self.cls_classifier(distill_features)

        features = torch.cat([text_features, distill_features], dim=1)
        cls_pred = self.cls_classifier(features)

        return text_features, distill_features, \
               cls_pred, distill_pred, xlnet_embedding
