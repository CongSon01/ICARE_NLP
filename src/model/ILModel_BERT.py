import torch
import torch.nn as nn
from transformers import BertModel, BertForPreTraining

def get_tokens_id(x):
    tokens_id = x.new_ones(x.shape)
    for i in range(tokens_id.size(0)):
        ids = x[i].tolist()
        first_seq = ids.index(102)
        for j in range(first_seq + 1):
            tokens_id[i, j] = 0
    return tokens_id

class Predictor(torch.nn.Module):
    def __init__(self, num_class, hidden_size):
        super(Predictor, self).__init__()

        self.num_class = num_class

        self.dis = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, self.num_class)
        )

    def forward(self, z):
        return self.dis(z)
    
class ILModel(nn.Module):
    
    def __init__(self, n_tasks, n_class, hidden_size):
        super().__init__()

        self.n_class = n_class
        self.n_tasks = n_tasks
        self.Bert = BertModel.from_pretrained("bert-base-uncased")
        # self.Bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.config = self.Bert.config
        self.vocab_size = self.config.vocab_size

        self.hidden_size = hidden_size

        self.Text_Encoder = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.Tanh()
        )

        self.Distill_Encoder = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.Tanh()
        )

        self.cls_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, n_class)
        )

        # Modified classifier with 3 layers
        # self.cls_classifier = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_class)
        # )

        # Modified classifier with LSTM 3 layers
        # self.lstm_layers = 3
        # self.cls_classifier = nn.LSTM(
        #     input_size=self.hidden_size * 2,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.lstm_layers,
        #     # num_layers=1,  # Single layer LSTM
        #     batch_first=True,
        #     bidirectional=True
        # )
        # self.final_classifier = nn.Linear(self.hidden_size * 2, n_class)

    def forward(self, x, mask):

        tokens_ids = get_tokens_id(x)
        output = self.Bert(input_ids=x,
                           attention_mask=mask,
                           token_type_ids=tokens_ids,
                           return_dict=True)

        sequence_output = output.last_hidden_state
        bert_embedding = sequence_output[:, 0:1, :].squeeze(dim=1)  # convert to 1 dim

        text_features = self.Text_Encoder(bert_embedding)
        distill_features = self.Distill_Encoder(bert_embedding)

        distill_pred = self.cls_classifier(distill_features)

        # Pass through LSTM layers
        # lstm_output, _ = self.lstm(features)
        # lstm_output = lstm_output[:, -1, :] 
        # cls_pred = self.cls_classifier(lstm_output)

        features = torch.cat([text_features, distill_features], dim=1)
        cls_pred = self.cls_classifier(features)

        return text_features, distill_features, \
               cls_pred, distill_pred, bert_embedding
    