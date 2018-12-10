import torch
from torch import nn

from visdialch.utils import DynamicRNN


class LateFusionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(config["vocab_size"],
                                       config["word_embedding_size"],
                                       padding_idx=0)
        self.hist_rnn = nn.LSTM(config["word_embedding_size"],
                                config["lstm_hidden_size"],
                                config["lstm_num_layers"],
                                batch_first=True,
                                dropout=confg["dropout"])
        self.ques_rnn = nn.LSTM(config["word_embedding_size"],
                                config["lstm_hidden_size"],
                                config["lstm_num_layers"],
                                batch_first=True,
                                dropout=config["dropout"])
        self.dropout = nn.Dropout(p=config["dropout"])

        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        # fusion layer
        fusion_size = config["img_feature_size"] + config["lstm_hidden_size"] * 2
        self.fusion = nn.Linear(fusion_size, config["rnn_hidden_size"])

        if args.weight_init == 'xavier':
            nn.init.xavier_uniform_(self.fusion.weight)
        elif args.weight_init == 'kaiming':
            nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, batch):
        img = batch['img_feat']
        ques = batch['ques']
        hist = batch['hist']

        batch_size, num_rounds, _ = ques.size()

        # repeat image feature vectors to be provided for every round
        img = img.view(batch_size, 1, self.config["img_feature_size"])
                 .repeat(1, num_rounds, 1)
                 .view(batch_size * num_rounds,
                       self.config["img_feature_size"])

        # embed questions
        ques = ques.view(-1, ques.size(2))
        ques_embed = self.word_embed(ques)
        ques_embed = self.ques_rnn(ques_embed, batch['ques_len'])

        # embed history
        hist = hist.view(-1, hist.size(2))
        hist_embed = self.word_embed(hist)
        hist_embed = self.hist_rnn(hist_embed, batch['hist_len'])

        fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        fused_vector = self.dropout(fused_vector)

        fused_embedding = torch.tanh(self.fusion(fused_vector))
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)
        return fused_embedding
