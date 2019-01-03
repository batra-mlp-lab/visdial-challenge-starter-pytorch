import torch
from torch import nn

from visdialch.utils import DynamicRNN


class LateFusionEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(
            len(vocabulary), config["word_embedding_size"], padding_idx=vocabulary.PAD_INDEX
        )
        self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )
        self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )
        self.dropout = nn.Dropout(p=config["dropout"])

        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        # fusion layer
        fusion_size = config["img_feature_size"] + config["lstm_hidden_size"] * 2
        self.fusion = nn.Linear(fusion_size, config["lstm_hidden_size"])

        nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, batch):
        # shape: (batch_size, img_feature_size) - CNN fc7 features
        # shape: (batch_size, num_proposals, img_feature_size) - RCNN bottom-up features
        img = batch["img_feat"]
        # shape: (batch_size, 10, max_sequence_length)
        ques = batch["ques"]
        # shape: (batch_size, 10, max_sequence_length * 2 * 10), concatenated qa * 10 rounds
        hist = batch["hist"]
        # num_rounds = 10, even for test split (padded dialog rounds at the end)
        batch_size, num_rounds, max_sequence_length = ques.size()

        # average bottom-up features of all proposals, to form fc7-like features
        if img.dim() == 3:
            img = torch.mean(img, dim=1)  # shape: (batch_size, img_feature_size)

        # repeat image feature vectors to be provided for every round
        img = img.view(batch_size, 1, self.config["img_feature_size"])
        img = img.repeat(1, num_rounds, 1)
        img = img.view(batch_size * num_rounds, self.config["img_feature_size"])

        # embed questions
        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        ques_embed = self.word_embed(ques)
        ques_embed = self.ques_rnn(ques_embed, batch["ques_len"])

        # embed history
        hist = hist.view(batch_size * num_rounds, max_sequence_length * 20)
        hist_embed = self.word_embed(hist)
        hist_embed = self.hist_rnn(hist_embed, batch["hist_len"])

        fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        fused_vector = self.dropout(fused_vector)

        fused_embedding = torch.tanh(self.fusion(fused_vector))
        # shape: (batch_size, num_rounds, lstm_hidden_size)
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)
        return fused_embedding
