import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class LateFusionEncoder(nn.Module):

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Encoder specific arguments')
        parser.add_argument('-img_feature_size', default=4096,
                                help='Channel size of image feature')
        parser.add_argument('-img_spatial_size', default=14,
                                help='JSON file with image paths and vocab')
        parser.add_argument('-embed_size', default=300,
                                help='Size of the input word embedding')
        parser.add_argument('-rnn_hidden_size', default=512,
                                help='Size of the multimodal embedding')
        parser.add_argument('-num_layers', default=2,
                                help='Number of layers in LSTM')
        parser.add_argument('-max_history_len', default=60,
                                help='Size of the multimodal embedding')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.word_embed = nn.Embedding(opt.vocab_size, opt.embed_size, padding_idx=0)
        self.hist_rnn = nn.LSTM(opt.embed_size, opt.rnn_hidden_size, opt.num_layers,
                                batch_first=True, dropout=opt.dropout)
        self.ques_rnn = nn.LSTM(opt.embed_size, opt.rnn_hidden_size, opt.num_layers,
                                batch_first=True, dropout=opt.dropout)

        # fusion layer
        fusion_size = opt.img_feature_size + opt.rnn_hidden_size * 2
        self.fusion = nn.Linear(fusion_size, opt.rnn_hidden_size)

    def forward(self, img, ques, hist):
        # repeat image feature vectors to be provided for every round
        img = img.view(-1, 1, self.opt.img_feature_size)
        img = img.repeat(1, self.opt.max_ques_count, 1)
        img = img.view(-1, self.opt.img_feature_size)

        # embed questions
        ques = ques.view(-1, ques.size(2))
        ques_embed, _ = self.ques_rnn(self.word_embed(ques), None)
        # pick the last time step (final question encoding)
        ques_embed = ques_embed[:, -1, :]

        # embed history
        hist = hist.view(-1, hist.size(2))
        hist_embed, _ = self.hist_rnn(self.word_embed(hist), None)
        hist_embed = hist_embed[:, -1, :]

        fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        if self.opt.dropout > 0:
            fused_vector = F.dropout(fused_vector, self.opt.dropout,
                                     training=self.opt.training)

        fused_embedding = F.tanh(self.fusion(fused_vector))
        return fused_embedding
