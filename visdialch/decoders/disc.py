import torch
from torch import nn

from visdialch.utils import DynamicRNN


class DiscriminativeDecoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(len(vocabulary),
                                       config["word_embedding_size"],
                                       padding_idx=vocabulary.PAD_INDEX)
        self.option_rnn = nn.LSTM(config["word_embedding_size"],
                                  config["lstm_hidden_size"],
                                  batch_first=True)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # options are variable length padded sequences, use DynamicRNN
        self.option_rnn = DynamicRNN(self.option_rnn)

    def forward(self, enc_out, batch):
        """Given encoder output `enc_out` and candidate output option sequences,
        predict a score for each output sequence.

        Arguments
        ---------
        enc_out : torch.autograd.Variable
            Output from the encoder through its forward pass. (b, rnn_hidden_size)
        """
        options = batch['opt']
        options_len = batch['opt_len']

        # word embed options
        batch_size, num_rounds, num_options, max_opt_len = options.size()
        options = options.view(batch_size * num_rounds, num_options * max_opt_len)
        options_len = options_len.view(batch_size * num_rounds, num_options)

        options = self.word_embed(options)
        options = options.view(batch_size * num_rounds, num_options, max_opt_len, -1)
        enc_out = enc_out.view(batch_size * num_rounds, -1)

        # score each option
        scores = []
        for opt_id in range(num_options):
            opt = options[:, opt_id, :, :]
            opt_len = options_len[:, opt_id]
            opt_embed = self.option_rnn(opt, opt_len)
            scores.append(torch.sum(opt_embed * enc_out, 1))

        # return scores
        scores = torch.stack(scores, 1)
        log_probs = self.log_softmax(scores)
        return log_probs
