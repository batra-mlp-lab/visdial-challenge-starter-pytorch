import torch
import torch.nn as nn

from utils import DynamicRNN


class DiscriminativeDecoder(nn.Module):
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        # share word embedding
        self.word_embed = encoder.word_embed
        self.option_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size, batch_first=True)
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
        options : torch.LongTensor
            Candidate answer option sequences. (b, num_options, max_len + 1) 
        """
        options = batch['opt']
        options_len = batch['opt_len']
        # word embed options
        options = options.view(options.size(0) * options.size(1), options.size(2), -1)
        options_len = options_len.view(options_len.size(0) * options_len.size(1), -1)
        batch_size, num_options, max_opt_len = options.size()
        options = options.contiguous().view(-1, num_options * max_opt_len)
        options = self.word_embed(options)
        options = options.view(batch_size, num_options, max_opt_len, -1)

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
