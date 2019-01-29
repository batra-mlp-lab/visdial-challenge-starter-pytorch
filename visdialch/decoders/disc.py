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

        # options are variable length padded sequences, use DynamicRNN
        self.option_rnn = DynamicRNN(self.option_rnn)

    def forward(self, encoder_output, batch):
        """Given `encoder_output` and candidate option sequences, predict a score
        for each option sequence.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """

        options = batch['opt']
        batch_size, num_rounds, num_options, max_sequence_length = options.size()
        options = options.view(batch_size * num_rounds * num_options, max_sequence_length)

        options_length = batch['opt_len']
        options_length = options_length.view(batch_size * num_rounds * num_options)

        # shape: (batch_size * num_rounds * num_options, max_sequence_length, word_embedding_size)
        options_embed = self.word_embed(options)

        # shape: (batch_size * num_rounds * num_options, lstm_hidden_size)
        _, (options_embed, _) = self.option_rnn(options_embed, options_length)

        # repeat encoder output for every option
        # shape: (batch_size, num_rounds, num_options, max_sequence_length)
        encoder_output = encoder_output.unsqueeze(2).repeat(1, 1, num_options, 1)

        # shape now same as `options`, can calculate dot product similarity
        # shape: (batch_size * num_rounds * num_options, lstm_hidden_state)
        encoder_output = encoder_output.view(
            batch_size * num_rounds * num_options, self.config["lstm_hidden_size"]
        )

        # shape: (batch_size * num_rounds * num_options)
        scores = torch.sum(options_embed * encoder_output, 1)
        # shape: (batch_size, num_rounds, num_options)
        scores = scores.view(batch_size, num_rounds, num_options)
        return scores
