import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class DynamicRNN(nn.Module):
    def __init__(self, rnn_model):
        super().__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.

        Arguments
        ---------
        seq_input : torch.autograd.Variable
            Input sequence tensor (padded) for RNN model. (b, max_seq_len, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.autograd.Variable
            Initial (hidden, cell) states of RNN model.

        Returns
        -------
            A single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(
            sorted_seq_input, lengths=sorted_len, batch_first=True)

        if initial_state is not None:
            hx = initialState
            sorted_hx = [x.index_select(1, fwd_order) for x in hx]
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            hx = None
        _, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)

        rnn_output = h_n[-1].index_select(dim=0, index=bwd_order)
        return rnn_output

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(lens.contiguous().view(-1), 0, descending=True)
        _, bwd_order = torch.sort(fwd_order)
        if isinstance(sorted_len, Variable):
            sorted_len = sorted_len.data
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order
