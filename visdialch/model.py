from torch import nn


class EncoderDecoderModel(nn.Module):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.

    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        encoder_output = self.encoder(batch)
        decoder_output = self.decoder(encoder_output, batch)
        return decoder_output
