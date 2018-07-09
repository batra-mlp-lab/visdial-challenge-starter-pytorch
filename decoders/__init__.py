from .disc import DiscriminativeDecoder


def Decoder(model_args, encoder):
    name_dec_map = {
       'disc': DiscriminativeDecoder
    }
    return name_dec_map[model_args.decoder](model_args, encoder)

