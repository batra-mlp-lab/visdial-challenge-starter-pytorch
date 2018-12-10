from .disc import DiscriminativeDecoder


def Decoder(model_config, encoder):
    name_dec_map = {
       'disc': DiscriminativeDecoder
    }
    return name_dec_map[model_config["decoder"](model_config, encoder)
