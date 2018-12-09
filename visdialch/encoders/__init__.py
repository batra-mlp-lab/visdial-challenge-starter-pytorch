from .lf import LateFusionEncoder


def Encoder(model_args):
    name_enc_map = {
        'lf-ques-im-hist': LateFusionEncoder
    }
    return name_enc_map[model_args.encoder](model_args)

