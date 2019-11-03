from visdialch.encoders.lf import LateFusionEncoder
from visdialch.encoders.rva import RvAEncoder


def Encoder(model_config, *args):
    name_enc_map = {
    	"lf": LateFusionEncoder,
    	"rva": RvAEncoder,
    }
    return name_enc_map[model_config["encoder"]](model_config, *args)
