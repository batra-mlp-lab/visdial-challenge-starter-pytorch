import argparse
from datetime import datetime
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.model import EncoderDecoderModel
from visdialch.utils import process_ranks, scores_to_ranks, get_gt_ranks
from visdialch.utils.checkpointing import load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml", default="configs/lf_disc_vgg16_fc7_bs32.yml",
    help="Path to a config file listing reader, model and optimization parameters."
)
parser.add_argument(
    "--eval-json", default="data/visdial_1.0_val.json",
    help="Path to VisDial v1.0 val/test data, whichever to evaluate on."
)

parser.add_argument_group("Evaluation related arguments")
parser.add_argument(
    "--load-pthpath", default="checkpoints/checkpoint_xx.pth",
    help="Path to .pth file of pretrained checkpoint."
)
parser.add_argument(
    "--use-gt", action="store_true",
    help="Whether to use ground truth for retrieving ranks"
)

parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument(
    "--gpu-ids", nargs="+", type=int, default=-1,
    help="List of ids of GPUs to use."
)
parser.add_argument(
    "--cpu-workers", type=int, default=4,
    help="Number of CPU workers for reading data."
)
parser.add_argument(
    "--overfit", action="store_true",
    help="Overfit model on 5 examples, meant for debugging."
)

parser.add_argument_group("Submission related arguments")
parser.add_argument(
    "--save-ranks-path", default="logs/ranks.json",
    help="Path (json) to save ranks, works only when use_gt=false."
)

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ================================================================================================
#   INPUT ARGUMENTS AND CONFIG
# ================================================================================================

args = parser.parse_args()

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))
if type(args.gpu_ids) == int: args.gpu_ids = [args.gpu_ids]
device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

# print config and args
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))


# ================================================================================================
#   SETUP DATASET, DATALOADER, MODEL
# ================================================================================================

val_dataset = VisDialDataset(args.eval_json, config["dataset"], args.overfit)
val_dataloader = DataLoader(
    val_dataset, batch_size=config["solver"]["batch_size"], num_workers=args.cpu_workers
)

if args.use_gt and "test" in dataset.split:
    print("Warning: No ground truth for test split, changing use_gt to False.")
    args.use_gt = False

# pass vocabulary to construct nn.Embedding
encoder = Encoder(config["model"], val_dataset.vocabulary)
decoder = Decoder(config["model"], val_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# share word embedding between encoder and decoder
decoder.word_embed = encoder.word_embed

# wrap encoder and decoder in a model
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

model_state_dict, _ = load_checkpoint(args.load_pthpath)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(model_state_dict)
else:
    model.load_state_dict(model_state_dict)
print("Loaded model from {}".format(args.load_pthpath))

# ================================================================================================
#   EVALUATION LOOP
# ================================================================================================

encoder.eval()
decoder.eval()

if args.use_gt:
    # --------------------------------------------------------------------------------------------
    # calculate automatic metrics and finish
    # --------------------------------------------------------------------------------------------
    all_ranks = []
    for i, batch in enumerate(tqdm(val_dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            output = model(batch)
        ranks = scores_to_ranks(output)
        gt_ranks = get_gt_ranks(ranks, batch["ans_ind"])
        all_ranks.append(gt_ranks)
    all_ranks = torch.cat(all_ranks, 0)
    process_ranks(all_ranks)
else:
    # --------------------------------------------------------------------------------------------
    # prepare json for submission
    # --------------------------------------------------------------------------------------------
    ranks_json = []
    for i, batch in enumerate(tqdm(val_dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
                output = model(batch)
        ranks = scores_to_ranks(output)
        ranks = ranks.view(-1, 10, 100)

        for i in range(len(batch["img_ids"])):
            # cast into types explicitly to ensure no errors in schema
            # round ids are 1-10, not 0-9
            if "test" in val_dataset.split:
                ranks_json.append({
                    "image_id": batch["img_ids"][i].item(),
                    "round_id": int(batch["num_rounds"][i].item()),
                    "ranks": [rank.item() for rank in ranks[i][batch["num_rounds"][i] - 1]]
                })
            else:
                for j in range(batch["num_rounds"][i]):
                    ranks_json.append({
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(j + 1),
                        "ranks": [rank.item() for rank in ranks[i][j]]
                    })

    print("Writing ranks to {}".format(args.save_ranks_path))
    os.makedirs(os.path.dirname(args.save_ranks_path), exist_ok=True)
    json.dump(ranks_json, open(args.save_ranks_path, "w"))
