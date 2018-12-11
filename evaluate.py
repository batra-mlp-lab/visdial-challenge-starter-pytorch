import argparse
from datetime import datetime
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from visdialch.dataloader import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.utils import process_ranks, scores_to_ranks, get_gt_ranks


parser = argparse.ArgumentParser()
parser.add_argument("--config-yml", default="configs/lf_disc_vgg16_fc7_bs20.yml",
                        help="Path to a config file listing reader, model and "
                             "optimization parameters.")

parser.add_argument_group("Evaluation related arguments")
parser.add_argument("--load-path", default="checkpoints/model.pth",
                        help="Path to load pretrained checkpoint from.")
parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Split to evaluate on")
parser.add_argument("--use-gt", action="store_true",
                        help="Whether to use ground truth for retrieving ranks")

parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument("--gpu-ids", nargs="+", type=int, default=-1,
                        help="List of ids of GPUs to use.")
parser.add_argument("--cpu-workers", type=int, default=4,
                        help="Number of CPU workers for reading data.")
parser.add_argument("--overfit", action="store_true",
                        help="Overfit model on 5 examples, meant for debugging.")

parser.add_argument_group("Submission related arguments")
parser.add_argument("--save-ranks-path", default="logs/ranks.json",
                        help="Path (json) to save ranks, works only when use_gt=false.")

# ----------------------------------------------------------------------------
# input arguments and config
# ----------------------------------------------------------------------------

args = parser.parse_args()
if args.use_gt and args.split == "test":
    print("Warning: No ground truth for test split, changing use_gt to False.")
    args.use_gt = False

# keys: {"dataset", "model", "training", "evaluation"}
config = yaml.load(open(args.config_yml))

# print config and args
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# set CPU/GPU device for execution
if args.gpu_ids[0] >= 0:
    device = torch.device("cuda", args.gpu_ids[0])
else:
    device = torch.device("cpu")


# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = VisDialDataset(config["dataset"], [args.split], overfit=args.overfit)
dataloader = DataLoader(dataset,
                        batch_size=config["evaluation"]["batch_size"],
                        shuffle=False,
                        collate_fn=dataset.collate_fn,
                        num_workers=args.cpu_workers)

# transfer some attributes from dataset to model args
for key in {"vocab_size", "max_ques_count"}:
    config["model"][key] = getattr(dataset, key)


# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

components = torch.load(args.load_path)

encoder = Encoder(config["model"])
decoder = Decoder(config["model"])
encoder.load_state_dict(components["encoder"])
decoder.load_state_dict(components["decoder"])

# transfer to assigned device for execution
encoder = encoder.to(device)
decoder = decoder.to(device)

# wrap around DataParallel to support multi-GPU execution
encoder = nn.DataParallel(encoder, args.gpu_ids)
decoder = nn.DataParallel(decoder, args.gpu_ids)

print("Loaded model from {}".format(args.load_path))
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

print("Evaluation start time: {}".format(
    datetime.strftime(datetime.now(), "%d-%b-%Y-%H:%M:%S")))
encoder.eval()
decoder.eval()

if args.use_gt:
    # ------------------------------------------------------------------------
    # calculate automatic metrics and finish
    # ------------------------------------------------------------------------
    all_ranks = []
    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = batch[key].to(device)

        with torch.no_grad():
            enc_out = encoder(batch)
            dec_out = decoder(enc_out, batch)
        ranks = scores_to_ranks(dec_out)
        gt_ranks = get_gt_ranks(ranks, batch["ans_ind"])
        all_ranks.append(gt_ranks)
    all_ranks = torch.cat(all_ranks, 0)
    process_ranks(all_ranks)
else:
    # ------------------------------------------------------------------------
    # prepare json for submission
    # ------------------------------------------------------------------------
    ranks_json = []
    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = batch[key].to(device)

        with torch.no_grad():
            enc_out = encoder(batch)
            dec_out = decoder(enc_out, batch)
        ranks = scores_to_ranks(dec_out)
        ranks = ranks.view(-1, 10, 100)

        for i in range(len(batch["img_ids"])):
            # cast into types explicitly to ensure no errors in schema
            if args.split == "test":
                ranks_json.append({
                    "image_id": batch["img_ids"][i].item(),
                    "round_id": int(batch["num_rounds"][i]),
                    "ranks": list(ranks[i][batch["num_rounds"][i] - 1])
                })
            else:
                for j in range(batch["num_rounds"][i]):
                    ranks_json.append({
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(j + 1),
                        "ranks": list(ranks[i][j])
                    })

    print("Writing ranks to {}".format(args.save_ranks_path))
    os.makedirs(os.path.dirname(args.save_ranks_path), exist_ok=True)
    json.dump(ranks_json, open(args.save_ranks_path, "w"))
