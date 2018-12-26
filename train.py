import argparse
from datetime import datetime
import itertools
import os
import shutil

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.utils import process_ranks, scores_to_ranks, get_gt_ranks


parser = argparse.ArgumentParser()
parser.add_argument("--config-yml", default="configs/lf_disc_vgg16_fc7_bs20.yml",
                        help="Path to a config file listing reader, model and "
                             "optimization parameters.")
parser.add_argument("--train-json", default="data/visdial_1.0_train.json",
                        help="Path to VisDial v1.0 training data.")
parser.add_argument("--val-json", default="data/visdial_1.0_val.json",
                        help="Path to VisDial v1.0 training data.")

parser.add_argument_group("Arguments independent of experiment reproducibility")

parser.add_argument("--gpu-ids", nargs="+", type=int, default=-1,
                        help="List of ids of GPUs to use.")
parser.add_argument("--cpu-workers", type=int, default=4,
                        help="Number of CPU workers for reading data.")
parser.add_argument("--overfit", action="store_true",
                        help="Overfit model on 5 examples, meant for debugging.")
parser.add_argument("--do-crossval", action="store_true",
                        help="Whether to perform cross-validation on val split. "
                             "Not recommended to set this flag if training is done "
                             "on train + val splits.")

parser.add_argument_group("Checkpointing related arguments")
parser.add_argument("--load-path", default="",
                        help="Path to load checkpoint from and continue training.")
parser.add_argument("--save-path", default="checkpoints/",
                        help="Path of directory to create checkpoint directory "
                             "and save checkpoints.")

# ------------------------------------------------------------------------------------------------
# input arguments and config
# ------------------------------------------------------------------------------------------------

args = parser.parse_args()

# keys: {"dataset", "model", "training", "evaluation"}
config = yaml.load(open(args.config_yml))

# print config and args
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

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


# ------------------------------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ------------------------------------------------------------------------------------------------

train_dataset = VisDialDataset(
    args.train_json, config["dataset"], overfit=args.overfit
)
train_dataloader = DataLoader(
    train_dataset, batch_size=config["training"]["batch_size"], num_workers=args.cpu_workers
)

val_dataset = VisDialDataset(
    args.val_json, config["dataset"], overfit=args.overfit
)
val_dataloader = DataLoader(
    val_dataset, batch_size=config["training"]["batch_size"], num_workers=args.cpu_workers
)


# ------------------------------------------------------------------------------------------------
# setup the model and optimizer
# ------------------------------------------------------------------------------------------------

# let the model know vocabulary size, to declare embedding layer
config["model"]["vocab_size"] = len(train_dataset.vocabulary)

encoder = Encoder(config["model"])
decoder = Decoder(config["model"])

# share word embedding between encoder and decoder
decoder.word_embed = encoder.word_embed

# load parameters from a checkpoint if specified
if args.load_path != "":
    components = torch.load(open(args.load_path))
    encoder.load_state_dict(components["encoder"])
    decoder.load_state_dict(components["decoder"])
    optimizer.load_state_dict(components["optimizer"])
    print("Loaded model from {}".format(args.load_path))

# declare criterion, optimizer and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=config["training"]["initial_lr"])
scheduler = lr_scheduler.StepLR(optimizer,
                                step_size=1,
                                gamma=config["training"]["lr_decay_rate"])

# transfer to assigned device for execution
encoder = encoder.to(device)
decoder = decoder.to(device)
criterion = criterion.to(device)

# wrap around DataParallel to support multi-GPU execution
encoder = nn.DataParallel(encoder, args.gpu_ids)
decoder = nn.DataParallel(decoder, args.gpu_ids)

print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))


# ------------------------------------------------------------------------------------------------
# preparation before training loop
# ------------------------------------------------------------------------------------------------

# record starting time of training, although it is a bit earlier than actual
train_begin = datetime.now()
train_begin_str = datetime.strftime(train_begin, "%d-%b-%Y-%H:%M:%S")
print(f"Training start time: {train_begin_str}")

# set starting epoch based on saved checkpoint name, or start from 1
start_epoch = 1
if args.load_path != "":
    # "path/to/model_epoch_xx.pth" -> xx + 1
    start_epoch = int(args.load_path.split("_")[-1][:-4]) + 1

# create a directory to save checkpoints and copy current config file in it
os.makedirs(os.path.join(args.save_path, train_begin_str))
shutil.copy(args.config_yml, os.path.join(args.save_path, train_begin_str))

# calculate the iterations per epoch
if config["training"]["training_splits"] == "trainval":
    ipe = (len(train_dataset) + len(val_dataset)) // config["training"]["batch_size"]
else:
    ipe = len(train_dataset) // config["training"]["batch_size"]
print("{} iter per epoch.".format(ipe))


# ------------------------------------------------------------------------------------------------
# training loop
# ------------------------------------------------------------------------------------------------

encoder.train()
decoder.train()
running_loss = 0.0
for epoch in range(start_epoch, config["training"]["num_epochs"] + 1):
    # combine data from train and val dataloader, if training using train + val splits
    if config["training"]["training_splits"] == "trainval":
        combined_dataloader = itertools.chain(train_dataloader, val_dataloader)
    else:
        combined_dataloader = itertools.chain(train_dataloader)

    for i, batch in enumerate(combined_dataloader):
        optimizer.zero_grad()

        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = batch[key].to(device)

        # --------------------------------------------------------------------
        # forward-backward pass and optimizer step
        # --------------------------------------------------------------------
        enc_out = encoder(batch)
        dec_out = decoder(enc_out, batch)
        cur_loss = criterion(dec_out, batch["ans_ind"].view(-1))
        cur_loss.backward()
        optimizer.step()

        # --------------------------------------------------------------------
        # update running loss and decay learning rates
        # --------------------------------------------------------------------
        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * cur_loss.item()
        else:
            running_loss = cur_loss.item()

        if optimizer.param_groups[0]["lr"] > config["training"]["minimum_lr"]:
            scheduler.step()

        # --------------------------------------------------------------------
        # print after every few iterations
        # --------------------------------------------------------------------
        if i % 100 == 0:
            # print current time, running average, learning rate, iteration, epoch
            print("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                datetime.now() - train_begin, epoch,
                    (epoch - 1) * ipe + i, running_loss,
                    optimizer.param_groups[0]["lr"]))

    # ------------------------------------------------------------------------
    # cross-validate and report automatic metrics
    # ------------------------------------------------------------------------
    if args.do_crossval:
        print(f"\nCross-validation after epoch {epoch}:")
        all_ranks = []
        for i, batch in enumerate(tqdm(val_dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)

            with torch.no_grad():
                enc_out = encoder(batch)
                dec_out = decoder(enc_out, batch)
            ranks = scores_to_ranks(dec_out)
            gt_ranks = get_gt_ranks(ranks, batch["ans_ind"])
            all_ranks.append(gt_ranks)
        all_ranks = torch.cat(all_ranks, 0)
        process_ranks(all_ranks)

    # ------------------------------------------------------------------------
    # save checkpoint
    # ------------------------------------------------------------------------
    torch.save({
        "encoder": encoder.module.state_dict(),
        "decoder": decoder.module.state_dict(),
        "optimizer": optimizer.state_dict()
    }, os.path.join(args.save_path, train_begin_str, f"model_epoch_{epoch}.pth"))
