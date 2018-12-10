import argparse
from datetime import datetime
import os

import pprint
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from visdialch.dataloader import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder


parser = argparse.ArgumentParser()
parser.add_argument("--config-yml", default="configs/lf_disc_vgg16_fc7_bs20.yml",
                        help="Path to a config file listing reader, model and "
                             "optimization parameters.")

parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument("--gpu-ids", nargs="+", type=int, default=-1,
                        help="List of ids of GPUs to use.")
parser.add_argument("--cpu-workers", type=int, default=8,
                        help="Number of CPU workers for reading data.")
parser.add_argument("--overfit", action="store_true",
                        help="Overfit model on 5 examples, meant for debugging.")

parser.add_argument_group("Checkpointing related arguments")
parser.add_argument("--load-path", default="",
                        help="Path to load checkpoint from and continue training.")
parser.add_argument("--save-path", default="checkpoints/",
                        help="Path of directory to create checkpoint directory "
                             "and save checkpoints.")

# ----------------------------------------------------------------------------
# input arguments and config
# ----------------------------------------------------------------------------

args = parser.parse_args()

# keys: {"dataset", "model", "training"}
config = yaml.load(open(args.config_yml))
pprint.format(config)

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# set CPU/GPU device for execution
if args.gpu_ids[0] > 0:
    device = torch.device("cuda", args.gpu_ids[0])
else:
    device = torch.device("cpu")


# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = VisDialDataset(config["dataset"], ["train"])
dataloader = DataLoader(dataset,
                        batch_size=config["training"]["batch_size"],
                        shuffle=False,
                        collate_fn=dataset.collate_fn,
                        num_workers=args.cpu_workers)

# transfer some attributes from dataset to model args
for key in {"vocab_size", "max_ques_count"}:
    config["model"][key] = getattr(dataset, key)


# ----------------------------------------------------------------------------
# setup the model and optimizer
# ----------------------------------------------------------------------------

encoder = Encoder(config["model"])
decoder = Decoder(config["model"], encoder)

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
                                gamma=config["lr_decay_rate"])

# transfer to assigned device for execution
encoder = encoder.to(device)
decoder = decoder.to(device)
criterion = criterion.to(device)

# wrap around DataParallel to support multi-GPU execution
encoder = nn.DataParallel(encoder, args.gpu_ids[0])
decoder = nn.DataParallel(decoder, args.gpu_ids[0])

print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))


# ----------------------------------------------------------------------------
# preparation before training loop
# ----------------------------------------------------------------------------

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
ipe = dataset.num_data_points["train"] // config["training"]["batch_size"]
print("{} iter per epoch.".format(ipe))


# ----------------------------------------------------------------------------
# training loop
# ----------------------------------------------------------------------------

encoder.train()
decoder.train()
running_loss = 0.0
for epoch in range(start_epoch, config["num_epochs"] + 1):
    for i, batch in enumerate(dataloader):
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
    # save checkpoint
    # ------------------------------------------------------------------------
    torch.save({
        "encoder": encoder.module.state_dict(),
        "decoder": decoder.module.state_dict(),
        "optimizer": optimizer.state_dict()
    }, os.path.join(args.save_path, "model_epoch_{}.pth".format(epoch)))
