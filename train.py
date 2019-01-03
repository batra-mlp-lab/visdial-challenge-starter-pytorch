import argparse
from datetime import datetime
import itertools
import os

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.model import EncoderDecoderModel
from visdialch.utils import process_ranks, scores_to_ranks, get_gt_ranks
from visdialch.utils.checkpointing import CheckpointManager, load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml", default="configs/lf_disc_vgg16_fc7_bs20.yml",
    help="Path to a config file listing reader, model and solver parameters."
)
parser.add_argument(
    "--train-json", default="data/visdial_1.0_train.json",
    help="Path to json file containing VisDial v1.0 training data."
)
parser.add_argument(
    "--val-json", default="data/visdial_1.0_val.json",
    help="Path to json file containing VisDial v1.0 validation data."
)


parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument(
    "--gpu-ids", nargs="+", type=int, default=0,
    help="List of ids of GPUs to use."
)
parser.add_argument(
    "--cpu-workers", type=int, default=4,
    help="Number of CPU workers for dataloader."
)
parser.add_argument(
    "--overfit", action="store_true",
    help="Overfit model on 5 examples, meant for debugging."
)
parser.add_argument(
    "--validate", action="store_true",
    help="Whether to validate on val split after every epoch."
)
parser.add_argument(
    "--in-memory", action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. Use only in "
         "presence of large RAM, atleast few tens of GBs."
)


parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath", default="checkpoints/",
    help="Path of directory to create checkpoint directory and save checkpoints."
)
parser.add_argument(
    "--load-pthpath", default="",
    help="To continue training, path to .pth file of saved checkpoint."
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

if isinstance(args.gpu_ids, int): args.gpu_ids = [args.gpu_ids]
device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

# print config and args
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))


# ================================================================================================
#   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, CHECKPOINT MANAGER
# ================================================================================================

train_dataset = VisDialDataset(args.train_json, config["dataset"], args.overfit, args.in_memory)
train_dataloader = DataLoader(
    train_dataset, batch_size=config["solver"]["batch_size"], num_workers=args.cpu_workers
)

val_dataset = VisDialDataset(args.val_json, config["dataset"], args.overfit, args.in_memory)
val_dataloader = DataLoader(
    val_dataset, batch_size=config["solver"]["batch_size"], num_workers=args.cpu_workers
)

# pass vocabulary to construct nn.Embedding
encoder = Encoder(config["model"], train_dataset.vocabulary)
decoder = Decoder(config["model"], train_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# share word embedding between encoder and decoder
decoder.word_embed = encoder.word_embed

# wrap encoder and decoder in a model
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["solver"]["initial_lr"])
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=config["solver"]["lr_gamma"])

checkpoint_manager = CheckpointManager(model, optimizer, args.save_dirpath, config=config)


# ================================================================================================
#   SETUP BEFORE TRAINING LOOP
# ================================================================================================

# if loading from checkpoint, adjust start epoch and load parameters
if args.load_pthpath == "":
    start_epoch = 0
else:
    # "path/to/checkpoint_xx.pth" -> xx
    start_epoch = int(args.load_pthpath.split("_")[-1][:-4])

    model_state_dict, optimizer_state_dict = load_checkpoint(args.load_pthpath)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    print("Loaded model from {}".format(args.load_pthpath))


# ================================================================================================
#   TRAINING LOOP
# ================================================================================================

running_loss = 0.0
train_begin = datetime.now()
for epoch in range(start_epoch, config["solver"]["num_epochs"] + 1):

    # --------------------------------------------------------------------------------------------
    #   ON EPOCH START  (combine dataloaders if training on train + val)
    # --------------------------------------------------------------------------------------------
    if config["solver"]["training_splits"] == "trainval":
        combined_dataloader = itertools.chain(train_dataloader, val_dataloader)
        iterations = (len(train_dataset) + len(val_dataset)) // config["solver"]["batch_size"] + 1
    else:
        combined_dataloader = itertools.chain(train_dataloader)
        iterations = len(train_dataset) // config["solver"]["batch_size"] + 1
    print(f"Number of iterations this epoch: {iterations}")

    for i, batch in enumerate(combined_dataloader):
        # ----------------------------------------------------------------------------------------
        #   ON ITERATION START  (shift all tensors to "device")
        # ----------------------------------------------------------------------------------------
        for key in batch:
            batch[key] = batch[key].to(device)

        # ----------------------------------------------------------------------------------------
        #   ITERATION: FORWARD - BACKWARD - STEP
        # ----------------------------------------------------------------------------------------
        optimizer.zero_grad()
        output = model(batch)
        batch_loss = criterion(output.view(-1, output.size(-1)), batch["ans_ind"].view(-1))
        batch_loss.backward()
        optimizer.step()

        # ----------------------------------------------------------------------------------------
        #   ON ITERATION END  (running loss, print training progress)
        # ----------------------------------------------------------------------------------------
        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
        else:
            running_loss = batch_loss.item()

        if optimizer.param_groups[0]["lr"] > config["solver"]["minimum_lr"]:
            scheduler.step()

        if i % 100 == 0:
            # print current time, epoch, iteration, running loss, learning rate
            print("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                    datetime.now() - train_begin, epoch, i,
                    running_loss, optimizer.param_groups[0]["lr"]
                )
            )

    # --------------------------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # --------------------------------------------------------------------------------------------
    checkpoint_manager.step()

    # validate and report automatic metrics
    if args.validate:
        print(f"Validation after epoch {epoch}:")
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

