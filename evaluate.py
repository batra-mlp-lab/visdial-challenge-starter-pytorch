import argparse
import datetime
import os

import torch
from torch.utils.data import DataLoader

from dataloader import VisDialDataset


parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)

parser.add_argument_group('Evaluation related arguments')
parser.add_argument('-load_path', default='checkpoints/model.pth',
                        help='Checkpoint to load path from')
parser.add_argument('-split', default='val', choices=['val', 'test'],
                        help='Split to evaluate on')
parser.add_argument('-use_gt', action='store_true',
                        help='Whether to use ground truth for retrieving ranks')
parser.add_argument('-batch_size', default=12, type=int, help='Batch size')
parser.add_argument('-gpuid', default=0, type=int, help='GPU id to use')

parser.add_argument_group('Submission related arguments')
parser.add_argument('-save_ranks', action='store_true', help='Whether to save retrieved ranks')
parser.add_argument('-save_path', default='logs/ranks.json', help='Path of json file to save ranks')

# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()
if args.use_gt and args.split == 'test':
    print("Warning: No ground truth available for test split, changing use_gt to False.")
    args.use_gt = False

# seed for reproducibility
torch.manual_seed(1234)

# set device and default tensor type
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    torch.cuda.set_device(args.gpuid)

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------

components = torch.load(args.load_path)
model_args = components['encoder'].args
model_args.gpuid = args.gpuid
model_args.batch_size = args.batch_size

# todo: remove this y saving only state dicts
components['encoder'].args.training = False
components['decoder'].args.training = False

# this is required by dataloader
args.img_norm = components['encoder'].args.img_norm

# set this because only late fusion encoder is supported yet
args.concat_history = True

for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = VisDialDataset(args, [args.split])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=dataset.collate_fn)

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

encoder = components['encoder']
decoder = components['decoder']
print("Loaded model from {}".format(args.load_path))

if args.gpuid >= 0:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

print("Evaluation start time: {}".format(
    datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')))
encoder.eval()
decoder.eval()
os.makedirs(args.save_path, exist_ok=True)

if args.use_gt:
    pass
    # evaluation and retrieval
else:
    pass
    # inference

if args.save_ranks:
    print("Writing ranks to {}".format(args.save_path))
    os.makedirs(os.path.dirname(args.save_path))
    json.dump(ranks, open(args.save_path, 'w'))
