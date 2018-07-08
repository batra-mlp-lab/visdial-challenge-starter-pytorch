import argparse
import datetime
import gc
import math
import os

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataloader import VisDialDataset
from encoders.lf import LateFusionEncoder
from decoders.disc import DiscriminativeDecoder


parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)
LateFusionEncoder.add_cmdline_args(parser)

parser.add_argument_group('Optimization related arguments')
parser.add_argument('-num_epochs', default=20, type=int, help='Epochs')
parser.add_argument('-batch_size', default=12, type=int, help='Batch size')
parser.add_argument('-lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('-lr_decay_rate', default=0.9997592083, type=float, help='Decay for lr')
parser.add_argument('-min_lr', default=5e-5, type=float, help='Minimum learning rate')
parser.add_argument('-weight_init', default='xavier', choices=['xavier', 'kaiming'],
                        help='Weight initialization strategy')
parser.add_argument('-overfit', action='store_true',
                        help='Overfit on 5 examples, meant for debugging')
parser.add_argument('-gpuid', default=0, type=int, help='GPU id to use')

parser.add_argument_group('Checkpointing related arguments')
parser.add_argument('-load_path', default='', help='Checkpoint to load path from')
parser.add_argument('-save_path', default='checkpoints/', help='Path to save checkpoints')
parser.add_argument('-save_step', default=2, type=int,
                        help='Save checkpoint after every save_step epochs')

# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()
start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
if args.save_path == 'checkpoints/':
    args.save_path += start_time

# seed for reproducibility
torch.manual_seed(1234)

# set device and default tensor type
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    torch.cuda.set_device(args.gpuid)

# transfer all options to model
model_args = args

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------

if args.load_path != '':
    components = torch.load(args.load_path)
    model_args = components['model_args']
    model_args.gpuid = args.gpuid
    model_args.batch_size = args.batch_size

    # this is required by dataloader
    args.img_norm = model_args.img_norm

# set this because only late fusion encoder is supported yet
args.concat_history = True

for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = VisDialDataset(args, ['train'])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=dataset.collate_fn)

# ----------------------------------------------------------------------------
# setting model args
# ----------------------------------------------------------------------------

# transfer some useful args from dataloader to model
for key in {'num_data_points', 'vocab_size', 'max_ques_count',
            'max_ques_len', 'max_ans_len'}:
    setattr(model_args, key, getattr(dataset, key))

# iterations per epoch
setattr(args, 'iter_per_epoch', 
    math.ceil(dataset.num_data_points['train'] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

encoder = LateFusionEncoder(model_args)
decoder = DiscriminativeDecoder(model_args, encoder)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay_rate)

if args.load_path != '':
    encoder.load_state_dict(components['encoder'])
    decoder.load_state_dict(components['decoder'])
    print("Loaded model from {}".format(args.load_path))
print("Encoder: {}".format('lf-ques-im-hist'))  # todo: support more encoders
print("Decoder: {}".format('disc'))

if args.gpuid >= 0:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    criterion = criterion.cuda()

# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

encoder.train()
decoder.train()
os.makedirs(args.save_path, exist_ok=True)

running_loss = 0.0
train_begin = datetime.datetime.utcnow()
print("Training start time: {}".format(
    datetime.datetime.strftime(train_begin, '%d-%b-%Y-%H:%M:%S')))

for epoch in range(1, model_args.num_epochs + 1):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        if args.gpuid >= 0:
            for key in batch:
                if not isinstance(batch[key], list):
                    batch[key] = Variable(batch[key].cuda())

        # --------------------------------------------------------------------
        # forward-backward pass and optimizer step
        # --------------------------------------------------------------------
        enc_out = encoder(batch)
        dec_out = decoder(enc_out, batch)
        cur_loss = criterion(dec_out, batch['ans_ind'].view(-1))
        cur_loss.backward()

        optimizer.step()
        gc.collect()

        # --------------------------------------------------------------------
        # update running loss and decay learning rates
        # --------------------------------------------------------------------
        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * cur_loss.data[0]
        else:
            running_loss = cur_loss.data[0]

        if optimizer.param_groups[0]['lr'] > args.min_lr:
            scheduler.step()

        # --------------------------------------------------------------------
        # print after every few iterations
        # --------------------------------------------------------------------
        if i % 100 == 0:
            # print current time, running average, learning rate, iteration, epoch
            print("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                datetime.datetime.utcnow() - train_begin, epoch,
                    (epoch - 1) * args.iter_per_epoch + i, running_loss,
                    optimizer.param_groups[0]['lr']))

    # ------------------------------------------------------------------------
    # save checkpoints and final model
    # ------------------------------------------------------------------------
    if epoch % args.save_step == 0:
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': encoder.args
        }, os.path.join(args.save_path, 'model_epoch_{}.pth'.format(epoch)))

torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': encoder.args
}, os.path.join(args.save_path, 'model_final.pth'))
