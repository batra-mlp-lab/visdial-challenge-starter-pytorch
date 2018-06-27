import argparse
import gc
import math

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import VisDialDataset
from encoders.lf import LateFusionEncoder
from decoders.disc import DiscriminativeDecoder


parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)
LateFusionEncoder.add_cmdline_args(parser)

parser.add_argument_group('Optimization related arguments')
parser.add_argument('-batch_size', default=4, help='Batch size')
parser.add_argument('-learning_rate', default=1e-3, help='Learning rate')
parser.add_argument('-dropout', default=0.5, help='Dropout')
parser.add_argument('-num_epochs', default=20, help='Epochs')
parser.add_argument('-weight_init', default='xavier', choices=['xavier', 'kaiming'],
                        help='Weight initialization strategy')
parser.add_argument('-gpuid', default=0, help='GPU id to use')

# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()

# seed for reproducibility
torch.manual_seed(1234)

# set device and default tensor type
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    torch.cuda.set_device(args.gpuid)

# transfer all options to model
model_args = args

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

args.concat_history = True
dataset = VisDialDataset(args, ['train'])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=dataset.collate_fn)

# ----------------------------------------------------------------------------
# setting model parameters
# ----------------------------------------------------------------------------

# transfer some useful args from dataloader to model
for key in {'num_data_points', 'vocab_size', 'max_ques_count',
            'max_ques_len', 'max_ans_len'}:
    setattr(model_args, key, getattr(dataset, key))

model_args.training = True

# iterations per epoch
setattr(model_args, 'iter_per_epoch', 
    math.ceil(dataset.num_data_points['train'] / args.batch_size))
print("{} iter per epoch.".format(model_args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

print("Encoder: {}".format('lf-ques-im-hist'))  # todo: support more encoders
print("Decoder: {}".format('disc'))  # todo : support more? or not?

encoder = LateFusionEncoder(model_args)
decoder = DiscriminativeDecoder(model_args)
# todo - add weight init

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.learning_rate)

if args.gpuid >= 0:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    criterion = criterion.cuda()

# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

print("Training...")
encoder.train()
decoder.train()

running_loss = 0.0
for epoch in range(1, model_args.num_epochs + 1):
    for i, train_batch in enumerate(dataloader):
        optimizer.zero_grad()

        if args.gpuid >= 0:
            for key in train_batch:
                train_batch[key] = train_batch[key].cuda()
        
        # forward pass to encoder
        img = Variable(train_batch['img_feat'])
        ques = Variable(train_batch['ques_fwd'])
        hist = Variable(train_batch['hist'])
        enc_out = encoder(img, ques, hist)

        # forward pass to decoder
        options = Variable(train_batch['opt'])
        ans_ind = Variable(train_batch['ans_ind'])
        dec_out = decoder(enc_out, options)

        cur_loss = criterion(dec_out, ans_ind.view(-1))
        cur_loss.backward()

        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * cur_loss.data[0]
        else:
            running_loss = cur_loss.data[0]
        optimizer.step()
        gc.collect()

        # print after every few iterations
        if i % 100 == 0:
            # print current time, running average, learning rate, iteration, epoch
            print('[Epoch:%d][Iter:%d][Loss:%.05f][lr:%f]' % (epoch, i, running_loss, args.learning_rate))
