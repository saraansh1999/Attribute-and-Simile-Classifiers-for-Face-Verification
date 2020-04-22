import time, os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

from train import *
from networks import *
from datasets import *

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument('--train_dir', action='store', required=True)
parser.add_argument('--test_dir', action='store', required=True)
parser.add_argument('--att_file', action='store', required=True)
parser.add_argument('--trait', action='store', required=True)
parser.add_argument('--model', action='store', required=True)
parser.add_argument('--save_path', action='store', required=True)

# Optional arguments
parser.add_argument('--sample_thresh', action='store', default=0.75)
parser.add_argument('--batch_size', action='store', default=128)
parser.add_argument('--testbatch_size', action='store', default=128)
parser.add_argument('--num_epochs', type=int, action='store', default=50)

# Default arguments.0
parser.add_argument('--lr', action='store', type=float, default=0.001)
parser.add_argument('--weight_decay', action='store', default=0.00004)
parser.add_argument('--step_size', action='store', type=int, default=12)
parser.add_argument('--gamma', action='store', default=0.1)
parser.add_argument('--workers', action='store', default=10)

args = parser.parse_args()

print("Trait ->", args.trait)

if not os.path.exists(args.save_path):
	os.makedirs(args.save_path)

print('==> Options:', args)
print("Using GPU:", torch.cuda.is_available())

phases = ['train', 'test']
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

dataset = {}
if args.model != 'linear':
	dataset['train'], dataset['test'] = CNNDataset(args.train_dir, 'train', args, args.sample_thresh), CNNDataset(args.test_dir, 'test', args)
else:
	dataset['train'], dataset['test'] = MyDataset(args.train_dir, 'train', args, args.sample_thresh), MyDataset(args.test_dir, 'test', args)

dataloader = {}
dataloader['train'] = torch.utils.data.DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
dataloader['test'] = torch.utils.data.DataLoader(dataset['test'], batch_size=args.testbatch_size, shuffle=False, num_workers=args.workers)

loss_fn = nn.CrossEntropyLoss()

if args.model == 'linear':
	model = NetBN()
elif args.model == 'CNN':
	model = CNNNet()
elif args.model == 'mobilenet':
	model = mobilenet_v2()

if torch.cuda.is_available():
	model = model.cuda()
print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

train(optimizer, scheduler, dataloader, loss_fn, model, args, phases)