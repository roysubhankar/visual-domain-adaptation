import sys
sys.path.append("..")
import os
from argparse import ArgumentParser
import numpy as np
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from datasets import SVHN, MNIST, MNISTM
from model import SVHNConvNet

def test(args, model, dataloader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, y, _ in dataloader['target_test']:
            data, y = data.cuda(), y.cuda()
            class_out, _ = model(data, 0)
            pred = F.softmax(class_out, dim=1).max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    print('[%2d/%d]\tTest Accuracy %.4f (%d/%d)'
          %(epoch, args.num_epochs, correct / len(dataloader['target_test'].dataset) * 100,
            correct, len(dataloader['target_test'].dataset)))

def main(args, model, dataloader):

    # loss criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optmizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    iters = 0

    # load the target loader
    target_iter = iter(dataloader['target_train'])

    for epoch in range(args.num_epochs):
        model.train()
        # iterate through the dataloader
        for i, source_batch in enumerate(dataloader['source_train'], 0):
            source_data, source_y, source_domain = source_batch
            source_data, source_y, source_domain = source_data.cuda(), source_y.cuda(), source_domain.cuda()

            try:
                target_data, _, target_domain = target_iter.next()
            except:
                target_iter = iter(dataloader['target_train'])
                target_data, _, target_domain = target_iter.next()

            target_data, target_domain = target_data.cuda(), target_domain.cuda()

            # alpha
            p = float(i + epoch * len(dataloader['source_train'])) \
                / (args.num_epochs * len(dataloader['source_train']))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # train on source data
            out_source_class, out_source_domain = model(source_data, alpha)
            err_y_source = criterion(out_source_class, source_y)
            err_domain_source = criterion(out_source_domain, source_domain)

            # train on target data
            _, out_target_domain = model(target_data, alpha)
            err_domain_target = criterion(out_target_domain, target_domain)

            # update the weights
            optmizer.zero_grad()
            (err_y_source + err_domain_source + err_domain_target).backward()
            optmizer.step()

            if iters % 100 == 0:
                print('[%2d/%d][%3d/%d]\tClass Loss:%.4f\tDomain Loss:%.4f/%0.4f'
                      %(epoch, args.num_epochs, i, len(dataloader['source_train']),
                        err_y_source.item(), err_domain_source.item(), err_domain_target.item()))

            iters += 1

        test(args, model, dataloader, epoch)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--nc', default=3, type=int, help='number of channels in the image (default: %(default))')
    parser.add_argument('--nclasses', default=10, type=int, help='number of classes (default: %(default))')
    parser.add_argument('--ndomains', default=2, type=int, help='number of domains (default: %(default))')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size (default: %(default))')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate (default: %(default))')
    parser.add_argument('--num_epochs', default=50, type=int, help='number of learning epochs (default: %(default))')
    parser.add_argument('--logdir', default='log', type=str, help='path to the log dir (default: %(default))')
    parser.add_argument('--comment', default=datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), type=str,
                        help='Comment to be appended to the model name to identify the run (default: %(default))')
    parser.add_argument('--model_name', default='dann', type=str,
                        help='Name of the model you want to use. (default: %(default))')
    parser.add_argument('--source', default='svhn', choices=['mnist', 'mnistm', 'svhn', 'syn', 'usps'],
                        help='source dataset (default: %(default))')
    parser.add_argument('--target', default='mnist', choices=['mnist', 'mnistm', 'svhn', 'syn', 'usps'],
                        help='target dataset (default: %(default))')
    parser.add_argument('--data_root', type=str, default='../data', help='path to dataset root (default: %(default))')
    args = parser.parse_args()
    args.run_name = os.path.join(args.logdir, '-'.join([args.model_name, args.comment]))
    print(args)

    assert args.source != args.target, "source and target can not be the same!!!"

    # svhn-like transforms
    dset_transforms = list()
    dset_transforms.append(transforms.ToTensor())
    dset_transforms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    dset_transforms = transforms.Compose(dset_transforms)

    # mnist-like grayscale dataset transforms
    gray_transforms = list()
    gray_transforms.append(transforms.Resize((32, 32)))
    gray_transforms.append(transforms.ToTensor())
    gray_transforms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    gray_transforms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    gray_transforms = transforms.Compose(gray_transforms)

    # mnistm transforms
    mnistm_transforms = list()
    mnistm_transforms.append(transforms.Resize((32, 32)))
    mnistm_transforms.append(transforms.ToTensor())
    mnistm_transforms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    mnistm_transforms = transforms.Compose(mnistm_transforms)

    # create the dataloaders
    dataloader = {}
    if args.source == 'svhn':
        dataloader['source_train'] = DataLoader(SVHN(os.path.join(args.data_root, args.source), split='train',
                                                transform=dset_transforms, domain_label=0, download=True),
                                                batch_size=args.batch_size, shuffle=True, drop_last=True)
    elif args.source == 'mnist':
        dataloader['source_train'] = DataLoader(MNIST(os.path.join(args.data_root, args.source), train=True,
                                                transform=gray_transforms, domain_label=0, download=True),
                                                batch_size=args.batch_size, shuffle=True, drop_last=True)
    elif args.source == 'mnistm':
        dataloader['source_train'] = DataLoader(MNISTM(os.path.join(args.data_root, args.source), train=True,
                                                      transform=mnistm_transforms, domain_label=0, download=True),
                                                batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        raise NotImplementedError

    if args.target == 'svhn':
        dataloader['target_train'] = DataLoader(SVHN(os.path.join(args.data_root, args.target), split='train',
                                                transform=dset_transforms, domain_label=1, download=True),
                                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataloader['target_test'] = DataLoader(SVHN(os.path.join(args.data_root, args.target), split='test',
                                               transform=dset_transforms, domain_label=1, download=True),
                                               batch_size=args.batch_size, shuffle=False, drop_last=False)
    elif args.target == 'mnist':
        dataloader['target_train'] = DataLoader(MNIST(os.path.join(args.data_root, args.target), train=True,
                                                transform=gray_transforms, domain_label=1, download=True),
                                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataloader['target_test'] = DataLoader(MNIST(os.path.join(args.data_root, args.target), train=False,
                                                transform=gray_transforms, domain_label=1, download=True),
                                                batch_size=args.batch_size, shuffle=False, drop_last=False)
    elif args.target == 'mnistm':
        dataloader['target_train'] = DataLoader(MNISTM(os.path.join(args.data_root, args.target), train=True,
                                                      transform=mnistm_transforms, domain_label=1, download=True),
                                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataloader['target_test'] = DataLoader(MNISTM(os.path.join(args.data_root, args.target), train=False,
                                                     transform=mnistm_transforms, domain_label=1, download=True),
                                               batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        raise NotImplementedError

    # define the model
    model = SVHNConvNet(nc=args.nc, nclasses=args.nclasses, ndomains=args.ndomains).cuda()
    print(model)

    os.makedirs(args.run_name, exist_ok=True)
    with open(os.path.join(args.run_name, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    main(args, model, dataloader)