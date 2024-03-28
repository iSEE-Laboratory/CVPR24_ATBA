import numpy as np
import pdb
import torch
import torch.nn.functional as F
from utils import AverageMeter
from copy import deepcopy


__loss_name__ = ['total', 'tok_loss', 'fr_loss', 'glc_loss']


def train_epoch(epoch, model, loss_fn, train_loader, optim, logger, device, args):
    model.train()
    loss_meters = {name: AverageMeter(name, logger) for name in __loss_name__}
    for i, data in enumerate(train_loader):
        feat = data['feat'].to(device)
        tr = [t.to(device) for t in data['transcript']]
        mask = data['mask'].to(device)

        out = model(feat, mask)
        loss_out = loss_fn(epoch, out['tok_cls'], out['fr_cls'], mask, tr,
                           data['multi_hot'].to(device), out['feat'], data['name'])

        optim.zero_grad()
        loss_out['total'].backward()
        optim.step()

        [loss_meters[name].update(loss_out[name], feat.shape[0]) for name in __loss_name__]
        if i % 10 == 0:
            print('[{}/{}]\ttotal loss: {}'.format(i, epoch, loss_out['total']))
    avg_losses = {name: loss_meters[name].done(epoch) for name in __loss_name__}
    return avg_losses['total']
