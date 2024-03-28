import torch
import numpy as np
import options
from datasets import MyDataset, collate_fn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import model as model_pkg
from models import loss
import os
from torch import nn
import json
import shutil

import train
from test import test_all
from datetime import datetime


class WarmupLR:
    def __init__(self, optimizer, num_warm, lr_init) -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        self.lr_init = lr_init
        self.lr = [group['lr'] for group in self.optimizer.param_groups]
        self.num_step = 0

    def __compute(self, lr) -> float:
        return self.lr_init + (lr - self.lr_init) * ((self.num_step - 1) / (self.num_warm - 1))
        # return lr * min(self.num_step ** (-0.5), self.num_step * self.num_warm ** (-1.5))

    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optim(model, args):
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception("Unknown optimizer")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=args.epoch - args.warmup - args.warm_epc, eta_min=args.lr * args.decay_rate)
        elif args.lr_decay == 'multistep':
            scheduler = torch.optim.\
                lr_scheduler.MultiStepLR(optim, milestones=[args.epoch - args.warmup - args.warm_epc - 100], gamma=args.decay_rate)
        else:
            raise Exception("Unknown Scheduler")
    else:
        scheduler = None
    return scheduler


if __name__ == '__main__':
    args = options.parser.parse_args()
    dt = datetime.now()
    uid = dt.strftime('%y%m%d_%H%M%S_')
    orig_exp = args.exp_name
    args.exp_name = uid + args.exp_name

    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 10000)

    print('=============seed: {}============='.format(seed))
    setup_seed(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    '''
    1. load data
    '''
    '''
    train data
    '''
    train_data = MyDataset(args.dataset, args.root, 'train.split{}.bundle'.format(args.split), args.sample_rate, 'rand')
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=8)
    print(len(train_data))

    '''
    test data
    '''
    test_data = MyDataset(args.dataset, args.root, 'test.split{}.bundle'.format(args.split), args.sample_rate, 'mid')
    test_loader = DataLoader(test_data, batch_size=32, collate_fn=collate_fn, shuffle=False, num_workers=8)
    print(len(test_data))
    print('=============Load dataset successfully=============')

    '''
    2. load model
    '''
    in_dim = train_data.feat_dim
    n_class = train_data.n_cls
    model = model_pkg.Trans(in_dim, args.hidden_dim, args.n_head, args.n_encoder, n_class, args.dropout, args).to(device)
    loss_fn = loss.LossFn(n_class, train_data.bg_cls, args, device).to(device)
    train_fn = train.train_epoch
    if args.ckpt is not None:
        checkpoint = torch.load('./ckpt/' + args.ckpt + '.pkl', map_location='cpu')
        model.load_state_dict(checkpoint)
    print('=============Load model successfully=============')
    print(args)

    '''
    test mode
    '''
    if args.test:
        ret = test_all(0, model, test_loader, None, device, args, test_data.bg_cls, fully_eva=True)
        print('Test results:')
        for k, v in ret.items():
            print('{}: {}'.format(k, v))

        if args.save:
            np.save(f'{orig_exp}.npy', ret)
        raise SystemExit

    '''
    3. record
    '''
    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.exp_name):
        os.makedirs("./logs/" + args.exp_name)
    logger = SummaryWriter(os.path.join('./logs/', args.exp_name))
    with open(os.path.join('./logs', args.exp_name, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    '''
    4. train
    '''
    optim = get_optim(model, args)
    warmup = WarmupLR(optim, args.warmup, 0.01 * args.lr) if args.warmup else None
    scheduler = None
    print('=============Begin training=============')

    second_stage = False
    for epc in range(args.epoch):
        if epc == args.warm_epc:
            second_stage = True
            optim = get_optim(model, args)      # re-initialize the optimizer
            warmup = WarmupLR(optim, args.warmup, 0.1 * args.lr) if args.warmup else None
            scheduler = get_scheduler(optim, args)      # initialize the scheduler
        if second_stage and warmup is not None and epc < args.warmup + args.warm_epc:
            # second stage warmup
            warmup.step()
        elif warmup is not None and epc < args.warmup:
            # first stage warmup
            warmup.step()
        print('Epoch {} lr: {:.6f}'.format(epc, optim.state_dict()['param_groups'][0]['lr']))
        avg_loss = train_fn(epc, model, loss_fn, train_loader, optim, logger, device, args)
        if second_stage and scheduler is not None and epc >= args.warmup + args.warm_epc:
            scheduler.step()
        ret = test_all(epc, model, test_loader, logger, device, args, test_data.bg_cls)
        print('Epoch: {}\tAvg Loss: {:.4f}\tTest Acc: {:.3f}'
              .format(epc, avg_loss, ret['acc']))

    torch.save(model.state_dict(), './ckpt/' + args.exp_name + '.pkl')
    ret = test_all(0, model, test_loader, None, device, args, test_data.bg_cls, fully_eva=True)
    print('Final Test:')
    for k, v in ret.items():
        print('{}: {}'.format(k, v))
    if args.save:
        np.save(f'{orig_exp}.npy', ret)
