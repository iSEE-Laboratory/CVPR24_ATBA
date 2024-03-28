import torch
from torch import nn
from models.transformer import Transformer
import torch.nn.functional as F
from datasets import MyDataset, collate_fn
from torch.utils.data import DataLoader
import numpy as np


class Trans(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_cls, dropout, args):
        super(Trans, self).__init__()
        self.in_proj = nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=hidden_dim)

        if args.dataset == 'breakfast':
            max_len = 10000
        elif args.dataset == 'hollywood':
            max_len = 3000
        elif args.dataset == '50salads':
            max_len = 20000
        else:
            max_len = 10000
        self.pos_embedding = nn.Embedding(max_len // args.sample_rate, hidden_dim)

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=0,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            dropout=dropout,
            norm_first=True
        )

        self.cls_token = nn.Embedding(n_cls, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_cls)

        self.n_cls = n_cls

        self.args = args

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.pos_embedding.weight, std=0.01)

    def forward(self, x, mask):
        # x (b, t, c), mask (b, t)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)
        x += self.pos_embedding.weight.unsqueeze(0)[:, :t].to(x.device)

        cls_token = self.cls_token.weight.unsqueeze(0).repeat(b, 1, 1)  # (b, n, c)
        tokens = torch.cat([cls_token, x], dim=1)
        cls_mask = torch.ones((b, self.n_cls), device=mask.device).bool()
        tok_mask = torch.cat([cls_mask, mask], dim=1)

        # pyramid local masks
        for i, mod in enumerate(self.transformer.encoder.layers):
            local_mask = torch.zeros((t, t)).bool()
            local_r = min(2 ** i, t - 1)
            for j in range(-local_r, local_r + 1):
                local_mask |= torch.diag_embed(torch.ones(t-abs(j)).bool(), offset=j, dim1=-2, dim2=-1)
            src_mask = torch.ones((t+self.n_cls, t+self.n_cls)).bool()
            src_mask[self.n_cls:, self.n_cls:] = local_mask
            src_mask = src_mask.to(x.device)

            src_mask = src_mask.unsqueeze(0).repeat(b, 1, 1)
            # To avoid the attention NAN bug
            # if not, the padding tokens (~tok_mask) can attend nothing, and produce all zero pre-softmax weights
            src_mask[~tok_mask] = True

            tokens = mod(tokens, src_mask=~src_mask, src_key_padding_mask=~tok_mask)

        cls_prob = self.classifier(tokens)
        return {'tok_cls': torch.diagonal(cls_prob[:, :self.n_cls, :], dim1=1, dim2=2),
                'fr_cls': cls_prob[:, self.n_cls:, :],
                'feat': tokens}
