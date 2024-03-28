from torch.utils.data import Dataset
import os
import numpy as np
import random
import torch


class MyDataset(Dataset):
    def __init__(self, dataset_name, root, split_name, sample_rate, sample_type):
        assert dataset_name in ('breakfast', 'hollywood', 'crosstask')
        self.dataset_name = dataset_name
        self.root = os.path.join(root, dataset_name)
        self.sample_rate = sample_rate
        self.sample_type = sample_type
        self.video_lst, self.gts, self.trans, self.n_cls = self.load_data(split_name)

        self.bg_cls = 0         # background class id
        if dataset_name == 'crosstask':
            self.feat_dim = 3200
        else:
            self.feat_dim = 2048

    def load_data(self, split_name):
        # load video names
        samples = []
        with open(os.path.join(self.root, 'splits', split_name), 'r') as f:
            for line in f:
                line = line.strip()
                line = os.path.splitext(line)
                samples.append(line[0])

        # load label2idx mapping
        label2idx = {}
        with open(os.path.join(self.root, 'mapping.txt'), 'r') as f:
            for line in f:
                line = line.strip().split()
                label2idx[line[1]] = int(line[0])

        # read labels and transcripts
        gts, trans = [], []
        for name in samples:
            with open(os.path.join(self.root, 'groundTruth', name + '.txt'), 'r') as f:
                gt = [label2idx[line.strip()] for line in f]
            with open(os.path.join(self.root, 'transcripts', name + '.txt'), 'r') as f:
                tr = [label2idx[line.strip()] for line in f]
            gts.append(gt)
            trans.append(tr)
        return samples, gts, trans, len(label2idx)

    def __len__(self):
        return len(self.video_lst)

    def __getitem__(self, idx):
        feat = np.load(os.path.join(self.root, 'features', self.video_lst[idx] + '.npy'))  # (t, c)
        if feat.dtype == np.float64:
            feat = feat.astype(np.float32)
        if self.dataset_name == 'breakfast' or self.dataset_name == 'crosstask':
            feat = feat.T       # (t, c)
        gt = np.array(self.gts[idx])
        tr = np.array(self.trans[idx])
        if self.dataset_name == 'hollywood':
            # The features are extracted by us, and the number of frames may not be exactly the same as
            # the provided label length (The new feature is either 6 or 7 timestamps longer than the label)
            diff = feat.shape[0] - gt.shape[0]
            if diff > 0:
                feat = feat[:gt.shape[0]]
            elif diff < 0:
                gt = gt[:feat.shape[0]]
        assert feat.shape[0] == gt.shape[0]

        # gt in original length (not downsampled)
        raw_gt = gt.copy()
        raw_len = gt.shape[0]

        # temporal downsampling
        sampled_ts = self.sampling_fun(feat.shape[0], self.sample_rate, self.sample_type)
        feat, gt = feat[sampled_ts], gt[sampled_ts]

        # video-level multi-hot labels
        vid_la = set(tr)
        multihot = np.zeros(self.n_cls)
        for la in vid_la:
            multihot[la] = 1

        ret = {
            'name': self.video_lst[idx],
            'feat': feat,
            'gt': gt,
            'transcript': tr,
            'multi_hot': multihot,

            'raw_gt': raw_gt,
            'raw_len': raw_len,
        }
        return ret

    def sampling_fun(self, T, GAP, sample_type):
        '''
        ref: DPDTW (CVPR21)
        '''
        start_idxes = list(range(0, T, GAP))
        N = len(start_idxes)
        idxes = start_idxes + [T]

        sample_ts = []
        for i in range(N):
            start_i = idxes[i]
            end_i = idxes[i+1] - 1
            assert start_i <= end_i, (start_i, end_i)
            if sample_type == 'mid':
                sample_ts.append(int((start_i+end_i)/2))
            elif sample_type == 'rand':
                sample_ts.append(random.randint(start_i, end_i))
            else:
                raise ValueError('Unknown sample method: {}'.format(sample_type))
        return sample_ts


def collate_fn(sample):
    max_len = max([s["feat"].shape[0] for s in sample])

    name_lst, feat_lst, gt_lst, mask_lst = [], [], [], []
    for s in sample:
        name_lst.append(s['name'])
        feat, gt = s['feat'], s['gt']
        t = feat.shape[0]
        pad_t = max_len - t
        feat = np.pad(feat, ((0, pad_t), (0, 0)), mode='constant', constant_values=0)
        gt = np.pad(gt, (0, pad_t), mode='constant', constant_values=0)
        feat, gt = torch.from_numpy(feat), torch.from_numpy(gt)
        mask = torch.zeros(max_len)
        mask[:t] = torch.ones(t)
        feat_lst.append(feat)
        gt_lst.append(gt)
        mask_lst.append(mask.bool())

    feat_lst = torch.stack(feat_lst, dim=0)     # (b, t, c)
    gt_lst = torch.stack(gt_lst, dim=0)         # (b, t)
    mask_lst = torch.stack(mask_lst, dim=0)     # (b, t)
    tr_lst = [torch.LongTensor(s['transcript']) for s in sample]
    mh_lst = [torch.from_numpy(s['multi_hot']).int() for s in sample]
    mh_lst = torch.stack(mh_lst, dim=0)         # (b, cls)

    raw_gt_lst = [torch.LongTensor(s['raw_gt']) for s in sample]
    raw_len_lst = torch.LongTensor([s['raw_len'] for s in sample])

    ret = {
        'name': name_lst,
        'feat': feat_lst,
        'gt': gt_lst,
        'transcript': tr_lst,
        'mask': mask_lst,
        'multi_hot': mh_lst,

        'raw_gt': raw_gt_lst,
        'raw_len': raw_len_lst,
    }
    return ret
