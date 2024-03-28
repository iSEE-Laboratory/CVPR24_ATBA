'''
code copied from https://github.com/ZijiaLewisLu/ICCV21-TASL/blob/main/src/inference.py
'''


import numpy as np


def compute_mof(gt_list, pred_list):
    match, total = 0, 0
    for gt_label, recognized in zip(gt_list, pred_list):
        correct = recognized == gt_label

        match += correct.sum()
        total += len(gt_label)

    mof = match / total
    return mof


def compute_IoU_IoD(gt_list, pred_list):
    IOU, IOU_NB = [], []
    IOD, IOD_NB = [], []
    for ground_truth, recognized in zip(gt_list, pred_list):

        unique = list(set(np.unique(ground_truth)))  # .union(set(np.unique(recognized)))

        video_iou = []
        video_iod = []
        for i in unique:
            recog_mask = recognized == i
            gt_mask = ground_truth == i
            union = np.logical_or(recog_mask, gt_mask).sum()
            intersect = np.logical_and(recog_mask, gt_mask).sum()  # num of correct prediction
            num_recog = recog_mask.sum()

            video_iou.append(intersect / (union + 1e-6))
            video_iod.append(intersect / (num_recog + 1e-6))

        IOU.append(np.mean(video_iou))
        IOD.append(np.mean(video_iod))

        video_iou_noBG = [v for (a, v) in zip(unique, video_iou) if a != 0]
        IOU_NB.append(np.mean(video_iou_noBG))

        video_iod_noBG = [v for (a, v) in zip(unique, video_iod) if a != 0]
        IOD_NB.append(np.mean(video_iod_noBG))

    return np.mean(IOU), np.mean(IOU_NB), np.mean(IOD), np.mean(IOD_NB)