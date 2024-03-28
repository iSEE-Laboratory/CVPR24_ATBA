# [CVPR2024] Efficient and Effective Weakly-Supervised Action Segmentation via Action-Transition-Aware Boundary Alignment  

This is the official PyTorch implementation for CVPR2024 paper *Efficient and Effective Weakly-Supervised Action Segmentation via Action-Transition-Aware Boundary Alignment*.



## Environments

- A single GTX1080Ti
- Python 3.9.12
- PyTorch 1.11.0+cu113



## Data

The datasets can be download in [Link](https://drive.google.com/drive/folders/1bOvo2g05gI0jArgN_vznxSg_ns32NDbd?usp=sharing). Please create a ``./data`` folder and put them in. Note that this link does not include the features. Please download the features from the following links and put them into the ``features`` subfolder of each dataset:

- Breakfast: We use the features of MS-TCN. [Link](https://github.com/yabufarha/ms-tcn).
- Hollywood: The features are extracted by us. [Link](https://drive.google.com/drive/folders/1bOvo2g05gI0jArgN_vznxSg_ns32NDbd?usp=sharing).
- CrossTask: We use the features of POC. [Link](https://github.com/ZijiaLewisLu/CVPR22-POC).



## Running

Training commands for three datasets. Please fill in or select the args enclosed by {} first.

- Breakfast

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --split {1-4} --sample-rate 10 --seed 0 --epoch 400 --cs-kernel 31 --exp-name {custom experiment name}
```

- Hollywood

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --dataset hollywood --split {1-10} --sample-rate 5 --seed 0 --epoch 300 --cs-kernel 23 --bgw 0.8 --exp-name {custom experiment name}
```

- CrossTask

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --dataset crosstask --split 1 --sample-rate 1 --seed 0 --epoch 300 --cs-kernel 31 --bdy-scale 0.1 --bgw 0.8 --exp-name {custom experiment name}
```

The running log is automatically saved to the ``./logs`` folder (TensorBoard file ). The final checkpoint is automatically saved to the ``./ckpt`` folder.

**Running config:** You can also access all hyper-parameters and options in ``options.py``, and change them freely in the running command.

**Only Testing:** Adding the command options ``--test --ckpt {name of checkpoint}``.



# Citation

```
@inproceedings{xu2024efficient,
title = {Efficient and Effective Weakly-Supervised Action Segmentation via Action-Transition-Aware Boundary Alignment},
author = {Xu, Angchi and Zheng, Wei-Shi},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2024}
}
```

