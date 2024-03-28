import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='breakfast')
parser.add_argument('--root', type=str, default='./data/')
parser.add_argument('--split', type=int)
parser.add_argument('--sample-rate', type=int, default=1, help='use for downsampling. take 1 frame per x frames')

parser.add_argument('--exp-name', type=str, help='name used to save model and logs')
parser.add_argument("--ckpt", default=None, help="ckpt for trained model")
parser.add_argument("--test", action='store_true', help="only evaluate, don't train")

parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)

parser.add_argument('--optim', type=str, default='adamw')

parser.add_argument("--lr-decay", type=str, default='cos', help='use which decay scheduler')
parser.add_argument("--decay-rate", type=float, default=0.01, help="final lr = x * initial lr")
parser.add_argument("--warmup", type=int, default=10, help="warmup epoch number")

parser.add_argument('--hidden-dim', type=int, default=256)
parser.add_argument('--n-head', type=int, default=1)
parser.add_argument('--n-encoder', type=int, default=6)

parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=0.1)

parser.add_argument('--bdy-kernel', type=int, default=7, help="wb in the paper")
parser.add_argument('--bdy-scale', type=float, default=0.3, help="mu in the paper")

parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--warm-epc', type=int, default=40, help="epoch number of first stage")

parser.add_argument('--cs-kernel', type=int, default=31, help='wa in the paper')
parser.add_argument('--candidate-mul', type=int, default=4, help="lambda in the paper")

parser.add_argument('--cts-temp', type=float, default=0.2, help="tau in the paper")

parser.add_argument('--bgw', type=float, default=1.0, help='weight for pseudo background frames')

parser.add_argument("--save", action='store_true', help='to save the evaluation dict')