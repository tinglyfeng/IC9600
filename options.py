import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--warm', type=int, default=1, help='num of epoch for warming up')
parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay of optimizer')
parser.add_argument('--lr_decay_rate',type = float,default=0.2, help='lr decay of optimizer')
parser.add_argument('--milestone',type = list,default=[10,20], help = 'perform lr decay in each milestone')


parser.add_argument('--batch_size',type=int, default=64, help='batch size for dataloader')
parser.add_argument('--num_workers',type=int,default=8, help = 'num of worker for dataloader')
parser.add_argument('--epoch',type=int,default=30, help='total epoch of training')
parser.add_argument('--image_size',type = int ,default= 512, help = 'input size of model')
parser.add_argument('--gpu_id',type = int,default=0, help = 'gpu id')

parser.add_argument('--ck_save_dir',type = str, default='cks', help ='directory to save checkpoints')

args = parser.parse_args()








