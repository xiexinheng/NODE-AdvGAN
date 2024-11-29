from tqdm import tqdm
import requests
import zipfile
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from utility import init_params, test_target_model_performance, load_target_model
from advGAN import AdvGAN_Attack
import wandb
import torch
import logging
import os, time, glob
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='Configure the model training settings.')

parser.add_argument('--target', default='FMNIST', help='target_dataset')
parser.add_argument('--num_channel', default=1, help='')
parser.add_argument('--batch_size', default=200, help='')
parser.add_argument('--epochs', default=150, help='AdvGAN_epochs')
parser.add_argument('--lr', default=0.002, help='AdvGAN_learning_rate')
parser.add_argument('--lr_halve', default=True, help='')
parser.add_argument('--lr_h_rate', default=0.5, help='')
parser.add_argument('--lr_h_n_steps', default=60, help='')

parser.add_argument('--l_inf_bound', default=15 / 255, type=float, help='')
parser.add_argument('--l_inf_bound_train', default=10 / 255, type=float, help='')
parser.add_argument('--alpha', default=0.01, type=float, help='')
parser.add_argument('--beta', default=0.01, type=float, help='')
parser.add_argument('--gamma', default=1,type=float, help='')
parser.add_argument('--kappa', default=0,type=float, help='')
parser.add_argument('--c', default=0.1, type=float,help='')
parser.add_argument('--n_steps_D', default=1, help='D_number_of_steps_per_batch')
parser.add_argument('--n_steps_G', default=1, help='G_number_of_steps_per_batch')
parser.add_argument('--target_model', default='vgg16_bn', help='')

parser.add_argument('--use_cuda', default=True, help='')
parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help='')
parser.add_argument('--use_wandb', default=True, help='')
parser.add_argument('--save_dir', default= '', help='')
parser.add_argument('--save_test_images', default= False, help='')
parser.add_argument('--save_test_npy', default= False, help='')
parser.add_argument('--training_clamp', default= True, help='')
parser.add_argument('--test_transferability', default=True , help='')
parser.add_argument('--transfer_model_names', default=['vgg16_bn', 'vgg19_bn', 'resnet34', 'resnet18',
                                                       'densenet121', 'densenet169'] , help='')
parser.add_argument('--seed', default=201812, help='')

parser.add_argument('--odeint_adjoint', default=False, help='')
parser.add_argument('--ODE_vector_field', default='_VectorField', help='')
parser.add_argument('--G_model', default='NODE_AdvGAN', help='')
parser.add_argument('--N_t', default=5, type=int, help='')
parser.add_argument('--t', default=0.05, type=float, help='')
parser.add_argument('--solver', default='euler', help='')
parser.add_argument('--is_initialize', default=True, help='')
parser.add_argument('--test_transferability_per_times', default=20, help='')

parser.add_argument('--add_transform', default=True, help='')
parser.add_argument('--is_targeted_attack', default=False, type=bool, help='')
parser.add_argument('--target_label', default=1, type=int, help='')

args = parser.parse_args()
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if using multi-GPU
np.random.seed(args.seed)
random.seed(args.seed)

save_dir = f'./snapshot/save_{args.target}_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    # logD
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y %H:%M:%S',
                        filename=save_dir + 'info.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info(args)
args.save_dir = save_dir

print(args)
wandb.init(project="NODE-AdvGAN", entity="ywugroup")

wandb.log({"save_dir": save_dir})

device = torch.device('cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu')

print('\nPREPARING DATASETS...')
train_dataloader, test_dataloader, n_labels, n_channels, test_set_size = init_params(args)

print('\nLOADING AND TESTING TARGET MODEL...')
target_model = load_target_model(args.target_model, args.target).to(device)
target_model.eval()
test_target_model_performance(args, test_dataloader, target_model, test_set_size)

# train AdvGAN
print('\nTRAINING ADVGAN...')
advGAN = AdvGAN_Attack(
                        device,
                        target_model,
                        n_labels,
                        n_channels,
                        args
                    )
advGAN.train(train_dataloader,test_dataloader)

wandb.finish()