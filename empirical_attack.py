import math
import argparse,os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from architectures import ARCHITECTURES, get_architecture,CNNet4l,ProbCNNet4l,Lambda_var
from datasets import get_dataset, DATASETS
from attacks import Attacker, PGD_L2,DDN
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from train_utils import AverageMeter, accuracy, init_logfile, log,  requires_grad_
from bounds import PBBobj
from core import Smooth
from datetime import datetime
import time


parser = argparse.ArgumentParser(description='Compute empiricall loss on a (subset of) evaluation data')

parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('model_path', type=str, help='folder to evaluated model)')

parser.add_argument('output_dir', type=str, help='folder to save the output)')


parser.add_argument('--input_noise_sd', default=0.5, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")


# PAC-bayes bounds parameters 
parser.add_argument('--prior', default='learnt', type=str,choices={"learnt","rand"},
                    help="Whether the prior is learned or randomly initialized")
parser.add_argument('--prior_perc', default=0.5, type=float,
                    help="The percentage of training data with wich the prior is trained")

parser.add_argument('--batch', default=1000, type=int,
                    help="The batch size for the noisey evaluation of the model that CERTIFY uses")
parser.add_argument('--mc_samples', default=100, type=int,
                    help="The batch size for the noisey evaluation of the model that CERTIFY uses")
parser.add_argument('--start_index', default=0,type =int,help="The start index of the evaluation set")
parser.add_argument('--end_index', default=5000,type =int,help="The end index of the evaluation set")

args = parser.parse_args()


torch.manual_seed(7)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

prior = args.prior == 'learnt'
dataset = args.dataset
workers = 1
sigma = args.input_noise_sd

batch = args.batch
mc_sample = args.mc_samples

start_index = args.start_index
end_index = args.end_index
train_dataset = get_dataset(dataset, 'train')

if prior:
    print('learnt prior')     
    prior_length = int(np.floor(len(train_dataset)* args.prior_perc))
    train_length = len(train_dataset) - prior_length
    train_dataset, prior_train_dataset = torch.utils.data.random_split(train_dataset,[train_length,prior_length])
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=4)        



chkpnt_pth = args.model_path
out_dir = args.output_dir
os.makedirs(out_dir, exist_ok=True)
checkpoint = torch.load(chkpnt_pth)
base_classifier = get_architecture(checkpoint["arch"],is_prob=True)
base_classifier.load_state_dict(checkpoint["state_dict"])
requires_grad_(base_classifier, False)
base_classifier.eval()


# kl = base_classifier.compute_kl()


start = time.time()

attacker = PGD_L2(steps=100, device='cuda', max_norm=0.5)
 

radii = np.arange(0,1.65,0.2)
f_name = out_dir+'certout_'+str(mc_sample)+'_'+str(sigma)+'_radii'
np.savez_compressed(f_name,a=radii)
indecies = []
correct_counts = []
num_batches = int((end_index - start_index)//batch)
cnt = 0
for (x,y) in tqdm(train_loader):
    cnt= cnt +1
    x = x.cuda()
    y = y.cuda()
    errors =  np.zeros_like(radii,dtype=np.int64)
    for _ in range(mc_sample):   
        base_classifier.sample()
        b_errors = []
        for eps in radii:
            attacker.max_norm = eps  
            x_adv = attacker.attack(base_classifier,x,y)
            predictions = base_classifier(x_adv).argmax(1)
            err = (predictions != y).sum()
            b_errors.append(err.item())
        b_errors = np.array(b_errors)
        errors  = errors + b_errors

    if (cnt) %10 == 1:
        f_name = out_dir+'empirical_attack_'+str(mc_sample)+'_'+str(sigma)+'_'+str(start_index)+'_'+str(end_index)
        errors_rate = errors/(mc_sample* cnt*batch)
        np.savez_compressed(f_name,a=errors_rate,b=radii)
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush=True)
    if cnt >= num_batches:
        break

f_name = out_dir+'empirical_attack_'+str(mc_sample)+'_'+str(sigma)+'_'+str(start_index)+'_'+str(end_index)
errors_rate = errors/(mc_sample* num_batches*batch)
np.savez_compressed(f_name,a=errors_rate,b=radii)





