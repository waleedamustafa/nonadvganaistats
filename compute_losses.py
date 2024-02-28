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

parser.add_argument('--alpha', default=0.001, type=float,
                    help=" ")
parser.add_argument('--N', default=10000, type=int,
                    help="The number of noisy evaluation of the model that CERTIFY uses")
parser.add_argument('--N0', default=100, type=int,
                    help="The initial number of noisey evaluation of the model that CERTIFY uses")

parser.add_argument('--batch', default=10000, type=int,
                    help="The batch size for the noisey evaluation of the model that CERTIFY uses")
parser.add_argument('--mc_samples', default=300, type=int,
                    help="The batch size for the noisey evaluation of the model that CERTIFY uses")
parser.add_argument('--start_index', default=0,type =int,help="The start index of the evaluation set")
parser.add_argument('--end_index', default=50000,type =int,help="The end index of the evaluation set")

args = parser.parse_args()


torch.manual_seed(7)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

prior = args.prior == 'learnt'
dataset = args.dataset
workers = 1
sigma = args.input_noise_sd
alpha = args.alpha
N = args.N
N0 = args.N0
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



chkpnt_pth = args.model_path
out_dir = args.output_dir
os.makedirs(out_dir, exist_ok=True)
checkpoint = torch.load(chkpnt_pth)
base_classifier = get_architecture(checkpoint["arch"],is_prob=True)
base_classifier.load_state_dict(checkpoint["state_dict"])
# base_classifier.train()
#recomputing the kl-div
# base_classifier(torch.stack([train_dataset[0][0], train_dataset[1][0]]).cuda())  # dummy pass for kl div
base_classifier.eval()
smoothed_classifier = Smooth(base_classifier, 10, sigma)

# kl = base_classifier.compute_kl()


start = time.time()


 

radii = np.arange(0,2,0.1)
f_name = out_dir+'certout_'+str(mc_sample)+'_'+str(sigma)+'_radii'
np.savez_compressed(f_name,a=radii)
indecies = []
correct_counts = []
with tqdm(range(start_index,end_index), ncols=80) as t:
    for i in t:
        x = train_dataset[i][0]
        y = train_dataset[i][1]
        corr =  np.zeros_like(radii,dtype=np.int64)
        for _ in range(mc_sample):        
            smoothed_classifier.base_classifier.sample()    
            prediction, radius = smoothed_classifier.certify(x.cuda(), N0, N, alpha, batch)
        
            if prediction == y:
                corr = corr + (radii <= radius)
        correct_counts.append(corr[None,:])
        indecies.append(i)
        if (i+1) %100 == 0:
            f_name = out_dir+'certout_'+str(mc_sample)+'_'+str(sigma)+'_'+str(start_index)+'_'+str(end_index)
            cert_out = np.concatenate((np.array(indecies)[:,None],np.concatenate(correct_counts,axis=0)),axis=1)
            np.savez_compressed(f_name,a=cert_out)
            print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), f'{str(t)}', i, flush=True)

f_name = out_dir+'certout_'+str(mc_sample)+'_'+str(sigma)+'_'+str(start_index)+'_'+str(end_index)
cert_out = np.concatenate((np.array(indecies)[:,None],np.concatenate(correct_counts,axis=0)),axis=1)
np.savez_compressed(f_name,a=cert_out)
print(i)





