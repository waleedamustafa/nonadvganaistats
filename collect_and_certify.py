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
from train_utils import AverageMeter, accuracy, init_logfile, log, requires_grad_
from bounds import PBBobj
from core import Smooth
import time
from datasets import DATASETS

parser = argparse.ArgumentParser(description='Collect emperical losses and compute certificates')

parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('empirical_losses_folder', type=str,help="The folder in which the emperical losses are saved")
parser.add_argument('model_path', type=str, help='path of the evaluated model')

parser.add_argument('output_dir', type=str, help='folder to save the output)')




# PAC-bayes bounds parameters 
parser.add_argument('--width_multiplier', default=1, type=int, metavar='N',
                    help='The number by which the width of the network is multiplied (default: 1)')

parser.add_argument('--alpha', default=0.001, type=float,
                    help="The probability of certify giving wrong certificate")

parser.add_argument('--raw', default=0.01, type=float,
                    help="The confidence of the emperical certificates")

parser.add_argument('--mc_samples', default=300, type=int,
                    help="The batch size for the noisey evaluation of the model that CERTIFY uses")


args = parser.parse_args()


torch.manual_seed(7)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



alpha = args.alpha
raw = args.raw
mc_sample = args.mc_samples

chkpnt_pth = args.model_path
out_dir = args.output_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
checkpoint = torch.load(chkpnt_pth)
base_classifier = get_architecture(checkpoint["arch"],is_prob=True,width_multiplier=args.width_multiplier)
base_classifier.load_state_dict(checkpoint["state_dict"])
base_classifier.train().cuda()
#recomputing the kl-div
dummy_input = torch.rand(2,1,28,28) if args.dataset == 'mnist' else torch.rand(2,3,32,32)
base_classifier(dummy_input.cuda())

kl = base_classifier.compute_kl()
emp_loss_folder = args.empirical_losses_folder 
empirical_losses_files = []
radii_file = ''
for path in os.listdir(emp_loss_folder):
    if path.endswith('radii.npz'):
        radii_file = path
    elif path.endswith('cert.npz'):
        continue
    elif path.endswith('.npz'):
        empirical_losses_files.append(path)
loss_counts = []

radii = np.load(emp_loss_folder+'/'+radii_file)['a']
for path in empirical_losses_files:
    counts = np.load(emp_loss_folder+'/'+path)['a']
    loss_counts.append(counts)

loss_counts = np.concatenate(loss_counts,axis=0)[:,1:]

num_examples = loss_counts.shape[0]
bound = PBBobj(objective='fquad',mc_samples=mc_sample,n_bound=num_examples,delta_test=0.01)

loss_counts = loss_counts.sum(axis = 0)
loss_counts = num_examples * mc_sample - loss_counts # we invert the numbers as compute loss computes the correct counts
certs = []
emp_loss = []

for loss_count in loss_counts:
    emp_loss_bound = bound.compute_empirical_cert(loss_count,num_examples*mc_sample,alpha,raw)
    emp_loss.append(emp_loss_bound)   
    cert = bound.compute_01_risk(emp_loss_bound,kl)   
    certs.append(cert)
certs = np.array(certs)
emp_loss = np.array(emp_loss)
f_name = out_dir + '/final_cert'
np.savez_compressed(f_name,a=certs,b=radii,c=emp_loss)