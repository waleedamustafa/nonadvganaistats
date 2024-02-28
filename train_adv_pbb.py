import math
import argparse,os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
import os
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from architectures import ARCHITECTURES, get_architecture,CNNet4l,ProbCNNet4l,Lambda_var
from datasets import get_dataset, DATASETS
from attacks import Attacker, PGD_L2,DDN
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from train_utils import AverageMeter, accuracy, init_logfile, log,  requires_grad_
from bounds import PBBobj
from torch.optim import SGD, Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from datasets import DatasetFromSubset


parser = argparse.ArgumentParser(description='Adversarial certifiable stochastic model training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--width_multiplier', default=1, type=int, metavar='N',
                    help='The number by which the width of the network is multiplied (default: 1)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=20,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', action='store_true',
#                     help='if true, tries to resume training from existing checkpoint')
# parser.add_argument('--pretrained-model', type=str, default='',
#                     help='Path to a pretrained model')

# Adversarial certification parameters 
parser.add_argument('--input_noise_sd', default=0.5, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")


# PAC-bayes bounds parameters 
parser.add_argument('--pac_bayes_objective', default='fquad', type=str,
                    help="The PAC-Bayes bound used in training")
parser.add_argument('--initial_lambda', default=6.0, type=float,
                    help="the initial value of lambda variable in the flamb bound ")  # not used in fquad
parser.add_argument('--delta', default=0.025, type=float,
                    help="The confidence parameter in the pac-bayes bound")  # we don't need delta-test
parser.add_argument('--prior', default='rand', type=str,choices={"learnt","rand"},
                    help="Whether the prior is learned or randomly initialized")
parser.add_argument('--prior_weight_sd', default=0.03, type=float,  # maybe decrease a bit?
                    help="The prior wieght standard deviation")
parser.add_argument('--pmin', default=1e-5, type=float,
                    help="The minimum accepted probability for random network")
parser.add_argument('--kl_penalty', default=0.1, type=float,  # decrease further?
                    help="The kl penalty for the training objective")
parser.add_argument('--prior_learning_rate', default=0.001, type=float,
                    help="The learning rate with wich the prior is trained")
parser.add_argument('--prior_gamma', default=0.1, type=float,
                    help="prior learning rate is multiplied by  prior_alpha")
parser.add_argument('--prior_perc', default=0.5, type=float,
                    help="The percentage of training data with wich the prior is trained")
parser.add_argument('--prior_epochs', default=50, type=int,
                    help="The number of epochs with wich the prior is trained")
parser.add_argument('--prior_weight_decay', default=0.0001, type=float,
                    help="The weight decay with wich the prior is trained")
parser.add_argument('--prior_lr_step_size', type=int, default=10,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--prior_augmentation', action='store_true',
                    help="Whether data augmentation is used in prior")
parser.add_argument('--prior_momentum', default=0.95, type=float,
                    help="The momentum with wich the prior is trained")
parser.add_argument('--prior-dropout', default=0.2, type=float,
                    help="The dropout usind for the prior network")
#####################
# Adversarial training params
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--attack', default='PGD', type=str, choices=['DDN', 'PGD'])
parser.add_argument('--epsilon', default=1.0, type=float)
parser.add_argument('--num-steps', default=10, type=int)
parser.add_argument('--warmup', default=1, type=int, help="Number of epochs over which \
# -                    the maximum allowed perturbation increases linearly from zero to args.epsilon.")
parser.add_argument('--num-noise-vec', default=4, type=int,
                    help="number of noise vectors to use for finding adversarial examples. `m_train` in the paper.")
# parser.add_argument('--train-multi-noise', action='store_true', 
#                     help="if included, the weights of the network are optimized using all the noise samples. \
# -                       Otherwise, only one of the samples is used.")
parser.add_argument('--no-grad-attack', action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")

# PGD-specific
parser.add_argument('--random-start', default=True, type=bool)

# DDN-specific
parser.add_argument('--init-norm-DDN', default=2.0, type=float)
parser.add_argument('--gamma-DDN', default=0.05, type=float)


args = parser.parse_args()

def get_minibatches(batch, num_batches):
    X = batch[0]
    y = batch[1]

    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]


def trainNNet(net, optimizer, epoch, train_loader, adv_train,
               attacker, no_grad_attack, num_noise_vector, noise_sd, device='cuda', verbose=False):
    """Train function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training
    
     adv_train: boolean
        Whether to use adversarial training or normal training

    attacker: Attacker object
        The attack used in adversarial training

    no_grad_attack: boolean
        Wehether using no grad attacks in adversarial training or not.   

    num_noise_vector: int
            The number of noise vectors to be added each example in the trainining

    noise_sd: float
            The standard deviation of noise added to training examples

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print training metrics

    """
    # train and report training metrics
    net.train()
    total, correct, avgloss = 0.0, 0.0, 0.0
    for batch_id, batch in enumerate(tqdm(train_loader)):
        mini_batches = get_minibatches(batch, num_noise_vector)
    
        for (data,target) in mini_batches:            
            data, target = data.to(device), target.to(device)
            data = data.repeat((1, num_noise_vector, 1, 1)).view(torch.Size((data.shape[0] * num_noise_vector, *data.shape[1:])))
            noise = torch.randn_like(data, device='cuda') * noise_sd

            if adv_train:
                requires_grad_(net, False)
                net.eval()
                data = attacker.attack(net, data, target, 
                                        noise=noise, 
                                        num_noise_vectors=num_noise_vector, 
                                        no_grad=no_grad_attack
                                        )
                net.train()
                requires_grad_(net,True)

           
            target = target.unsqueeze(1).repeat(1, num_noise_vector).reshape(-1,1).squeeze()
        
            net.zero_grad()
            logits = net(data + noise)
            log_softmax = F.log_softmax(logits, dim=1)
            
            loss = F.nll_loss(log_softmax, target)
            loss.backward()
            optimizer.step()
            pred = logits.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            avgloss = avgloss + loss.detach()
    # show the average loss and KL during the epoch
    if verbose:
        print(
            f"-Epoch {epoch :.5f}, Train loss: {avgloss/batch_id :.5f}, Train err:  {1-(correct/total):.5f}")

def trainPNNet(net, optimizer, pbobj, epoch, train_loader,adv_train,
               attacker,no_grad_attack,num_noise_vector,noise_sd, lambda_var=None, 
                                optimizer_lambda=None, verbose=False):
    """Train function for a probabilistic NN (including CNN)

    Parameters
    ----------
    net : ProbNNet/ProbCNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    pbobj : pbobj object
        PAC-Bayes inspired training objective to use for training

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training

    adv_train: boolean
        Whether to use adversarial training or normal training

    attacker: Attacker object
        The attack used in adversarial training

    no_grad_attack: boolean
        Wehether using no grad attacks in adversarial training or not.   

    num_noise_vector: int
            The number of noise vectors to be added each example in the trainining

    noise_sd: float
            The standard deviation of noise added to training examples

    lambda_var : Lambda_var object
        Lambda variable for training objective flamb

    optimizer_lambda : optim object
        Optimizer to use for the learning the lambda_variable

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print test metrics

    """
    net.train()
    # variables that keep information about the results of optimising the bound
    avgerr, avgbound, avgkl, avgloss = 0.0, 0.0, 0.0, 0.0
    avgerr = AverageMeter()
    avgbound = AverageMeter()
    avgkl = AverageMeter()
    avgloss = AverageMeter()

    if pbobj.objective == 'flamb':
        lambda_var.train()
        # variables that keep information about the results of optimising lambda (only for flamb)
        avgerr_l, avgbound_l, avgkl_l, avgloss_l = 0.0, 0.0, 0.0, 0.0

    if pbobj.objective == 'bbb':
        clamping = False
    else:
        clamping = True

    for batch_id, batch in enumerate(tqdm(train_loader)):
        mini_batches = get_minibatches(batch, num_noise_vector)
    
        for (data,target) in mini_batches:            
            data, target = data.to(pbobj.device), target.to(pbobj.device)
            data = data.repeat((1, num_noise_vector, 1, 1)).view(torch.Size((data.shape[0] * num_noise_vector, *data.shape[1:])))
            noise = torch.randn_like(data, device='cuda') * noise_sd
            net.sample()
            if adv_train:
                # requires_grad_(net, False)
                
                net.eval()
                
                data = attacker.attack(net, data, target, 
                                        noise=noise, 
                                        num_noise_vectors=num_noise_vector, 
                                        no_grad=no_grad_attack
                                        )
                net.train()
                # requires_grad_(net,True)

            net.zero_grad()
            target = target.unsqueeze(1).repeat(1, num_noise_vector).reshape(-1,1).squeeze()
            bound, kl, _, loss, err = pbobj.train_obj(
                net, data +noise, target, lambda_var=lambda_var, clamping=clamping)

            bound.backward()
            optimizer.step()
            avgbound.update( bound.item(),data.shape[0])
            avgkl.update(kl)
            avgloss.update(loss.item(),data.shape[0])
            avgerr.update( err,data.shape[0])

            if pbobj.objective == 'flamb':
                # for flamb we also need to optimise the lambda variable
                lambda_var.zero_grad()
                bound_l, kl_l, _, loss_l, err_l = pbobj.train_obj(
                    net, data, target, lambda_var=lambda_var, clamping=clamping)
                bound_l.backward()
                optimizer_lambda.step()
                avgbound_l += bound_l.item()
                avgkl_l += kl_l
                avgloss_l += loss_l.item()
                avgerr_l += err_l

    if verbose:
        # show the average of the metrics during the epoch
        print(
            f"-Batch average epoch {epoch :.0f} results, Train obj: {avgbound.avg :.5f}, KL/n: {avgkl.val :.5f}, NLL loss: {avgloss.avg :.5f}, Train 0-1 Error:  {avgerr.avg :.5f}")
        if pbobj.objective == 'flamb':
            print(
                f"-After optimising lambda: Train obj: {avgbound_l/batch_id :.5f}, KL/n: {avgkl_l/batch_id :.5f}, NLL loss: {avgloss_l/batch_id :.5f}, Train 0-1 Error:  {avgerr_l/batch_id :.5f}, last lambda value: {lambda_var.lamb_scaled.item() :.5f}")



def computeRiskCertificates(net, toolarge, pbobj, device='cuda', lambda_var=None, 
                            train_loader=None, whole_train=None):
    """Function to compute risk certificates and other statistics at the end of training

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    toolarge: bool
        Whether the dataset is too large to fit in memory (computation done in batches otherwise)

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    lambda_var : Lambda_var object
        Lambda variable for training objective flamb

    train_loader: DataLoader object
        Data loader for computing the risk certificate (multiple batches, used if toolarge=True)

    whole_train: DataLoader object
        Data loader for computing the risk certificate (one unique batch, used if toolarge=False)

    """
    net.eval()
    with torch.no_grad():
        if toolarge:
            train_obj, kl, loss_ce_train, err_01_train, risk_ce, risk_01 = pbobj.compute_final_stats_risk(
                net, lambda_var=lambda_var, clamping=True, data_loader=train_loader)
        else:
            # a bit hacky, we load the whole dataset to compute the bound
            for data, target in whole_train:
                data, target = data.to(device), target.to(device)
                train_obj, kl, loss_ce_train, err_01_train, risk_ce, risk_01 = pbobj.compute_final_stats_risk(
                    net, lambda_var=lambda_var, clamping=True, input=data, target=target)

    return train_obj, risk_ce, risk_01, kl, loss_ce_train, err_01_train

def test_training_step():
    train_dataset = get_dataset('mnist', 'train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)
    train_loader_cert = DataLoader(train_dataset,shuffle=False,batch_size=512)

    model = CNNet4l(0.5,n_classes = 10)
    pmodel = ProbCNNet4l(np.log(np.exp(0.03)-1),init_net=model,device='cuda').to('cuda')

    bound = PBBobj('fquad', 1e-5, 10, 0.25,
                    0.25, 100, 0.1, torch.device('cuda'), n_posterior = 50000, n_bound=50000)
    optimizer = optim.SGD(pmodel.parameters(), lr=0.01, momentum=0.95)
    attacker = PGD_L2(steps=5, device='cuda', max_norm=2.0)
    for i in range(1):
        trainPNNet(pmodel,optimizer,bound,i,train_loader,False,attacker,False,2,0.5,verbose=True)
    print(computeRiskCertificates(pmodel,True,bound,train_loader=train_loader_cert))

def main():
    
    torch.manual_seed(7)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # copy_code(args.outdir)
    args_file = open(args.outdir+'/args', "w")
    for arg in vars(args):
        args_file.write( str(arg) + ' '+ str(getattr(args, arg))+ '\n')
    args_file.close()

    csv_file = open(args.outdir+'/xls', "w")
    csv_file.write(os.path.basename(str(getattr(args, 'outdir')))+ '\n')
    for arg in vars(args):
        if str(arg) in ("dataset", "outdir", "workers", "gpu", "print_freq"):
            continue
        csv_file.write(str(getattr(args, arg)) + ',')
    csv_file.close()
    

    train_dataset = get_dataset(args.dataset, 'train')
    model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    
    device = 'cuda'


    if args.attack == 'PGD':
        print('Attacker is PGD')
        attacker = PGD_L2(steps=args.num_steps, device='cuda', max_norm=args.epsilon)
    elif args.attack == 'DDN':
        print('Attacker is DDN')
        attacker = DDN(steps=args.num_steps, device='cuda', max_norm=args.epsilon, 
                    init_norm=args.init_norm_DDN, gamma=args.gamma_DDN)
    else:
        raise Exception('Unknown attack')
    
    if args.prior == 'learnt':
        print('learnt prior')
        prior_length = int(np.floor(len(train_dataset)* args.prior_perc))
        train_length = len(train_dataset) - prior_length
        train_dataset, prior_train_dataset = torch.utils.data.random_split(train_dataset,[train_length,prior_length])
        
        if args.prior_augmentation and args.dataset == 'cifar10':
            prior_train_dataset = DatasetFromSubset(prior_train_dataset,transform=transforms.Compose([
            # transforms.RandomCrop(32, padding=3),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15)
        ]))
            prior_train_loader = DataLoader(prior_train_dataset, shuffle=True, batch_size=args.batch,
                             num_workers=args.workers)
        else:
            prior_train_loader = DataLoader(prior_train_dataset, shuffle=True, batch_size=args.batch,
                             num_workers=args.workers)
        net = get_architecture(args.arch, dropout=args.prior_dropout,width_multiplier=args.width_multiplier)
        prior_epochs= args.prior_epochs
        prior_weight_decay= args.prior_weight_decay
        optimizer = SGD(net.parameters(), lr=args.prior_learning_rate, 
                        momentum=args.prior_momentum,weight_decay=prior_weight_decay)
        scheduler = StepLR(optimizer, step_size=args.prior_lr_step_size, gamma=args.prior_gamma)
        
        for i in range(prior_epochs):
           attacker.max_norm = np.min([args.epsilon, (i + 1) * args.epsilon/args.warmup])
           trainNNet(net,optimizer,i,prior_train_loader,
                     args.adv_training,attacker,args.no_grad_attack,
                     args.num_noise_vec,args.input_noise_sd,verbose=True)
           scheduler.step()
           
        
    else:
        net = None
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                             num_workers=args.workers)
        
    objective = args.pac_bayes_objective
   
    rho_prior = math.log(math.exp(args.prior_weight_sd)-1.0)
    pnet = get_architecture(args.arch,True,net,rho_prior=rho_prior,width_multiplier=args.width_multiplier)
    classes = 10
    mc_samples = 100
    posterior_n_size = len(train_loader.dataset)
    bound_n_size = len(train_loader.dataset)
    bound = PBBobj(objective, args.pmin, classes, args.delta,
                    args.delta, mc_samples, args.kl_penalty, device,
                                     n_posterior = posterior_n_size, n_bound=bound_n_size)

    if objective == 'flamb':
        lambda_var = Lambda_var(args.initial_lamb, posterior_n_size).to(device)
        optimizer_lambda = optim.SGD(lambda_var.parameters(), 
                                     lr=args.lr, momentum=args.momentum)
    else:
        optimizer_lambda = None
        lambda_var = None

    optimizer = optim.SGD(pnet.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
 
    for epoch in range(args.epochs):
        attacker.max_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon/args.warmup])
        trainPNNet(pnet, optimizer, bound, epoch, train_loader,args.adv_training,attacker,
                   args.no_grad_attack, args.num_noise_vec, args.input_noise_sd,
                                    lambda_var, optimizer_lambda, True)
        scheduler.step()
     
        if epoch % 10 ==0:
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': pnet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path )
    
    torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': pnet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path)

        

if __name__ =='__main__':
    main()