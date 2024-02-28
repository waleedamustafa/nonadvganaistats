
# evaluate a smoothed classifier on a dataset
# this file is based on code publicly available at
#    https://github.com/mperezortiz/PBB/blob/master/pbb/models.py
# written by María Pérez Orti
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np 

ARCHITECTURES = ['simple_mnist_2cls','simple_mnist','ProbCNNet13l','simple_cifar','ProbCNNet13lv2','ProbCNNet13lv3','ProbCNNet13lv21']


class CNNet4l(nn.Module):
    """Implementation of a standard Convolutional Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    n_classes : int
        Number of classes 

    """

    def __init__(self, dropout_prob,n_classes=2,width_multiplier=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32*width_multiplier, 3, 1)
        self.conv2 = nn.Conv2d(32*width_multiplier, 64*width_multiplier, 3, 1)
        self.fc1 = nn.Linear(9216*width_multiplier, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.d = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        x = self.d(self.conv1(x))
        x = F.relu(x)
        x = self.d(self.conv2(x))
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.d(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1) # remmember to check this out once more
        return x


class CNNet4l_cifar(nn.Module):
    """Implementation of a standard Convolutional Neural Network with 4 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    n_classes : int
        Number of classes 

    """

    def __init__(self, dropout_prob,n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.d = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        x = self.d(self.conv1(x))
        x = F.relu(x)
        x = self.d(self.conv2(x))
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.d(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1) # remmember to check this out once more
        return x

def get_architecture(model_name,is_prob=False,init_net=None,rho_prior=0.03,dropout=0.2,width_multiplier=1):
    if is_prob:
        if model_name == 'simple_mnist':
            return ProbCNNet4l(rho_prior=rho_prior,init_net=init_net,width_multiplier=width_multiplier)
        elif model_name == 'simple_cifar':
            return ProbCNNet4l_cifar(rho_prior=rho_prior,init_net=init_net)
        elif model_name == 'ProbCNNet13l':
            return ProbCNNet13l(rho_prior=rho_prior,init_net=init_net)
        elif model_name == 'ProbCNNet13lv2':
            return ProbCNNet13lv2(rho_prior=rho_prior,init_net=init_net)
        elif model_name == 'ProbCNNet13lv3':
            return ProbCNNet13lv3(rho_prior=rho_prior,init_net=init_net)
        elif model_name == 'ProbCNNet13lv21':
            return ProbCNNet13lv21(rho_prior=rho_prior,init_net=init_net)
        else:
            raise Exception('Unknown architecture')
    else:
        if model_name == 'simple_mnist_2cls':
            return CNNet4l(dropout,2).cuda()
        elif model_name == 'simple_mnist':
            return CNNet4l(dropout,10,width_multiplier).cuda()
        elif model_name == 'simple_cifar':
            return CNNet4l_cifar(dropout,10).cuda()
        elif model_name =='ProbCNNet13l':
            return CNNet13l(dropout).cuda()
        elif model_name == 'ProbCNNet13lv2':
            return CNNet13lv2(dropout).cuda()
        elif model_name == 'ProbCNNet13lv3':
            return CNNet13lv3(dropout).cuda()
        elif model_name == 'ProbCNNet13lv21':
            return CNNet13lv21(dropout).cuda()
        else:
            raise Exception('Unknown architecture')

    
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used works best if :math:`\text{mean}` is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
    
    
   
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1. - eps), max=(1. - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class Gaussian(nn.Module):
    """Implementation of a Gaussian random variable, using softplus for
    the standard deviation and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Gaussian.

    rho : Tensor of floats
        Scale parameter of the Gaussian (to be transformed to std
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the Gaussian is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu.to(device), requires_grad=not fixed)
        self.rho = nn.Parameter(rho.to(device), requires_grad=not fixed)
        self.device = device

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))
        

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(
            torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div


class Laplace(nn.Module):
    """Implementation of a Laplace random variable, using softplus for
    the scale parameter and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Laplace distr.

    rho : Tensor of floats
        Scale parameter for the distribution (to be transformed
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the distribution is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def scale(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Laplace distribution
        # we do scaling due to numerical issues
        epsilon = (0.999*torch.rand(self.scale.size())-0.49999).to(self.device)
        result = self.mu - torch.mul(torch.mul(self.scale, torch.sign(epsilon)),
                                     torch.log(1-2*torch.abs(epsilon)))
        return result

    def compute_kl(self, other):
        # Compute KL divergence between two Laplaces distr. (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = self.scale
        b0 = other.scale
        term1 = torch.log(torch.div(b0, b1))
        aux = torch.abs(self.mu - other.mu)
        term2 = torch.div(aux, b0)
        term3 = torch.div(b1, b0) * torch.exp(torch.div(-aux, b1))

        kl_div = (term1 + term2 + term3 - 1).sum()
        return kl_div
    
class ProbLinear(nn.Module):
    """Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_layer : Linear object
        Linear layer object used to initialise the prior

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(self, in_features, out_features, rho_prior, prior_dist='gaussian', device='cuda', init_prior='weights', init_layer=None, init_layer_prior=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            # Initialise distribution means using truncated normal
            weights_mu_init = trunc_normal_(torch.Tensor(
                out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_features)

        weights_rho_init = torch.ones(out_features, in_features) * rho_prior
        bias_rho_init = torch.ones(out_features) * rho_prior

        if init_prior == 'zeros':
            bias_mu_prior = torch.zeros(out_features) 
            weights_mu_prior = torch.zeros(out_features, in_features)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(
                out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_prior = torch.zeros(out_features) 
        elif init_prior == 'weights': 
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                # otherwise initialise to posterior weights
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
        else: 
            raise RuntimeError(f'Wrong type of prior initialisation!')

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), device=device, fixed=False)
        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0
        self.c_weight = None
        self.c_bias = None
        self.sample()

    def sample(self):
        self.c_weight = self.weight.sample()
        
        # self.c_weight.detach()
        self.c_bias = self.bias.sample()
        
        # self.c_bias.detach()

    def posterior(self):
        self.c_weight = self.weight.mu
        self.c_bias = self.bias.mu

    def forward(self, input):
        # if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            # self.sample()
        weight = self.c_weight
        bias = self.c_bias
        # else:
        #     # otherwise we use the posterior mean
        #     weight = self.weight.mu
        #     bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                self.bias.compute_kl(self.bias_prior)

        return F.linear(input, weight, bias)


class ProbConv2d(nn.Module):
    """Implementation of a Probabilistic Convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the layer

    out_channels : int
        Number of output channels for the layer

    kernel_size : int
        size of the convolutional kernel

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    stride : int
        Stride of the convolution

    padding: int
        Zero-padding added to both sides of the input

    dilation: int
        Spacing between kernel elements

    init_layer : Linear object
        Linear layer object used to initialise the prior

    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_prior, prior_dist='gaussian',
                 device='cuda', stride=1, padding=0, dilation=1, init_prior='weights', init_layer=None, init_layer_prior=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        # Compute and set sigma for the truncated gaussian of weights
        in_features = self.in_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = 1/np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            weights_mu_init = trunc_normal_(torch.Tensor(
                out_channels, in_channels, *self.kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_channels)

        # set scale parameters
        weights_rho_init = torch.ones(
            out_channels, in_channels, *self.kernel_size) * rho_prior
        bias_rho_init = torch.ones(out_channels) * rho_prior

        if init_prior == 'zeros':
            bias_mu_prior = torch.zeros(out_features) 
            weights_mu_prior = torch.zeros(out_features, in_features)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(
                out_channels, in_channels, *self.kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_prior = torch.zeros(out_features) 
        elif init_prior == 'weights': 
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                # otherwise initialise to posterior weights
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
                weights_mu_prior = weights_mu_prior
                bias_mu_prior = bias_mu_prior
        else: 
            raise RuntimeError(f'Wrong type of prior initialisation!')

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), device=device, fixed=False)
        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0
        # The current weights and biasses after last sample 
        self.c_weight = None
        self.c_bias = None
        self.sample()

    def sample(self):
        self.c_weight = self.weight.sample()
        
        # self.c_weight.detach()
        self.c_bias = self.bias.sample()
        
        # self.c_bias.detach()

    def posterior(self):
        self.c_weight = self.weight.mu
        self.c_bias = self.bias.mu
    

    def forward(self, input):
        # if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            # self.sample()
        weight = self.c_weight
        bias = self.c_bias
            
        # else:
        #     # otherwise we use the posterior mean
        #     weight = self.weight.mu
        #     bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior) + self.bias.compute_kl(self.bias_prior)

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
    
class ProbCNNet4l(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size,
    number of classes and kernel size).

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : CNNet object
        Network object used to initialise the prior

    """

    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None,width_multiplier=1):
        super().__init__()

        self.conv1 = ProbConv2d(
            1, 32*width_multiplier, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(
            32*width_multiplier, 64*width_multiplier, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv2 if init_net else None)
        self.fc1 = ProbLinear(9216*width_multiplier, 128, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc1 if init_net else None)
        self.fc2 = ProbLinear(128, 10, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc2 if init_net else None)
    def sample(self):
        self.conv1.sample()
        self.conv2.sample()
        self.fc1.sample()
        self.fc2.sample()
    def posterior(self):
        self.conv1.posterior()
        self.conv2.posterior()
        self.fc1.posterior()
        self.fc2.posterior()

    def forward(self, x, clamping=True, pmin=1e-4):
        # forward pass for the network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = output_transform(self.fc2(x), clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.fc1.kl_div + self.fc2.kl_div

class ProbCNNet4l_cifar(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 4 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes and kernel size).

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : CNNet object
        Network object used to initialise the prior

    """

    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        super().__init__()

        self.conv1 = ProbConv2d(
            3, 32, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(
            32, 64, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv2 if init_net else None)
        self.fc1 = ProbLinear(12544, 128, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc1 if init_net else None)
        self.fc2 = ProbLinear(128, 10, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc2 if init_net else None)
    def sample(self):
        self.conv1.sample()
        self.conv2.sample()
        self.fc1.sample()
        self.fc2.sample()
    def posterior(self):
        self.conv1.posterior()
        self.conv2.posterior()
        self.fc1.posterior()
        self.fc2.posterior()

    def forward(self, x, clamping=True, pmin=1e-4):
        # forward pass for the network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = output_transform(self.fc2(x), clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.fc1.kl_div + self.fc2.kl_div

class CNNet13l(nn.Module):
    """Implementation of a Convolutional Neural Network with 13 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).
    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.
    """

    def __init__(self, dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.fcl1 = nn.Linear(2048, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, 10)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        # conv layers
        x = F.relu(self.d(self.conv1(x)))
        x = F.relu(self.d(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv3(x)))
        x = F.relu(self.d(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv5(x)))
        x = F.relu(self.d(self.conv6(x)))
        x = F.relu(self.d(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv8(x)))
        x = F.relu(self.d(self.conv9(x)))
        x = F.relu(self.d(self.conv10(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        # x = F.log_softmax(x, dim=1)
        return x


class CNNet13lv2(nn.Module):
    """Implementation of a Convolutional Neural Network with 13 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).
    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.
    """

    def __init__(self, dropout_prob):
        super().__init__()
        self.norm = NormalizeLayer(_CIFAR10_MEAN,_CIFAR10_STDDEV)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, affine=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128, affine=False)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256, affine=False)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256, affine=False)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256, affine=False)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512, affine=False)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512, affine=False)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512, affine=False)
        self.fcl1 = nn.Linear(2048, 1024)
        self.bn11 = nn.BatchNorm1d(1024, affine=False)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, 10)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        # conv layers
        x = self.norm(x)
        x = self.bn1(F.relu(self.d(self.conv1(x))))
        x = self.bn2(F.relu(self.d(self.conv2(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn3(F.relu(self.d(self.conv3(x))))
        x = self.bn4(F.relu(self.d(self.conv4(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn5(F.relu(self.d(self.conv5(x))))
        x = self.bn6(F.relu(self.d(self.conv6(x))))
        x = self.bn7(F.relu(self.d(self.conv7(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn8(F.relu(self.d(self.conv8(x))))
        x = self.bn9(F.relu(self.d(self.conv9(x))))
        x = self.bn10(F.relu(self.d(self.conv10(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.bn11(F.relu(self.d(self.fcl1(x))))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        return x


class CNNet13lv21(CNNet13lv2):
    def __init__(self, dropout_prob):
        super().__init__(dropout_prob)
        self.norm = NormalizeLayer([0, 0, 0], [1, 1, 1])



class CNNet13lv3(nn.Module):
    """Implementation of a Convolutional Neural Network with 13 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).
    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.
    """

    def __init__(self, dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, affine=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128, affine=False)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256, affine=False)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256, affine=False)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256, affine=False)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512, affine=False)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512, affine=False)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512, affine=False)
        self.fcl1 = nn.Linear(2048, 1024)
        self.bn11 = nn.BatchNorm1d(1024, affine=False)
        self.fcl2 = nn.Linear(1024, 512)
        self.bn12 = nn.BatchNorm1d(512, affine=False)
        self.fcl3 = nn.Linear(512, 10)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        # conv layers
        x = self.bn1(F.relu(self.d(self.conv1(x))))
        x = self.bn2(F.relu(self.d(self.conv2(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn3(F.relu(self.d(self.conv3(x))))
        x = self.bn4(F.relu(self.d(self.conv4(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn5(F.relu(self.d(self.conv5(x))))
        x = self.bn6(F.relu(self.d(self.conv6(x))))
        x = self.bn7(F.relu(self.d(self.conv7(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn8(F.relu(self.d(self.conv8(x))))
        x = self.bn9(F.relu(self.d(self.conv9(x))))
        x = self.bn10(F.relu(self.d(self.conv10(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.bn11(F.relu(self.d(self.fcl1(x))))
        x = self.bn12(F.relu(self.d(self.fcl2(x))))
        x = self.fcl3(x)
        return x

class ProbCNNet13l(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 13 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).
    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cuda')
    init_net : CNNet object
        Network object used to initialise the prior
    """

    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        super().__init__()
        self.norm = NormalizeLayer(_CIFAR10_MEAN,_CIFAR10_STDDEV)
        self.conv1 = ProbConv2d(in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None)
        self.conv3 = ProbConv2d(in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None)
        self.conv4 = ProbConv2d(in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None)
        self.conv5 = ProbConv2d(in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None)
        self.conv6 = ProbConv2d(in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None)
        self.conv7 = ProbConv2d(in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv7 if init_net else None)
        self.conv8 = ProbConv2d(in_channels=256, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv8 if init_net else None)
        self.conv9 = ProbConv2d(in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv9 if init_net else None)
        self.conv10 = ProbConv2d(in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                 device=device, kernel_size=3, padding=1, init_layer=init_net.conv10 if init_net else None)
        self.fc1 = ProbLinear(2048, 1024, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fc2 = ProbLinear(1024, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fc3 = ProbLinear(512, 10, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl3 if init_net else None)
    def sample(self):
        self.conv1.sample()
        self.conv2.sample()
        self.conv3.sample()
        self.conv4.sample()
        self.conv5.sample()
        self.conv6.sample()
        self.conv7.sample()
        self.conv8.sample()
        self.conv9.sample()
        self.conv10.sample()
        self.fc1.sample()
        self.fc2.sample()
        self.fc3.sample()
    def posterior(self):
        self.conv1.posterior()
        self.conv2.posterior()
        self.conv3.posterior()
        self.conv4.posterior()
        self.conv5.posterior()
        self.conv6.posterior()
        self.conv7.posterior()
        self.conv8.posterior()
        self.conv9.posterior()
        self.conv10.posterior()
        self.fc1.posterior()
        self.fc2.posterior()
        self.fc3.posterior()

    def forward(self, x, clamping=True, pmin=1e-4):
        # conv layers
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.conv7.kl_div + self.conv8.kl_div + self.conv9.kl_div + self.conv10.kl_div + self.fc1.kl_div + self.fc2.kl_div + self.fc3.kl_div


class ProbCNNet13lv2(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 13 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).
    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cuda')
    init_net : CNNet object
        Network object used to initialise the prior
    """

    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        super().__init__()
        self.norm = NormalizeLayer(_CIFAR10_MEAN,_CIFAR10_STDDEV)
        self.conv1 = ProbConv2d(in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None)
        self.bn1 = nn.BatchNorm2d(32, affine=False).to(device)
        
        self.conv2 = ProbConv2d(in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None)
        self.bn2 = nn.BatchNorm2d(64, affine=False).to(device)
        self.conv3 = ProbConv2d(in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None)
        self.bn3 = nn.BatchNorm2d(128, affine=False).to(device)
        self.conv4 = ProbConv2d(in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None)
        self.bn4 = nn.BatchNorm2d(128, affine=False).to(device)
        self.conv5 = ProbConv2d(in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None)
        self.bn5 = nn.BatchNorm2d(256, affine=False).to(device)
        self.conv6 = ProbConv2d(in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None)
        self.bn6 = nn.BatchNorm2d(256, affine=False).to(device)
        self.conv7 = ProbConv2d(in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv7 if init_net else None)
        self.bn7 = nn.BatchNorm2d(256, affine=False).to(device)
        self.conv8 = ProbConv2d(in_channels=256, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv8 if init_net else None)
        self.bn8 = nn.BatchNorm2d(512, affine=False).to(device)
        self.conv9 = ProbConv2d(in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv9 if init_net else None)
        self.bn9 = nn.BatchNorm2d(512, affine=False).to(device)
        self.conv10 = ProbConv2d(in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                 device=device, kernel_size=3, padding=1, init_layer=init_net.conv10 if init_net else None)
        self.bn10 = nn.BatchNorm2d(512, affine=False).to(device)
        self.fc1 = ProbLinear(2048, 1024, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl1 if init_net else None)
        self.bn11 = nn.BatchNorm1d(1024, affine=False).to(device)
        self.fc2 = ProbLinear(1024, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fc3 = ProbLinear(512, 10, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl3 if init_net else None)

        if init_net is not None:
            for bn, bn_init in zip([
                    self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6, self.bn7, self.bn8, self.bn9, self.bn10, self.bn11
                ], [
                    init_net.bn1, init_net.bn2, init_net.bn3, init_net.bn4, init_net.bn5, init_net.bn6, init_net.bn7, init_net.bn8, 
                    init_net.bn9, init_net.bn10, init_net.bn11
                ]):
                bn.load_state_dict(bn_init.state_dict())


    def sample(self):
        self.conv1.sample()
        self.conv2.sample()
        self.conv3.sample()
        self.conv4.sample()
        self.conv5.sample()
        self.conv6.sample()
        self.conv7.sample()
        self.conv8.sample()
        self.conv9.sample()
        self.conv10.sample()
        self.fc1.sample()
        self.fc2.sample()
        self.fc3.sample()
    def posterior(self):
        self.conv1.posterior()
        self.conv2.posterior()
        self.conv3.posterior()
        self.conv4.posterior()
        self.conv5.posterior()
        self.conv6.posterior()
        self.conv7.posterior()
        self.conv8.posterior()
        self.conv9.posterior()
        self.conv10.posterior()
        self.fc1.posterior()
        self.fc2.posterior()
        self.fc3.posterior()

    def forward(self, x, clamping=True, pmin=1e-4):
        # conv layers
        # It is important that the batchnorm layers do not update their weights or running statistics here
        self.bn1.eval()
        self.bn2.eval()
        self.bn3.eval()
        self.bn4.eval()
        self.bn5.eval()
        self.bn6.eval()
        self.bn7.eval()
        self.bn8.eval()
        self.bn9.eval()
        self.bn10.eval()
        self.bn11.eval()
    
        x = self.norm(x)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.bn7(F.relu(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn8(F.relu(self.conv8(x)))
        x = self.bn9(F.relu(self.conv9(x)))
        x = self.bn10(F.relu(self.conv10(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.bn11(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.conv7.kl_div + self.conv8.kl_div + self.conv9.kl_div + self.conv10.kl_div + self.fc1.kl_div + self.fc2.kl_div + self.fc3.kl_div


class ProbCNNet13lv21(ProbCNNet13lv2):
    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        super().__init__(rho_prior, prior_dist, device, init_net)
        self.norm = NormalizeLayer([0, 0, 0], [1, 1, 1])
        
        

class ProbCNNet13lv3(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 13 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).
    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cuda')
    init_net : CNNet object
        Network object used to initialise the prior
    """

    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        super().__init__()
        self.norm = NormalizeLayer(_CIFAR10_MEAN,_CIFAR10_STDDEV)
        self.conv1 = ProbConv2d(in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None)
        self.bn1 = nn.BatchNorm2d(32, affine=False).to(device)
        self.conv2 = ProbConv2d(in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None)
        self.bn2 = nn.BatchNorm2d(64, affine=False).to(device)
        self.conv3 = ProbConv2d(in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None)
        self.bn3 = nn.BatchNorm2d(128, affine=False).to(device)
        self.conv4 = ProbConv2d(in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None)
        self.bn4 = nn.BatchNorm2d(128, affine=False).to(device)
        self.conv5 = ProbConv2d(in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None)
        self.bn5 = nn.BatchNorm2d(256, affine=False).to(device)
        self.conv6 = ProbConv2d(in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None)
        self.bn6 = nn.BatchNorm2d(256, affine=False).to(device)
        self.conv7 = ProbConv2d(in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv7 if init_net else None)
        self.bn7 = nn.BatchNorm2d(256, affine=False).to(device)
        self.conv8 = ProbConv2d(in_channels=256, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv8 if init_net else None)
        self.bn8 = nn.BatchNorm2d(512, affine=False).to(device)
        self.conv9 = ProbConv2d(in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv9 if init_net else None)
        self.bn9 = nn.BatchNorm2d(512, affine=False).to(device)
        self.conv10 = ProbConv2d(in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                 device=device, kernel_size=3, padding=1, init_layer=init_net.conv10 if init_net else None)
        self.bn10 = nn.BatchNorm2d(512, affine=False).to(device)
        self.fc1 = ProbLinear(2048, 1024, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl1 if init_net else None)
        self.bn11 = nn.BatchNorm1d(1024, affine=False).to(device)
        self.fc2 = ProbLinear(1024, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl2 if init_net else None)
        self.bn12 = nn.BatchNorm1d(512, affine=False).to(device)
        self.fc3 = ProbLinear(512, 10, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl3 if init_net else None)

        if init_net is not None:
            for bn, bn_init in zip([
                    self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6, self.bn7, self.bn8, self.bn9, self.bn10, self.bn11,
                    self.bn12
                ], [
                    init_net.bn1, init_net.bn2, init_net.bn3, init_net.bn4, init_net.bn5, init_net.bn6, init_net.bn7, init_net.bn8, 
                    init_net.bn9, init_net.bn10, init_net.bn11, init_net.bn12
                ]):
                bn.load_state_dict(bn_init.state_dict())


    def sample(self):
        self.conv1.sample()
        self.conv2.sample()
        self.conv3.sample()
        self.conv4.sample()
        self.conv5.sample()
        self.conv6.sample()
        self.conv7.sample()
        self.conv8.sample()
        self.conv9.sample()
        self.conv10.sample()
        self.fc1.sample()
        self.fc2.sample()
        self.fc3.sample()
    def posterior(self):
        self.conv1.posterior()
        self.conv2.posterior()
        self.conv3.posterior()
        self.conv4.posterior()
        self.conv5.posterior()
        self.conv6.posterior()
        self.conv7.posterior()
        self.conv8.posterior()
        self.conv9.posterior()
        self.conv10.posterior()
        self.fc1.posterior()
        self.fc2.posterior()
        self.fc3.posterior()

    def forward(self, x, clamping=True, pmin=1e-4):
        # conv layers
        x = self.norm(x)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.bn7(F.relu(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn8(F.relu(self.conv8(x)))
        x = self.bn9(F.relu(self.conv9(x)))
        x = self.bn10(F.relu(self.conv10(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.bn11(F.relu(self.fc1(x)))
        x = self.bn12(F.relu(self.fc2(x)))
        x = self.fc3(x)
        # x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.conv7.kl_div + self.conv8.kl_div + self.conv9.kl_div + self.conv10.kl_div + self.fc1.kl_div + self.fc2.kl_div + self.fc3.kl_div

def output_transform(x, clamping=True, pmin=1e-4):
    """Computes the log softmax and clamps the values using the
    min probability given by pmin.

    Parameters
    ----------
    x : tensor
        output of the network

    clamping : bool
        whether to clamp the output probabilities

    pmin : float
        threshold of probabilities to clamp.
    """
    # lower bound output prob
    output = F.log_softmax(x, dim=1)
    if clamping:
        output = torch.clamp(output, np.log(pmin))
    return output
class Lambda_var(nn.Module):
    """Class for the lambda variable included in the objective
    flambda

    Parameters
    ----------
    lamb : float
        initial value

    n : int
        Scaling parameter (lamb_scaled is between 1/sqrt(n) and 1)

    """

    def __init__(self, lamb, n):
        super().__init__()
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=True)
        self.min = 1/np.sqrt(n)

    @property
    def lamb_scaled(self):
        # We restrict lamb_scaled to be between 1/sqrt(n) and 1.
        m = nn.Sigmoid()
        return (m(self.lamb) * (1-self.min) + self.min)
    
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds