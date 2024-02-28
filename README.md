# Non-vacuous Generalization Bounds for Adversarial Risk in Stochastic Neural Networks
Here we provide the implementation for the paper *Non-vacuous generalization bounds for stochastic neural networks in adversarial environments*

## Table of contents
* [Installation](#installation)
* [Train](#train)
* [Compute Adversarial Certificates](#compute_certificate)
* [Examples](#examples)

## Installation
We recommended to use a virtual environment to install the code:

    virtualenv -p python3 nvagb/data/venv 
    source nvagb/data/venv/bin/activate
    pip install -r nvagb/requirements.txt

The CIFAR-10 and MNIST datasets are automatically downloaded once requested.
    
## Train
To train the model we call ``train_adv_pbb.py``.  E.g., 

    python train_adv_pbb.py DATASET ARCHITECTURE OUTDIR_0

See 

    python train_adv_pbb.py --help

for more information.

## Compute_Certificate
To compute the adversarial risk certificate of a model we call ``compute_losses.py``, followed by ``collect_and_certify.py``:

    python compute_losses.py DATASET OUTDIR_0/checkpoint.pth.tar OUTDIR_1 
    python collect_and_certify.py DATASET OUTDIR_1 OUTDIR_0/checkpoint.pth.tar OUTDIR_2

Again, use ``--help`` for more information.

To visualize the results we call ``plot_final_certificate.py``:

    python plot_final_certificate.py OUTDIR_2/final_cert.npz 
    
 
## Examples
For a full example for MNIST, \
Train the model:

    python train_adv_pbb.py mnist simple_mnist results/my_mnist_experiment --epochs 50 --lr 1e-3 --lr_step_size 20 --gamma 0.5 --momentum 0.9 --input_noise_sd 0.5 --prior learnt --prior_weight_sd 0.03 --kl_penalty 0.2 --prior_learning_rate 5e-3 --prior_perc 0.5 --prior_epochs 50 --prior_weight_decay 0.0001 --prior_lr_step_size 20 --prior_gamma 0.5 --adv-train --epsilon 1.0 --num-steps 10 --warmup 10 --num-noise-vec 4 

Compute the certificate for the first 10000 examples:    

    python compute_losses.py mnist results/my_mnist_experiment/checkpoint.pth.tar results/my_mnist_experiment/losses --start_index 0 --end_index 10000  --input_noise_sd 0.5 --prior_perc 0.5  --mc_samples 300 
    python collect_and_certify.py mnist results/my_mnist_experiment/losses results/my_mnist_experiment/checkpoint.pth.tar results/my_mnist_experiment/certificate --mc_samples 300 
    python plot_final_certificate.py results/my_mnist_experiment/certificate/final_cert.npz 



For CIFAR-10, \
Train the model:

    python train_adv_pbb.py cifar10 ProbCNNet13lv2 results/my_cifar_experiment --epochs 100 --lr 1e-3 --lr_step_size 60 --gamma 0.1 --momentum 0.9 --prior learnt --prior_weight_sd 0.01 --kl_penalty 0.1 --prior_learning_rate 5e-3 --prior_epochs 100 --prior_weight_decay 0 --prior_augmentation --prior_lr_step_size 60 --prior_gamma 0.1 --adv-training --epsilon 1.0 --num-steps 10 --warmup 10 --num-noise-vec 4 --input_noise_sd 0.5 --prior_perc 0.7 

Compute the certificate for the first 10000 examples:    

    python compute_losses.py cifar10 results/my_cifar_experiment/checkpoint.pth.tar results/my_cifar_experiment/losses --start_index 0 --end_index 10000  --input_noise_sd 0.5 --prior_perc 0.7  --mc_samples 300 
    python collect_and_certify.py cifar10 results/my_cifar_experiment/losses results/my_cifar_experiment/checkpoint.pth.tar results/my_cifar_experiment/certificate --mc_samples 300 
    python plot_final_certificate.py results/my_cifar_experiment/certificate/final_cert.npz 

