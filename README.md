# Unsupervised Domain Adaptation in PyTorch

This repository contains popular Unsupervised Domain Adaptation (UDA) frameworks implemented with PyTorch. This repository will be frequently updated.

## Requirements
Python3.6

## Installation
```pip install -r requirements.txt```

# Models implemented
- [DANN](#dann)
--------------------
# [DANN](http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf)
**Key idea**: Domain Adversarial Training makes the features domain agnostic by using a domain confusion loss. It comprises of three
networks: feature extractor, label predictor and domain predictor. The label predictor minimizes the standard supervised loss
on the source domain. The domain predictor maximizes the domain confusion on both the source and target domains. 
This is implemented through a **Gradient Reversal Layer** (GRL) that performs gradient ascent on the feature extractor with the loss obtained from the domain predictor. 
Its called adversarial training because this resembles a vanilla GAN training.
| Model | SVHN->MNIST|
---------|:-----------:|
| Paper Implementation | 73.85|
| This Implementation | 76.13|
