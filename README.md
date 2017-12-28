# BayesNN

[![Build Status](https://travis-ci.org/jacobcvt12/BayesNN.jl.svg?branch=master)](https://travis-ci.org/jacobcvt12/BayesNN.jl)
[![codecov](https://codecov.io/gh/jacobcvt12/BayesNN.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jacobcvt12/BayesNN.jl)

Bayesian Neural Networks fit with Variational Inference

# Description

`BayesNN` is a Julia package for fitting Bayesian Neural Networks. Currently, models are limited to classification problems only and are fit using the "reparameterization method" as described in Kingma, D. P. & Welling, M. (2013). Auto-Encoding Variational Bayes.. CoRR, abs/1312.6114. 

At the present, a prior is provided for the Neural Network weights and the posterior is approximated using a Gaussian variational density with the independence assumption. In the future, more flexible variational densities will be added and regression capabilities will be implemented.

# Example

```julia
# load packages
using BayesNN
using Distributions
using StatsFuns
using GradDescent

# data size
N = 1000  # number of observations
D = 3     # number of covariates including intercept

# generate data
srand(1)                  # insure reproducibility
X = rand(Normal(), N, D)  # generate covariates
X[:, 1] = 1               # make first column intercept
b = rand(Normal(), D)     # coefficients
θ = logistic.(X * b)      # simulate dependent variables
y = rand.(Bernoulli.(θ))  # simulate dependent variables

# drop the intercept
X = X[:, 2:3]

# construct Neural Network object
nn = Nnet(y, X, hidden_layers=1, nodes=3)

# fit neural network with variational inference
BayesNN.fit(nn, Adam(α=1.0))

# get predictecd fit and compare to true prob.
θ_star = predict(nn, X)
hcat(θ, θ_star)
```
