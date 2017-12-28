# BayesNN

[![Build Status](https://travis-ci.org/jacobcvt12/BayesNN.jl.svg?branch=master)](https://travis-ci.org/jacobcvt12/BayesNN.jl)

Bayesian Neural Networks fit with Variational Inference

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
