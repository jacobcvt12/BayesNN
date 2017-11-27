"""
*Bayesian Neural Networks for Julia.*

# Introduction

Formulates and fits Bayesian Neural Network with Variational Inference.

# Examples

```julia
using Distributions, BayesNN, StatsFuns

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

net = Nnet(y, X)
```

"""
module BayesNN

using StatsFuns
using Distributions
using GradDescent

export 
    Nnet,
    fit,
    predict

include("helper.jl")
include("Nnet.jl")

end # module
