mutable struct Nnet
    y::Array{Int64,1}
    X::Array{Float64,2}
    N::Int64
    D::Int64
    hidden_layers::Int64
    nodes::Int64
    weights::Int64
    λ::Array{Float64,2}
    μ::Array{Float64,1}
    σ::Array{Float64,1}
    prior::Distributions.Normal{Float64}
    iter::Int64
end

"Construct Neural Network"
function Nnet(y::Array{Int64, 1},
              X::Array{Float64, 2};
              hidden_layers::Int64=1,
              nodes::Int64=2,
              prior::Distributions.Normal{Float64}=Normal())

    N = length(y)
    D = size(X)[2]

    # still need to account for more than one hidden layer
    weights = D * nodes + # layer 1
              hidden_layers + 1 + # biases
              nodes # final layer

    # initialize variational parameters
    λ = rand(Normal(), weights, 2)
    truncate!(λ, weights)

    μ = λ[:, 1]
    σ = softplus.(λ[:, 2])

    Nnet(y, X, N, D, 
         hidden_layers, nodes, weights,
         λ, μ, σ, prior, 0)
end

function fit(nn::Nnet,
             opt::GradDescent.Optimizer;
             tol=0.005, 
             maxiter=1000, 
             elboiter=10)
    return 0
end

function predict(nn::Nnet,
                 X::Array{Float64, 2})
    return 0
end


