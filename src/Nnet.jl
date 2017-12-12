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

# estimate of evidence lower bound
function ℒ(y, X, prior, μ, σ)
    ϵ = rand(Normal(), length(μ))
    z = μ + ϵ .* σ

    log_prior = logpdf.(prior, z)
    log_lik = 0

    for n in 1:length(y)
        layer1 = sigmoid.([dot(z[2:3], X[n, :]), dot(z[4:5], X[n, :])] + z[1])
        layer2 = sigmoid(z[6] + dot(z[7:8], layer1))
        log_lik += logpdf(Bernoulli(layer2), y[n])
    end

    log_joint = log_prior + log_lik

    entropy = logpdf.(Normal.(μ, σ), z)

    f = log_joint - entropy

    return f
end

function fit(nn::Nnet,
             opt::GradDescent.Optimizer;
             tol=0.005, 
             maxiter=1000, 
             msg=100,
             elboiter=10)
    N = length(nn.y)
    D = 8

    mean_δ = 1.0
    i = 1

    while mean_δ > tol && i < maxiter
        nn.μ = nn.λ[:, 1]
        nn.σ = softplus.(nn.λ[:, 2])

        g = ForwardDiff.jacobian(ϕ -> ℒ(nn.y, nn.X, nn.prior, ϕ[1:D], 
                                        ϕ[(D+1):2D]), 
                                 vcat(nn.μ, nn.σ))
        ∇ℒ = hcat(diag(g[1:D, 1:D]),
                  diag(g[1:D, (D+1):2D]))

        δ = update(opt, ∇ℒ)
        nn.λ += δ
        truncate!(nn.λ, D)

        mean_δ = mean(δ .^ 2)

        if (i % msg == 0)
            println(i, " ", mean_δ)
        end
        
        i += 1
    end

    # calculate elbo
    nn.μ = nn.λ[:, 1]
    nn.σ = softplus.(nn.λ[:, 2])
    elbo = 0

    for i in 1:elboiter
        elbo += mean(ℒ(nn.y, nn.X, nn.prior, nn.μ, nn.σ))
    end

    elbo = elbo / elboiter

    return elbo
end

function predict(nn::Nnet,
                 X::Array{Float64, 2})

    out = zeros(size(X)[1])

    for n in 1:size(X)[1]
        layer1 = sigmoid.([dot(nn.μ[2:3], X[n, :]), dot(nn.μ[4:5], X[n, :])] + nn.μ[1])
        out[n] = sigmoid(nn.μ[6] + dot(nn.μ[7:8], layer1))
    end

    return out
end

