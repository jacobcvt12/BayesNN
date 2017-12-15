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
    weights = D * nodes + nodes + # input to layer 1
              # layer n to layer n + 1
              (hidden_layers - 1) * ((nodes + 1) * nodes)  +
              nodes + 1 # final layer to f(X)

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
function ℒ(nn::Nnet, μ, σ)
    # extract elements from Nnet
    y = nn.y
    X = nn.X
    prior = nn.prior
    hidden_layer = nn.hidden_layers
    nodes = nn.nodes

    # draw values from q
    ϵ = rand(Normal(), length(μ))
    z = μ + ϵ .* σ

    # initialize vectors for calculating ELBO
    # note that the type is important for AD
    log_lik = zeros(eltype(μ), 1)
    log_prior = zeros(eltype(μ), length(μ))
    entropy = zeros(eltype(μ), length(μ))
    f = zeros(eltype(μ), length(μ))

    # get number of covariates
    N, D = size(X)

    # store layer calculations
    layer = zeros(eltype(μ), nodes)
    layer_p1 = zeros(eltype(μ), nodes)
    out = zeros(eltype(μ), N)

    for n in 1:N
        # number of used weights
        j = 0

        # build first hidden layer
        for node in 1:nodes
            i = j + 1
            j = i + D
            layer[node] = sigmoid(dot(z[i:(j-1)], X[n, :]) + z[j])
        end

        # build other hidden layers
        if hidden_layer > 1
            for l in 2:hidden_layer
                for node in 1:nodes
                    i = j + 1
                    j = i + nodes
                    layer_p1[node] = sigmoid(dot(z[i:(j-1)], layer) + z[j])
                end

                layer = copy(layer_p1)
            end
        end

        # build output
        i = j + 1
        j = i + nodes
        out[n] = sigmoid(dot(z[i:(j-1)], layer) + z[j])
        log_lik += logpdf(Bernoulli(out[n]), y[n])
    end

    # construct ELBO estimate
    for i in 1:length(μ)
        # first calculate log prior and entropy
        log_prior[i] = logpdf(nn.prior, z[i])
        entropy[i] = logpdf(Normal(μ[i], σ[i]), z[i])
        f[i] = log_lik[1] + log_prior[i] - entropy[i]
    end

    return f
end

function fit(nn::Nnet,
             opt::GradDescent.Optimizer;
             tol=0.005, 
             maxiter=1000, 
             msg=100,
             elboiter=10)
    N = length(nn.y)
    D = nn.weights

    mean_δ = 1.0
    i = 1

    while mean_δ > tol && i < maxiter
        nn.μ = nn.λ[:, 1]
        nn.σ = softplus.(nn.λ[:, 2])

        g = ForwardDiff.jacobian(ϕ -> ℒ(nn, ϕ[1:D], ϕ[(D+1):2D]), 
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
        elbo += mean(ℒ(nn, nn.μ, nn.σ))
    end

    elbo = elbo / elboiter

    return elbo
end

function predict(nn::Nnet,
                 X::Array{Float64, 2})
    # extract elements from Nnet
    hidden_layer = nn.hidden_layers
    nodes = nn.nodes

    # get number of covariates
    N, D = size(X)

    # store layer calculations
    layer = zeros(nodes)
    layer_p1 = zeros(nodes)
    out = zeros(N)

    # for right now, fix z at mean
    z = nn.μ

    for n in 1:N
        # number of used weights
        j = 0

        # build first hidden layer
        for node in 1:nodes
            i = j + 1
            j = i + D
            layer[node] = sigmoid(dot(z[i:(j-1)], X[n, :]) + z[j])
        end

        # build other hidden layers
        if hidden_layer > 1
            for l in 2:hidden_layer
                for node in 1:nodes
                    i = j + 1
                    j = i + nodes
                    layer_p1[node] = sigmoid(dot(z[i:(j-1)], layer) + z[j])
                end

                layer = copy(layer_p1)
            end
        end

        # build output
        i = j + 1
        j = i + nodes
        out[n] = sigmoid(dot(z[i:(j-1)], layer) + z[j])
    end

    return out
end

