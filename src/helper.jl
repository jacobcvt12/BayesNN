function truncate!(λ, D)
    # truncate variational standard deviation
    for d in 1:D
        λ[d, 2] = λ[d, 2] < invsoftplus(1e-5) ? 
                  invsoftplus(1e-5) : λ[d, 2]
        λ[d, 2] = λ[d, 2] > softplus(log(prevfloat(Inf))) ? 
                  invsoftplus(log(prevfloat(Inf))) : λ[d, 2]
    end
end

function sigmoid(z)
    return 1.0 / (1.0 + exp(-z))
end
