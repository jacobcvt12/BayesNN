function fwd_prop(X::Array{Float64,1},
                  nodes::Int64,
                  hidden_layer::Int64,
                  z,
                  layer,
                  layer_p1)
    # number of used weights
    j = 0

    # number of covariates
    D = length(X)

    # build first hidden layer
    for node in 1:nodes
        i = j + 1
        j = i + D
        layer[node] = sigmoid(dot(z[i:(j-1)], X) + z[j])
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
    out = sigmoid(dot(z[i:(j-1)], layer) + z[j])

    return out
end
