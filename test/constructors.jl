@testset "Constructors" begin
    using Distributions, BayesNN

    N = 1000  # number of observations
    D = 3     # number of covariates including intercept

    # generate data
    srand(1)                 
    X = rand(Normal(), N, D)  
    y = rand(Bernoulli(0.5), N)  

    @test_nowarn net = Nnet(y, X)
end
