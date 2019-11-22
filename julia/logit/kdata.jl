using KNITRO, Random

include("data.jl")

distmat = (X,Z) -> pairwise(Euclidean(), X', Z', dims=2)
kappa = (X,Z,sigma) -> exp.( -distmat(X,Z)/(2*sigma^2) );

n = 1000; p = 2
t = 2*pi*rand(div(n,2),1)
R = 2.5
r = R*(1 .+ .2*rand(div(n,2),1)); # radius
X1 = [cos.(t).*r sin.(t).*r]
X = [randn(div(n,2),2); X1]
y = [ones(div(n,2),1);-ones(div(n,2),1)];

function gausspred(G, X, h; sigma=1)
    K1 = kappa(G, X, sigma)
    return theta( K1 * h ) #Prediction on test set
end
