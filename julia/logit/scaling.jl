using Statistics

abstract type AbstractScaler end


struct NormalScaler <: AbstractScaler end

function scale!(::NormalScaler, X::Array{T, 2}) where T
    n, d = size(X)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)

    @inbounds for i in 1:d
        X[:, i] .= (X[:, i] .- μ[i]) ./ σ[i]
    end
end

struct MinMaxScaler <: AbstractScaler end

function scale(::MinMaxScaler, X::Array{T, 2}) where T

end

