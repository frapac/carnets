# Define logistic loss
using LinearAlgebra
using Optim
using Base.Threads
import Base: length
abstract type AbstractLoss end

BLAS.set_num_threads(1)
struct Logit <: AbstractLoss end

@inline function expit(t::T) where T
    return one(T) / (one(T) + exp(-t))
    #= return 0.5 + 0.5 * tanh(0.5 * t) =#
end

function log1pexp(t::T) where T
    if t < -33.3
        return t
    elseif t <= -18.0
        return t - exp(t)
    elseif t <= 37.0
        return -log1p(exp(-t))
    else
        return -exp(-t)
    end
end

function loss(::Logit, x, y)
    return -log1pexp(x*y)
end

# Define statistical problem
#
abstract type AbstractProblem end

struct LogitData{T}
    X::Array{T, 2}
    y::Vector{T}
    y_pred::Vector{T}
end
function LogitData(X::Array{T, 2}, y::Vector{T}) where T
    @assert size(X, 1) == size(y, 1)
    return LogitData(X, y, zeros(T, size(y, 1)))
end
length(dat::LogitData) = length(dat.y)
dim(dat) = size(dat.X, 2)

function ploss(ω::Vector{T}, data::LogitData{T}) where T
    # Compute dot product
    mul!(data.y_pred, data.X, ω)

    res = Atomic{Float64}(0.0)

    @threads for i in 1:length(data)
        atomic_add!(res, @inbounds loss(Logit(), data.y_pred[i], data.y[i]))
    end
    return 1.0 / length(data) * res[]
end

function diffloss(ω::Vector{T}, data::LogitData{Float64}) where T
    # Compute dot product
    y_pred = data.X * ω
    res = zero(T)
    for i in 1:length(data)
        res = @inbounds loss(Logit(), data.y_pred[i], data.y[i])
    end
    return 1.0 / length(data) * res
end

function loss(ω::Vector{T}, data::LogitData{T}) where T
    # Compute dot product
    mul!(data.y_pred, data.X, ω)
    res = zero(T)
    @inbounds for i in 1:length(data)
        res += loss(Logit(), data.y_pred[i], data.y[i])
    end
    return one(T) / length(data) * res
end

function ∇loss(grad::Vector{T}, ω::Vector{T}, data::LogitData{T}) where T
    # Compute dot product
    fill!(grad, 0.0)
    mul!(data.y_pred, data.X, ω)
    invn = -one(T) / length(data)
    @inbounds for j in 1:length(data)
        tmp = invn * data.y[j] * expit(-data.y_pred[j] * data.y[j])
        @inbounds for i in 1:length(ω)
            grad[i] += tmp * data.X[j, i]
        end
    end
end

function hessvec_loss(hessvec::Vector{T}, ω::Vector{T}, vec::Vector{T}, data::LogitData{T}) where T
    p = dim(data)
    mul!(data.y_pred, data.X, ω)

    fill!(hessvec, zero(T))
    invn = one(T) / length(data)
    @inbounds for i in 1:length(data)
        σz = invn * expit(-data.y_pred[i] * data.y[i])

        acc = zero(T)
        @inbounds for j in 1:p
            acc += data.X[i, j] * vec[j]
        end
        # Rely on BLAS here
        pσ = σz * acc * 0.5
        @inbounds for j in 1:p
            hessvec[j] += pσ * data.X[i, j]
        end
    end
end

#= T = Float64 =#
#= N_SIZE = 4_000 =#
#= A = randn(T, N_SIZE, N_SIZE) =#
#= b = sign.(randn(T, N_SIZE)) =#
#= x = randn(T, N_SIZE) =#

#= dat = LogitData(A, b) =#

#= #1= loss(x, dat) =1# =#
#= #1= @time loss(x, dat) =1# =#

#= g = zeros(T, N_SIZE) =#
#= ∇loss(g, x, dat) =#
#= @time ∇loss(g, x, dat) =#

#= hvec = zeros(T, N_SIZE) =#
#= vec = zeros(T, N_SIZE) =#
#= hessvec_loss(hvec, x, vec, dat) =#
#= @time hessvec_loss(hvec, x, vec, dat) =#
