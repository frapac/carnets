using KNITRO, Random
using LinearAlgebra
using PyPlot
Random.seed!(27)

include("data.jl")
include("utils.jl")

BLAS.set_num_threads(1)


const Xp = AddBias(X)
const α = 1e-3

function callbackEvalF(kc, cb, evalRequest, evalResult, userParams)
    x = evalRequest.x
    dats = userParams[:data]
    evalResult.obj[1] = loss(x, dats)
    #= evalResult.obj[1] = L(Xp * x, y) =#
    return 0
end

σ(z) = 1. / (1. + exp(-z))

function init_z!(z, Xp, ω, n, p)
    for i in 1:n
        z[i] = 0.
        for j in 1:p
            @inbounds z[i] += ω[j] * Xp[i, j]
        end
    end
end

function eval_g!(grad, Xp, y, ω, z)
    n = 10000
    p = 3
    invn = -1. / n
    init_z!(z, Xp, ω, n, p)
    for i in 1:p
        for j in 1:n
            @inbounds grad[i] += invn * y[j] * Xp[j, i] * σ( -z[j] * y[j])
        end
    end
    return
end

function eval_h!(hess, ω, Xp, y)
    n = 10000
    p = 3

    z = zeros(Float64, n)
    init_z!(z, Xp, ω, n, p)

    for i in 1:n
        @inbounds xi = Xp[i, :]
        pxi = xi * xi'
        @inbounds psig = σ(z[i] * y[i]) * (1. - σ(z[i] * y[i]))
        count = 0
        for j in 1:p, k in j:p
            count += 1
            @inbounds hess[count] += pxi[j, k] * psig
        end
    end
    hess ./= n
    return
end

function eval_hessvec!(hessvec, ω, Xp, y)
    n = 10000
    p = 3

    z = zeros(Float64, n)
    init_z!(z, Xp, ω, n, p)

    for i in 1:n
        @inbounds xi = Xp[i, :]
        pxi = xi * xi'
        @inbounds psig = σ(z[i] * y[i]) * (1. - σ(z[i] * y[i]))
        count = 0
        for j in 1:p, k in j:p
            count += 1
            @inbounds hess[count] += pxi[j, k] * psig
        end
    end
    hess ./= n
    return
end

function callbackEvalG!(kc, cb, evalRequest, evalResult, userParams)
    ω = evalRequest.x
    z = zeros(Float64, n)
    dat = userParams[:data]
    ∇loss(evalResult.objGrad, ω, dat)
    #= eval_g!(evalResult.objGrad, Xp, y, ω, z) =#
    #= copyto!(evalResult.objGrad, (nablaE(ω, Xp, y))) =#
    return 0
end

function callbackEvalH!(kc, cb, evalRequest, evalResult, userParams)
    h = evalRequest.x
    eval_h!(evalResult.hess, h, Xp, y)
    #= z = y .* Xp * h =#
    #= # d = np.diag(y * y * expit(z) * expit(-z)) =#
    #= D = diagm(0 => vec(1/(length(y)) .* y .* y .* theta(z) .* theta(-z))) =#
    #= #1= D2 = diagm(0 => 2 * α * ones(length(h))) =1# =#
    #= H = Xp' * D * Xp =#
    #= count = 0 =#
    #= for i in 1:size(Xp, 2) =#
    #=     for j in i:size(Xp, 2) =#
    #=         count += 1 =#
    #=         @inbounds evalResult.hess[count] = H[i, j] =#
    #=     end =#
    #= end =#
    return 0
end

dat = LogitData(Xp, vec(y))

kc = KNITRO.KN_new()
KNITRO.KN_add_vars(kc, p+1)
KNITRO.KN_set_var_primal_init_values(kc, zeros(p+1))
cb = KNITRO.KN_add_objective_callback(kc, callbackEvalF)
KNITRO.KN_set_cb_grad(kc, cb, callbackEvalG!)
#= KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callbackEvalH!) =#

KNITRO.KN_set_param(kc, "outlev", 3)
KNITRO.KN_set_param(kc, "algorithm", 2)
KNITRO.KN_set_param(kc, "derivcheck_type", 2)

KNITRO.KN_set_cb_user_params(kc, cb, dat)

#= KNITRO.KN_set_param(kc, KNITRO.KN_PARAM_DERIVCHECK, KNITRO.KN_DERIVCHECK_ALL) =#
nStatus = KNITRO.KN_solve(kc)
w_opt = KNITRO.get_solution(kc)
KNITRO.KN_free(kc)


q = 201;
tx = range(minimum(X[:,1]),stop=maximum(X[:,1]),length=q)
ty = range(minimum(X[:,2]),stop=maximum(X[:,2]),length=q)
#= B,A = meshgrid( ty,tx ) =#
#= G = [A[:] B[:]]; =#
#= Theta = theta(AddBias(G)*w); =#
#= Theta = reshape(Theta, q, q); =#

#= imshow(Theta'[:,end:-1:1], extent=[minimum(tx), maximum(tx), minimum(ty), maximum(ty)]); =#
#
scatter(X[:, 1], X[:, 2])

tmin = minimum(X[:, 1])
tmax = maximum(X[:, 2])
v = range(tmin, stop=tmax, length=2)

v1, v2, v3 = w_opt
plot(v, -v1 / v2 * v .- v3 / v2)
ylim(-5, 5)
;
