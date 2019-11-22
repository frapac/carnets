using Mosek, MosekTools
using JuMP
using MathOptInterface
const MOI = MathOptInterface
const MOIB = MathOptInterface.Bridges

const MOSEK = JuMP.direct_model(MOIB.full_bridge_optimizer(Mosek.Optimizer(), Float64))
const COSMOP = JuMP.direct_model(MOIB.full_bridge_optimizer(COSMO.Optimizer(), Float64))
#= MOI.set(backend(model), MOI.RawParameter("MSK_IPAR_PRESOLVE_USE"), 0) =#

function mskfit(X, y, model=MOSEK)
    n, d = size(X)
    #= model = Model(with_optimizer(Mosek.Optimizer)) =#
    @variable(model, θ[1:d])
    @variable(model, t[1:n])

    for i in 1:n
        u = dot(X[i, :], θ) * y[i]
        tmp = @variable(model, [1:2], lower_bound=0.0)
        @constraint(model, sum(tmp) <= 1.0)
        @constraint(model, vec([u - t[i], 1, tmp[1]]) in MOI.ExponentialCone())
        @constraint(model, vec([- t[i],1, tmp[2]]) in MOI.ExponentialCone())
    end

    @objective(model, Min, 1.0 / n * sum(t))
    JuMP.optimize!(model)

    return JuMP.value.(model[:θ])
end

