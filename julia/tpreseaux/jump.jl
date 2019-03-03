# TP RESEAU D'EAU
# ajouter matrice sparse
# ajouter descente nesterov
# ajouter plot graph

using JuMP, Ipopt

include("data.jl")

################################################################################
# PRIMAL PROBLEM
function fprimal(qc::Array{Float64})
    q = q0 + B*qc
    z = r .* abs.(q) .* q
    # valeur du critère
    F = q'*z / 3. + pr'*Ar*q
    return F
end

α1 = Ar' * pr


model = Model(with_optimizer(Ipopt.Optimizer))
JuMP.register(model, :fprimal, 1, fprimal, autodiff=true)

nx = n - md
@variable(model, qc[1:nx])
@variable(model, q[1:n])
@constraint(model, q .== q0 + B*qc)
@NLobjective(model, Min,
             sum(r[i] * abs(q[i]) * q[i]^2 / 3 + α1[i] * q[i] for i in 1:n))
@time JuMP.optimize!(model)
