#'
#' S, I, R
using Distributions

# Probability of contagion
c = 0.1
k = 5

function simulation(T, n, β, γ, s0=n-1, i0=1, r0=0)
    states = zeros(Int, T, 3)
    states[1, :] .= [s0, i0, r0]
    for t in 1:T-1
        s, i, r = states[t, :]
        contagions = Binomial(s, β / n * i) |> rand
        removals = Binomial(i, γ) |> rand

        states[t+1, 1] = s - contagions
        states[t+1, 2] = i + contagions - removals
        states[t+1, 3] = r + removals
    end
    return states
end

states = simulation(800, 1000, 0.1, 0.01)

n = 1000
using OrdinaryDiffEq

function sir(du, u, p, t)
    n, β, γ = p
    du[1] = - β /n * u[1] * u[2]
    du[2] = β / n * u[2] * u[1] - γ * u[2]
    du[3] = γ * u[2]
end

u0 = [n-1, 1.0, 0.0]
tspan = (0.0, 800.0)
params = [n, 0.1, 0.01]
prob = ODEProblem(sir, u0, tspan, params)
sol = solve(prob, Tsit5())

