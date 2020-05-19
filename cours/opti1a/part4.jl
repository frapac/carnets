#' ---
#' title : Projet optimisation (4)
#' date: 17th May 2020
#' options:
#'   out_path : reports/decomposition.html
#'---

#' # Décomposition
#' Nous terminons maintenant le projet d'optimisation en abordant les algorithmes
#' de décomposition (partie 3 du projet). Par souci de simplicité, nous utiliserons
#' le modeleur JuMP pour formuler chacun des problèmes, et nous utiliserons le
#' solveur Xpress, qui a l'avantage de supporter la modification de problème
#' en mémoire (ceci nous évite de devoir recréer de nouveaux problèmes d'optimisation
#' à chaque itération).
#'
#' Les algorithmes de décomposition n'ont pas été vu pendant le cours d'optimisation 1A.
#' Nous conseillons aux étudiants intéressés de se référer :
#'
#' - aux notes de cours de [Stephen Boyd](https://stanford.edu/class/ee364b/lectures/decomposition_notes.pdf) pour une présentation succincte des principes généraux;
#' - au [cours de Guy Cohen](http://cermics.enpc.fr/~cohen-g/documents/ParisIcours-A4-NB.pdf) pour une présentation plus avancée.
#'
#' ---
#' ## Découplage des problèmes
#' Supposons maintenant que nous ayons un immeuble avec 2 appartements, et
#' que nous cherchions à optimiser le chauffage. Les deux immeubles étant
#' accolés, le chauffage d'un appartement aura un impact sur l'autre appartement,
#' et réciproquement. Le flux thermique entre les deux appartements dépendra
#' de l'isolation des murs intercalaires.
#'
#' Prenons les équations de la dynamique de notre système.
#' Notons $R_i$ la résistance intercalaire, $T^1$ la température de l'appartement
#' du dessous, $T^2$ la température de l'appartement du dessus.
#' Nous avons:
#' $$
#' \left\{
#' \begin{aligned}
#' T^1_{k+1} &= T^1_k + \beta (p_1^1 (T^e_k - T^1_k) + p_2^1 \Phi^s_k + p_3^1 P_k^1 +
#'              \frac 1R_i (T^2_k - T^1_k))  \\
#' T^2_{k+1} &= T^2_k + \beta (p_1^2 (T^e_k - T^2_k) + p_2^2 \Phi^s_k + p_3^2 P_k^2 +
#'              \frac 1R_i (T^1_k - T^2_k))  \\
#' \end{aligned}
#' \right.
#' $$
#' où nous notons $\beta = \dfrac{\Delta }{c_{air}}$.
#'
#' Le problème d'optimisation global s'écrit alors :
#' $$
#' \begin{aligned}
#' \min_{P, T} \; & \Delta \sum_{k=1}^H c_k (P_k^1 + P_k^2) \\
#'      \text{s.c.} & \quad T_{k+1} = f_k(T_k, p) & \forall k \\
#'                  & \quad 0 \leq P_k^i \leq \overline{P}^i & \forall k, i = 1, 2 \\
#'                  & \quad \underline{T}^i_{k} \leq T^i_k \leq \overline{T}_k^i & \forall k, i = 1, 2
#' \end{aligned}
#' $$
#' Le couplage est introduit par le terme d'échange $\pm \frac 1R_i (T^1_k - T^2_k)$
#' dans les fonctions de dynamique $f_k$. Autrement, le problème se décompose de manière
#' immédiate appartement par appartement.
#'
#' Nous allons introduire de nouvelles variables pour découpler le problème
#' d'optimisation appartement par appartement. Pour tout $k = 1, \cdots, H$,
#' introduisons les deux variables auxiliaires $(w_k^1, w_k^2)$ telles que
#' $$
#' w_k^1 = T_k^2 , \quad w_k^2 = T_k^1
#' $$
#' Ces nouvelles variables $w^1_k, w_k^2$ portent entièrement le couplage entre
#' les sous-problèmes à l'instant $k$. La dynamique découplée $g_k$ se réécrit :
#' $$
#' \left\{
#' \begin{aligned}
#' T^1_{k+1} &= T^1_k + \beta (p_1^1 (T^e_k - T^1_k) + p_2^1 \Phi^s_k + p_3^1 P_k^1 +
#'              \frac 1R_i (w^1_k - T^1_k))  \\
#' T^2_{k+1} &= T^2_k + \beta (p_1^2 (T^e_k - T^2_k) + p_2^2 \Phi^s_k + p_3^2 P_k^2 +
#'              \frac 1R_i (w^2_k - T^2_k))  \\
#' \end{aligned}
#' \right.
#' $$
#' et nous obtenons alors le problème d'optimisation
#' $$
#' \begin{aligned}
#' \min_{P, T, w} \; & \Delta \sum_{k=1}^H c_k (P_k^1 + P_k^2) \\
#'      \text{s.c.} & \quad T_{k+1} = g_k(T_k, p) & \forall k \\
#'                  & \quad w_k^1 = T_k^2 , \; w_k^2 = T_k^1 & \forall k \\
#'                  & \quad 0 \leq P_k^i \leq \overline{P}^i & \forall k, i = 1, 2 \\
#'                  & \quad \underline{T}^i_{k} \leq T^i_k \leq \overline{T}_k^i & \forall k, i = 1, 2
#' \end{aligned}
#' $$
#' Nous allons maintenant dualiser les contraintes $w_k^1 = T_k^2$ (resp. $w_k^2 = T_k^1$)
#' en introduisant des multiplicateurs $\lambda_k^1$ (resp. $\lambda_k^2$).
#' Le problème primal se réécrit :
#' $$
#' \begin{aligned}
#' \min_{P, T, w} \max_{\lambda^1, \lambda^2} \; & \Delta \sum_{k=1}^H c_k (P_k^1 + P_k^2) +
#'                      \lambda_k^1 (w_k^1 - T_k^2) + \lambda_k^2 (w_k^2 - T_k^1) \\
#'      \text{s.c.} & \quad T_{k+1} = g_k(T_k, p) & \forall k \\
#'                  & \quad 0 \leq P_k^i \leq \overline{P}^i & \forall k, i = 1, 2 \\
#'                  & \quad \underline{T}^i_{k} \leq T^i_k \leq \overline{T}_k^i & \forall k, i = 1, 2
#' \end{aligned}
#' $$
#' et nous obtenons le problème dual correspondant :
#' $$
#' \begin{aligned}
#' \max_{\lambda^1, \lambda^2} \min_{P, T, w}  \; & \Delta \sum_{k=1}^H c_k (P_k^1 + P_k^2) +
#'                      \lambda_k^1 (w_k^1 - T_k^2) + \lambda_k^2 (w_k^2 - T_k^1) \\
#'      \text{s.c.} & \quad T_{k+1} = g_k(T_k, p) & \forall k \\
#'                  & \quad 0 \leq P_k^i \leq \overline{P}^i & \forall k, i = 1, 2 \\
#'                  & \quad \underline{T}^i_{k} \leq T^i_k \leq \overline{T}_k^i & \forall k, i = 1, 2
#' \end{aligned}
#' $$
#' C'est ce problème dual qui nous intéresse ici. Notons que pour $\lambda_1, \lambda_2$
#' fixés, le problème intérieur (correspondant au `min`) se découple naturellement
#' en deux sous-problèmes $J^1$ et $J^2$:
#' $$
#' \begin{aligned}
#' J^i(\lambda^1, \lambda^2) =  \min_{P^i, T^i, w^i}  \; & \Delta \sum_{k=1}^H c_k P_k^i +
#'                      \lambda_k^i w_k^i - \lambda_k^{i+1} T_k^i)  \\
#'      \text{s.c.} & \quad T_{k+1} = f_k(T_k, p) & \forall k \\
#'                  & \quad 0 \leq P_k^i \leq \overline{P}^i & \forall k\\
#'                  & \quad \underline{T}^i_{k} \leq T^i_k \leq \overline{T}_k^i & \forall k
#' \end{aligned}
#' $$
#' Nous cherchons dès-lors à résoudre ce problème dual
#' $$
#' \max_{\lambda^1, \lambda^2} \; \sum_{i=1}^2 J^i(\lambda^1, \lambda^2) .
#' $$
#'
#' ### Algorithme de décomposition
#' Nous sommes maintenant en mesure de proposer un algorithme de décomposition.
#' Supposons donné deux multiplicateurs initiaux $\lambda^{1, (0)}, \lambda^{2, (0)}$.
#' Nous allons résoudre le problème de manière itérative, où à chaque itération $l$,
#' nous suivrons les étapes :
#'
#' - Résoudre de manière **indépendante** les sous-problèmes $J^1(\lambda^{1(l)}, \lambda^{2(l)})$
#'   et $J^2(\lambda^{1(l)}, \lambda^{2(l)})$
#' - Collecter les solutions de $J^1$: $(w^{1(l)}, T^{1(l)})$
#' - Collecter les solutions de $J^2$: $(w^{2(l)}, T^{2(l)})$
#' - Mettre à jour les multiplicateurs suivant
#' $$
#' \left\{
#' \begin{aligned}
#' \lambda^{1(l+1)} &= \lambda^{1(l)} + \rho (w^{1(l)} - T^{2(l)}) \\
#' \lambda^{2(l+1)} &= \lambda^{2(l)} + \rho (w^{2(l)} - T^{1(l)})
#' \end{aligned}
#' \right.
#' $$
#' où $\rho$ est un pas fixé.

#' Notons que toute la procédure décrite précédemment s'étend naturellement
#' à un système de $n$ appartements.


#' ---
#' ## Implémentation
#' Nous commençons par importer les paramètres usuels
# Include data from Part III:
include("data.jl")
# Thermal resistance between appartments
const Ri = 0.1 # K/W
const p4 = 1.0 / (Ri * V);
#' Attention ici : la résistance $R_i$ impacte très fortement le couplage
#' entre les sous-systèmes, et donc aura une influence sur la convergence
#' des algorithmes de décomposition.

#' Nous utiliserons dans la suite le solveur commercial Xpress. Ce dernier
#' présente en effet la particularité de pouvoir modifier le modèle en place,
#' sans devoir le reconstruire de zéro. Nous pouvons en tirer profit dans
#' l'algorithme de décomposition. En effet, à chaque itération de l'algorithme, seul
#' l'objectif change, les contraintes restant les mêmes. Nous utiliserons
#' alors le solveur de la manière suivante, pour chacun des sous-problèmes :
#' ```
#' - load local problem in solver
#' - at each iteration k, get price lambda_k
#'     i) Update objective in solver with new price lambda_k
#'     ii) Solve subproblem with warm-start
#'     iii) Get optimal solution
#' ```
using Printf
using LinearAlgebra
using JuMP
using Xpress
# Matplotlib
using PyPlot

SOLVER = () -> Xpress.Optimizer(OUTPUTLOG=0);


#' Commençons par écrire une fonction pour résoudre le problème de manière
#' globale.
"Build a global problem with `n` coupling subproblems."
function build_global(solver)
    n = 2 # number of subproblems
    tmin = [20.0, 22.0]
    tmax = [22.0, 24.0]
    model = Model(solver)
    @variable(model, 0.0 <= P[i=1:n, j=1:H] <= 1000.0)
    @variable(model, tmin[i] <= T[i=1:n, j=1:(H+1)] <= tmax[i])
    @objective(model, Min, cₜ ⋅ (P[1, :] + P[2, :]))
    @constraint(model, init_ctr[i=1:n], T[i, 1] == tmin[i])
    for t in 1:H
        @constraint(model, T[1, t+1] == T[1, t] + Δ / c_air * (
                            p[1] * (tₑₓₜ[t] - T[1, t]) +
                            2/3 * p[2] * Φₛ[t] +  # bottom position receives less sun
                            p3 * P[1, t] +
                            p4 * (T[2, t] - T[1, t])))
        @constraint(model, T[2, t+1] == T[2, t] + Δ / c_air * (
                            p[1] * (tₑₓₜ[t] - T[2, t]) +
                            p[2] * Φₛ[t] +
                            p3 * P[2, t] +
                            p4 * (T[1, t] - T[2, t])))
    end
    return model
end

# Build and solve global problem!
global_model = build_global(SOLVER)
JuMP.optimize!(global_model)
global_solve_time = JuMP.solve_time(global_model);

#' Passons maintenant à la décomposition proprement dite.
#' Nous commençons par écrire une fonction construisant le modèle associé
#' à chacun des sous-problèmes, en ajoutant une variable couplante `w`.
function build_subproblem(solver; tmin=20.0, tmax=22.0, coef_sun=1.0)
    # Instantiate model
    model = Model(solver)
    # Write problem as usual ...
    @variable(model, 0.0 <= P[1:H] <= 1000.0)
    @variable(model, tmin <= T[1:H+1] <= tmax)
    @objective(model, Min, cₜ ⋅ P)
    @constraint(model, T[1] == tmin)
    # ... but with a coupling variable: w = T_k^{neighbors}
    @variable(model, w[1:H])
    for t in 1:H
        @constraint(model, T[t+1] == T[t] + Δ / c_air * (
                            p[1] * (tₑₓₜ[t] - T[t]) + coef_sun * p[2] * Φₛ[t] + p3 * P[t] + p4 * (w[t] - T[t])))
    end
    return model
end

function solve_subproblem(submodel, λ1, λ2)
    # Update subproblem
    P = submodel[:P]
    w = submodel[:w]
    T = submodel[:T][1:end-1]
    # Update objective
    @objective(submodel, Min, cₜ ⋅ P + λ1 ⋅ w  - λ2 ⋅ T)
    JuMP.optimize!(submodel)
    # Here the coupling variable is the inner temperature + coupling variable w
    c♯ = JuMP.objective_value(submodel)
    T♯ = JuMP.value.(submodel[:T])[1:end-1]
    w♯ = JuMP.value.(submodel[:w])
    return (c♯, T♯, w♯)
end

#' L'algorithme de décomposition s'écrit alors :
function classical_decomposition(;maxit=100, ρ=1e-3)
    # Build each subproblem
    m1 = build_subproblem(SOLVER, tmin=20.0, tmax=22.0, coef_sun=2/3)
    m2 = build_subproblem(SOLVER, tmin=22.0, tmax=24.0, coef_sun=1.0)

    # Two coordination prices
    λ1 = zeros(H)
    λ2 = zeros(H)

    trace_cost = Float64[]
    trace_grad = Float64[]

    for i in 1:maxit
        # Here, the resolution of the two subproblems should be in a
        # parallelized fashion
        c1, t1, w1 = solve_subproblem(m1, λ1, λ2)
        c2, t2, w2 = solve_subproblem(m2, λ2, λ1)

        # (Sub)-Gradient
        ∇c1 = w1 - t2
        ∇c2 = w2 - t1

        # Update coordination prices
        λ1 .= λ1 + ρ * ∇c1
        λ2 .= λ2 + ρ * ∇c2

        cost = c1 + c2
        ∇c = [∇c1; ∇c2]
        push!(trace_cost, cost)
        push!(trace_grad, norm(∇c, Inf))
    end
    λ♯ = [λ1; λ2]
    return λ♯, m1, m2, trace_cost, trace_grad
end

λ♯, m1, m2, costs, grads = classical_decomposition(maxit=500);

#' Analysons la convergence de l'algorithme de décomposition :
fig = figure()
subplot(211)
plot(costs)
xlabel("Iteration")
ylabel("Costs")
grid()
subplot(212)
plot(log10.(grads[2:end]))
xlabel("Iteration")
ylabel("Norm of gradients (log-scale)")
grid()
display(fig)

#' On remarque que l'algorithme ne converge pas : l'objective oscille et
#' le gradient ne converge pas vers 0 en norme. Les solutions renvoyées par
#' l'algorithme sont de plus catastrophiques :
P1 = JuMP.value.(m1[:P]) # in problem 1
P2 = JuMP.value.(m2[:P]) # in problem 2
T1 = JuMP.value.(m1[:T])[1:H]
T2 = JuMP.value.(m2[:T])[1:H]
fig = figure()
subplot(211)
plot(tspan, P1, label="Heater 1")
plot(tspan, P2, label="Heater 2")
xlabel("Time (h)")
ylabel("Heaters (W)")
grid()
legend()
subplot(212)
plot(tspan, T1, label="Temp 1")
plot(tspan, T2, label="Temp 2")
xlabel("Time (h)")
ylabel("Temperature (celsius degree)")
grid()
display(fig)

#' Concertons nous. Nous avons un algorithme qui tourne, mais qui ne
#' converge pas. En fait, nous savons pourquoi. En effet, rappelons nous
#' que les sous-problèmes sont des problèmes linéaires, et nous n'avons pas
#' nécessairement unicité de la solution. La fonction duale $\Phi(\lambda^1, \lambda^2) =
#' J^1(\lambda^1, \lambda^2) + J^2(\lambda^1, \lambda^2)$ n'est donc pas différentiable,
#' mais *sous-différentiable*. C'est pourquoi nous observons ce phénomène de cyclage
#' qui empêche la convergence avec ce premier algorithme. Plutôt que de faire une
#' montée de gradient (rappelons que le problème dual est un problème de maximisation), un algorithme
#' *non-smooth* comme les méthodes de bundle serait ici plus approprié. Nous allons voir
#' dans la suite une autre méthode, qui permet de rendre le problème global *lisse*:
#' la décomposition par Lagrangien augmenté.

#' **Décomposition par Lagrangien augmenté.**
#' La décomposition par Lagrangien augmenté *lisse* le problème en ajoutant
#' une pénalisation quadratique. Pour tout $k$, les termes linéaires
#' $$
#' \lambda_k^1 (w_k^1 - T_k^2) + \lambda_k^2 (w_k^2 - T_k^1)
#' $$
#' sont *augmentés* par deux termes quadratiques
#' $$
#' \lambda_k^1 (w_k^1 - T_k^2) + \lambda_k^2 (w_k^2 - T_k^1)
#' + r (w_k^1 - T_k^2)^2 + r ( w_k^2 - T_k^1)^2
#' $$
#' où $r > 0$ est une constante. Cependant, les termes quadratiques rajoutent
#' un couplage entre les sous-problèmes, du fait des termes croisés.
#' Pour redécoupler le problème, on utilise la solution de l'itéré courant
#' $(T_k^{1(l)}, w_k^{1(l)})$ pour écrire
#' $$
#' (w_k^1 - T_k^2)^2 \approx (w_k^1)^2 + (T_k^2)^2 - T_k^{2(l)} w_k^1
#' - w_k^{1(l)} T_k^2
#' $$
#' et on procède de même pour le terme $r ( w_k^2 - T_k^1)^2$.
#' Nous ne rentrons pas trop dans les détails ici. Nous renvoyons au cours de Guy
#' Cohen pour plus de précisions.
#'
#' Les termes quadratiques sont ensuite rajoutés dans chacun des sous-problèmes.
#' De linéaires, ces derniers deviennent quadratiques, et donc plus difficile à
#' résoudre. Heureusement, les solveurs modernes sont efficaces pour résoudre
#' les problèmes quadratiques convexes, et Xpress gère nativement la résolution des QP.
function solve_subproblem(submodel, λ1, λ2, r, Tₙ, wₙ)
    P = submodel[:P]
    w = submodel[:w]
    T = submodel[:T][1:end-1]
    # Quadratic objective
    @objective(submodel, Min, cₜ ⋅ P + λ1 ⋅ w  - λ2 ⋅ T +
               r * (dot(w, w) - dot(Tₙ, w) + dot(T, T) - dot(wₙ, T)))
    JuMP.optimize!(submodel)
    # Here the coupling variable is the inner temperature + coupling variable w
    c♯ = JuMP.objective_value(submodel)
    T♯ = JuMP.value.(submodel[:T])[1:end-1]
    w♯ = JuMP.value.(submodel[:w])
    return (c♯, T♯, w♯)
end

function augmented_lagrangian_decomposition(;maxit=100, tol=1e-10)
    pmax = 1000.0
    m1 = build_subproblem(SOLVER, tmin=20.0, tmax=22.0, coef_sun=2/3)
    m2 = build_subproblem(SOLVER, tmin=22.0, tmax=24.0, coef_sun=1.0)

    # Two coordination prices
    λ1 = zeros(H)
    λ2 = zeros(H)
    t1 = zeros(H)
    t2 = zeros(H)
    w1 = zeros(H)
    w2 = zeros(H)

    trace_cost = Float64[]
    trace_grad = Float64[]

    ρ = 1.0
    r = 1.0

    tic = time()
    for i in 1:maxit
        c1, t1, w1 = solve_subproblem(m1, λ1, λ2, r, t2, w2)
        c2, t2, w2 = solve_subproblem(m2, λ2, λ1, r, t1, w1)

        # (Sub)-Gradient
        ∇c1 = w1 - t2
        ∇c2 = w2 - t1

        # Update coordination prices
        λ1 .= λ1 + ρ * ∇c1
        λ2 .= λ2 + ρ * ∇c2

        cost = c1 + c2
        ∇c = [∇c1; ∇c2]
        push!(trace_cost, cost)
        push!(trace_grad, norm(∇c, Inf))
        if norm(∇c, Inf) <= tol
            break
        end
    end
    println("Elapsed time (s): ", time() - tic)
    λ♯ = [λ1; λ2]
    return λ♯, m1, m2, trace_cost, trace_grad
end

res, m1, m2, costs, grads = augmented_lagrangian_decomposition(maxit=2000);

#' Analysons la convergence du nouvel algorithme :
opt_cost = JuMP.objective_value(global_model)
fig = figure()
subplot(211)
plot(log10.(costs .- opt_cost))
xlabel("Iteration")
ylabel("f - f_opt (log scale)")
grid()
subplot(212)
plot(log10.(grads[2:end]))
xlabel("Iteration")
ylabel("Norm of gradients (log-scale)")
grid()
display(fig)

#' La convergence est ici avérée : le gradient converge bien vers 0 (en moins de 70
#' itérations), et le coût se stabilise (mais notons qu'on ne converge pas exactement
#' vers le coût optimal, nous y reviendrons après). Notons que cette convergence
#' franche ne s'observe pas nécessairement : la performance de l'algorithme de
#' décomposition est très dépendante du couplage existant entre les sous-problèmes
#' (si nous augmentons ou diminuons la résistance $R_i$, la vitesse de convergence
#' sera différente). De plus, nous mettons environ 3s pour converger, alors que
#' la solution optimale est trouvée sur le problème global en
println("Time to compute global solution: ", global_solve_time)
#' ce qui est plus de 1000 fois plus rapide. En effet, sur des petits
#' problèmes, il est très dur de battre un algorithme aussi mature que le simplexe
#' en utilisant des algorithmes de décomposition. La décomposition est plus adaptée
#' pour traiter de très gros problèmes (plusieurs millions de variables et de contraintes),
#' ou pour résoudre des systèmes de manière décentralisée (par exemple qu'on ne souhaite pas
#' que quelqu'un ait accès à toutes les données du problèmes, pour des raisons de vie privée).

#' Si nous regardons maintenant les solutions renvoyées par l'algorithme :
P1 = JuMP.value.(m1[:P]) # in problem 1
P2 = JuMP.value.(m2[:P]) # in problem 2
T1 = JuMP.value.(m1[:T])[1:H]
T2 = JuMP.value.(m2[:T])[1:H]

fig = figure()
subplot(211)
plot(tspan, P1, label="Heater 1")
plot(tspan, P2, label="Heater 2")
xlabel("Time (h)")
ylabel("Heaters (W)")
grid()
legend()
subplot(212)
plot(tspan, T1, label="Temp 1")
plot(tspan, T2, label="Temp 2")
xlabel("Time (h)")
ylabel("Temperature (celsius degree)")
grid()
display(fig)
#' Nous observons que  l'algorithme renvoie des solutions faisables et
#' cohérentes. Par contre, si nous comparons avec les solutions optimales,
#' obtenues au début en résolvant le problème global avec un simplexe :
global_P = JuMP.value.(global_model[:P])
p1_diff = P1 - global_P[1, :]
p2_diff = P2 - global_P[2, :]
fig = figure()
subplot(211)
xlabel("Time (h)")
ylabel("Heater subproblem 1 (W)")
plot(tspan, P1, label="decomposition")
plot(tspan, global_P[1, :], label="simplex")
legend()
grid()
subplot(212)
ylabel("Heater subproblem 2 (W)")
xlabel("Time (h)")
plot(tspan, P2)
plot(tspan, global_P[2, :])
grid()
display(fig)
#' nous observons des différences. Nous ne convergeons donc pas exactement
#' vers la solution optimale. Ceci se voit bien si nous comparons les coûts finaux :
@printf("Solution (global optimisation): %.4f €\n", opt_cost)
@printf("Solution (decomposition): %.4f €\n", costs[end])
gap = (costs[end] - opt_cost) / opt_cost
@printf("Gap: %.4e\n", gap)
#' où nous avons un gap final de l'ordre de 12%...

#' Enfin, affichons les multiplicateurs renvoyés par l'algorithme de décomposition:
lam_1 = res[1:H]
lam_2 = res[H+1:end]
fig = figure()
xlabel("Time (h)")
ylabel("Multipliers")
plot(tspan, lam_1, label="lambda_1")
plot(tspan, lam_2, label="lambda_2")
legend()
grid()
display(fig)
#' Ces multiplicateurs nous donnent de nombreuses informations sur le problème.
#' Notamment, on peut les assimiler à un prix d'échange entre les sous-systèmes
#' (= combien l'appartement 1 est prêt à payer pour prendre le chaleur de l'appartement 2).
#' Ces multiplicateurs peuvent par exemple être utilisés pour fixer un système de prix
#' (par exemple, sur un réseau électrique, pour fixer les prix d'échanges entre différents
#' acteurs concurrents).


#' **Discussion finale.**
#' Il faudrait que nous poussions plus avant pour comprendre la différence que
#' nous observons entre la solution optimale et la solution renvoyée par l'algorithme
#' de décomposition. Ceci commence cependant à dépasser le cadre de cette correction
#' (ou pour être honnête : je ne vois pas actuellement comment corriger l'algorithme
#' de décomposition, malgré de multiples tentatives...).
#'
#' Nous avons finalement vu dans cette dernière partie la démarche à suivre pour
#' décomposer effectivement un problème d'optimisation. Nous avons touché du doigt
#' la difficulté d'obtenir un algorithme de décomposition efficace,
#' et introduit une méthode de Lagrangien
#' augmenté qui renvoie une solution proche de l'optimale.
#'
#' La dernière méthode s'apparente à l'algorithme *alternating direction method of multipliers* (ADMM)
#' qui est aujourd'hui l'algorithme de décomposition le plus utilisée (car simple à mettre
#' en oeuvre). Nous renvoyons [aux slides de Boyd](https://web.stanford.edu/~boyd/papers/pdf/admm_slides.pdf)
#' pour une présentation succincte de cette méthode.
#'
#' Notons aussi que nous avons vu ici une méthode de décomposition *duale*. Il
#' existe d'autre méthode de décomposition : la décomposition *primale*, et la décomposition
#' par prédiction (que l'on peut qualifier abusivement de *primale-duale*, car elle mélange
#' les deux dernières décompositions). Nous renvoyons encore une fois au cours de Guy Cohen
#' pour une présentation exhaustive de ces dernières méthodes.
