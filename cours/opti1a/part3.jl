#' ---
#' title : Projet optimisation (3)
#' date: 17th May 2020
#' options:
#'   out_path : reports/modeler.html
#'---

#' # Utilisation d'un modeleur
#' Pour terminer la correction du projet d'optimisation, nous présentons
#' dans cette dernière partie des notions plus avancées d'optimisation. Notre objectif
#' final reste d'obtenir un algorithme de décomposition efficace.
#'
#' Plutôt que Python, nous préférons utiliser pour cette dernière partie le langage Julia, qui
#' permet plus de flexibilité grâce à l'excellent modeleur JuMP. Nous détaillons
#' dans la suite comment utiliser un modeleur d'optimisation.

#' ---
#' Commençons par importer les paquets dont nous avons besoin.
# Standard library
using DelimitedFiles
using LinearAlgebra
# Modeler
using JuMP
# Matplotlib
using PyPlot
# Import an optimization solver
using GLPK

# Import data
data = readdlm("data_temp_GR4.txt")
# Time span ...
tspan = data[:, 1]
# ... and horizon
H = length(tspan)
# define some constant
begin const
    # Timestep = half an hour
    Δ = 1800.0
    c_air = 1.256e3
    # Max and min power
    Pᵤ = 1000.0
    Pₗ = 0.0
    # Max and min temperature
    Tₗ = 20.0
    Tᵤ = 22.0
    # Initial position
    T0 = 20.0
    # Up and low elec prices
    c_up = 0.18
    c_lo = 0.13
    # Take parameters as fitted by scipy
    p = [0.00255712 0.00075877]
    # Volume
    V = 1000.0
    # Coefficient of electrical heater
    ηₕ = 0.5
    p3 = ηₕ / V
end

# Build cost vector
cₜ = c_lo * ones(H)
# Full tariff between 7am and 11pm
cₜ[7 .<= tspan .% 24 .<= 23] .= c_up
# Rescale cost vector to get €/W
cₜ ./= 2 * 1000.0

# Copy for convenience
tₑₓₜ = data[:, 4]
Φₛ = data[:, 3]
# Winter condition
tₑₓₜ .-= 10.0;

#' ---
#' ## Modélisation
#' Maintenant que nous avons chargé les données, nous pouvons modéliser
#' le problème d'optimisation. Plutôt que d'écrire les matrices "à la main",
#' nous allons construire le problème de manière incrémentale en utilisant
#' un modeleur d'optimisation: JuMP.
#' Commençons par montrer comment formuler le problème puis le résoudre en
#' utilisant un solveur d'optimisation open-source, GLPK.
# Instantiate model and pass as argument the solver
model = JuMP.Model(GLPK.Optimizer)
# Write up the optimization model
## Variables for heater
@variable(model, Pₗ <= P[1:H] <= Pᵤ)
## Variables for temperature
@variable(model, Tₗ <= T[1:H+1] <= Tᵤ)
## Add objective
@objective(model, Min, cₜ ⋅ P)
# Add initial position as constraint (note that Julia is 1-indexed)
@constraint(model, T[1] == T0)
# Add dynamics as constraints
for t in 1:H
    @constraint(model, T[t+1] == T[t] + Δ / c_air * (
                      p[1] * (tₑₓₜ[t] - T[t]) + p[2] * Φₛ[t] + p3 * P[t]))
end
# Once the problem formulated, solve it!
@time JuMP.optimize!(model)
println("Resolution time: ", JuMP.solve_time(model))

#' Notons que le temps de résolution est beaucoup plus court qu'avec scipy:
#' on passe de 0.5s à environ 2ms.
#' On récupère la solution optimale via les commandes :
T♯ = JuMP.value.(model[:T])
P♯ = JuMP.value.(model[:P])
println("Objective value: ", JuMP.objective_value(model))

#' Analysons la solution
fig = figure()
subplots
plot(tspan, P♯[1:H] ./ 1e3)
xlabel("Time")
ylabel("Heater power (kW)")
grid(ls=":")
display(fig)

#' On retrouve la solution que nous avions obtenu avec `scipy`.

#' ---
#' ### Qu'est-ce qu'un modeleur ?
#' Expliquons maintenant comment le modeleur fonctionne. Le problème est
#' construit ici de manière incrémentale avec de JuMP. Les données du problème
#' sont ensuite copiées dans le solveur solveur en utilisant son API C. A ce moment, les données
#' sont passées en mémoire directement au solveur. JuMP est un modeleur efficace,
#' qui a le bon goût d'être open-source et moderne (il supporte la plupart des
#' solveurs d'optimisation, et permet une grande liberté dans la manière de formuler
#' les problèmes, notamment coniques). Les autres modeleurs pouvant être utilisés sont :
#'
#' - [PuLP](https://coin-or.github.io/pulp/) : un modeleur Python pour les problèmes linéaires. Bien que supporté
#'   par l'organisation COIN-OR, ce modeleur est lent. A titre personnel, je décourage son utilisation.
#' - [Pyomo](http://www.pyomo.org/) est un autre modeleur codé en Python, rapide et qui supporte une
#'   très grande classe de problèmes. Un des grands avantages de Pyomo est l'ensemble
#'   de ses extensions (pour le bi-niveau, l'optimisation stochastique, l'optimisation
#'   de trajectoire, etc).
#' - [CVXPY](cvxpy.org/) est un nouveau modeleur de plus en plus utilisé en Python (avec
#'   backend C++), qui permet de
#'   formuler de manière très simple des problèmes d'optimisation convexe (et
#'   uniquement convexe). Ce modeleur est notamment très utilisé dans la communauté
#'   machine-learning.
#' - Enfin, la référence des modeleurs : [Ampl](https://ampl.com/). Ce modeleur est commercial, mais offre
#'   les meilleures performances (bien que JuMP soit quasiment aussi rapide). Les deux
#'   grands atouts d'Ampl sont ses fonctions de *presolve* avancées (pour éliminer les
#'   variables et les contraintes inutiles avant de passer le problème au solveur)
#'   ainsi que ses outils de différentiation automatique très performants, qui
#'   permettent de formuler des problèmes non-linéaires complexes.
#'
#' ---
#' ### Quels solveurs utiliser ?
#' Ici, nous avons utilisé le solveur GLPK pour résoudre notre problème.
#' Notons cependant qu'un grand nombre de solveurs existent pour l'optimisation.
#' Pour les problèmes linéaires, vous avez notamment à disposition les solveurs open-source
#'
#' - GLPK, très robuste mais qui est moins rapide que les autres solveurs. Ce solveur fait cependant très bien l'affaire pour la plupart des problèmes ayant une taille raisonnable.
#' - Clp, un solveur plus rapide que GLPK mais dont le code est difficilement lisible.
#' - HiGHS, le petit nouveau, dont le code est bien construit et qui offre des performances intéressantes.
#'
#' En entreprise, vous pouvez être amené à utiliser des solveurs commerciaux.
#' Les trois principaux solveurs qui se partagent le marché actuellement sont Gurobi, Cplex
#' et Xpress. Gurobi a pour réputation d'offrir les meilleures performances.
#'
#' Notons enfin l'excellent Mosek, très efficace pour résoudre des problème
#' d'optimisation conique.
#'
#' Nous renvoyons les étudiants intéressés au tableau [suivant](http://www.juliaopt.org/JuMP.jl/v0.21/installation/#Getting-Solvers-1)
#' pour avoir une liste plus exhaustive de solveur. Les benchmarks comparant
#' les solveurs entre eux sont disponibles sur le [site de Hans Mittelmann](http://plato.asu.edu/ftp/lpsimp.html).
#'
#' Maintenant que nous avons vu comment formuler un problème avec un modeleur d'optimisation,
#' nous pouvons passer à la partie de décomposition proprement dite.
