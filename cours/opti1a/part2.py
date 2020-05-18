#' # Projet d'optimisation (2) - Optimisation du chauffage
#' Cette partie fait suite à la première partie du projet d'optimisation,
#' où nous avions modélisé le système.
#' Nous cherchons maintenant à résoudre le problème d'optimisation que
#' nous avions formulé.

#' Commençons par importer les paquets usuels :

import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#' Définissons les constantes de notre code.
# Input file
DATA_FILE = "data_temp_GR4.txt"
# Capacity of air (=rho_air * c_m_air)
C_AIR = 1.256e3
# Sampling (half-hour)
DELTA_T = 1800.0
# Load dataset
data = np.loadtxt(DATA_FILE)

#' ---
#' ## Errata concernant la modélisation
#' Tout d'abord, revenons sur la méthode utilisée dans la partie précédente
#' pour identifier les paramètres de l'équation différentielle. Il s'avère
#' que cette méthode comportait une erreur, car la formulation aboutissait
#' à un problème non convexe. Ceci compliquait la résolution et aboutissait
#' à une non-unicité de la solution optimale. En effet, notons que si $p =
#' (p_1, p_2, p_3)$ satisfait l'équation :
#' $$
#' T^i_{k+1} = T^i_k + \dfrac{p_1 \,\Delta }{c_{air}} (p_2 (T^e_k - T^i_k) + p_3 \Phi_s^k ),
#' $$
#' alors pour tout  $q \in \mathbb{R}$,
#' les vecteurs $p' = (\dfrac{p_1}{q}, q p_2, q p_3)$, satisferont aussi l'équation.
#'
#' En effet, quelle que soit la formulation adoptée, nous n'avons pas assez d'information pour
#' identifier ici tous les paramètres de l'équation : il nous
#' reste un degré de liberté.
#'
#' Pour éviter ce problème nous reformulons l'équation de dynamique
#' avec uniquement deux paramètres: $p = (p_1, p_2)$, tels que
#' $$
#' T^i_{k+1} = T^i_k + \dfrac{\Delta}{c_{air}} (p_1 (T^e_k - T^i_k) + p_2 \Phi_s^k ).
#' $$
#' En adoptant ce formalisme, nous obtenons une identification beaucoup plus
#' satisfaisante des paramètres du problème (mais nous ne pourrons plus identifier un à un
#' les paramètres physiques).

def fit(data, shift=0):
    # Initial position
    u0 = data[shift, 1]
    t_int = data[shift:, 1]
    phi_s = data[shift:, 2]
    t_ext = data[shift:, 3]
    n_data = np.shape(data)[0] - shift

    def model_euler(u0, p):
        temperature = np.zeros(n_data)
        temperature[0] = u0
        for i in range(n_data-1):
            temperature[i+1] = temperature[i] + DELTA_T  / C_AIR * (
                p[0] * (t_ext[i] - temperature[i]) + p[1] * phi_s[i])
        return temperature

    resid = lambda p : (t_int - model_euler(u0, p))**2
    p0 = np.ones(2)
    opt = optimize.least_squares(resid, p0, method='lm', ftol=1e-12)
    print("Optimal cost ", opt["cost"])
    print("Optimal params: ", opt["x"])
    model1 = model_euler(u0, opt["x"])
    return model1, opt

model1, sol = fit(data)
p_opt = sol["x"]

#' Pour prendre en compte la puissance apportée par le radiateur, nous réécrivons
#' l'équation de dynamique en incluant un paramètre supplémentaire $p_3$:
#' $$
#' T^i_{k+1} = T^i_k + \dfrac{\Delta}{c_{air}} (p_1 (T^e_k - T^i_k) + p_2 \Phi_s^k + p_3 P_k)
#' $$
#' où, d'après la partie précédente, nous avons $p_3 = \dfrac{\gamma}{V}$,
#' avec $\gamma$ le rendement du radiateur et $V$ le volume du bâtiment, que
#' nous ne connaissons pas a priori.

#' ---
#' ## Optimisation
#' Nous pouvons maintenant rentrer dans l'optimisation proprement dite.
#' En prenant en compte la correction, le problème d'optimisation s'écrit :
#' $$
#' \begin{aligned}
#' \min_{P, T^i} \; & \Delta \sum_{k=1}^H c_k P_k \\
#'      \text{s.c.} & \quad T^i_{k+1} = f_k(T^i_k, p) & \forall k \\
#'                  & \quad 0 \leq P_k \leq \overline{P} & \forall k \\
#'                  & \quad \underline{T}^i_{k} \leq T^i_k \leq \overline{T}_k^i & \forall k
#' \end{aligned}
#' $$
#' où $f_k$ est la fonction de dynamique que nous avions identifié:
#' $$
#' f(T_k^i, p) = T^i_k + \dfrac{\Delta }{c_{air}} (p_1 (T^e_k - T^i_k) + p_2 \Phi_s^k + p_3 P_k)
#' $$
#'

#' *Question 2.1.*
#' Notons d'abord que le problème d'optimisation précédent est un problème
#' linéaire (contraintes linéaires, objectif linéaire). Le problème est
#' donc automatiquement convexe (mais non strictement convexe).

#' Notons $H$ l'horizon du problème. Nous avons $2H+1$ variables de décision
#' ($H$ correspondant à la puissance $P$, $H+1$ pour la température $T^i$),
#' et $H+1$ contraintes ($H$ pour la dynamique de la température intérieure,
#' $1$ pour la position initiale).
#'
#' Ecrivons le problème linéaire sous la forme canonique :
#' $$
#' \min_{x} \; c^\top x \quad \text{s.c. } \; Ax = b, \; x \geq 0
#' $$
#' où $x$ est la variable de décision. En particulier, pour nous $x$ correspondra
#' aux vecteurs concaténés de la puissance électrique du radiateur et de la température
#' intérieure: $x = (P_1, \cdots, P_H, T^i_1, \cdots, T^i_{H+1})$.

#' Il nous reste à identifier la matrice $A$ ainsi que les vecteurs $c$ et $b$
#' pour pouvoir résoudre notre problème par une méthode classique.
#' Notons dans la suite $\beta = \dfrac{\Delta}{c_{air}}$.
#' Le vecteur coût $c$ s'écrit $c = (c^{elec}, 0_H)$ où $0$ indique le vecteur nul,
#' $c_{elec}$ l'évolution du prix de l'électricité pendant la période qui nous
#' intéresse. Le vecteur $b$ vérifie, pour tout $k = 0, \cdots, H$,
#' $$
#' b_k  = \beta (p_1 T^e_k + p_2 \Phi_k^s )
#' $$
#' tandis que la matrice $A$ est telle que, pour les mêmes indices $k$
#' $$
#' A_{k, k} = - \beta p_3 , \quad
#' A_{k, k+H} =  \beta p_1 - 1 , \quad
#' A_{k, k+H+1} = -1
#' $$
#'
#' **Remarque:** Ici, la matrice $A$ comporte peu de coefficients non-nuls.
#' Plutôt que de la définir de manière dense, il serait plus judicieux d'utiliser
#' directement une matrice *sparse*.
#'
#' Les deux principaux algorithmes utilisés pour résoudre les problèmes linéaires sont :
#'
#' - l'algorithme du simplexe (plus précis).
#' - les algorithmes de points intérieurs (plus efficaces pour les problèmes
#' de très grande taille).
#'
#' Nous utiliserons dans la suite l'algorithme du simplexe tel qu'implémenté
#' dans la suite `scipy`.
#'
#' *Question 2.2.*
#' Il nous reste à expliciter les dernières grandeurs physiques. Nous
#' prendrons une puissance maximale du radiateur égale à $1$kW. Le volume
#' du bâtiment sera pris égal à $1000 m^3$. Le rendement du radiateur sera
#' égal à 0.5 (une partie de la puissance est dissipée par conduction dans
#' le mur attenant).

# Time span
t_index = data[:, 0]
## Horizon
H = data.shape[0]

# Write a function to optimize heating pattern
def optimize_heater(params, V=1000.0, t_min=19.0, t_max=22.0, p_max=1000.0,
                    yield_heater=0.5, condition="Normal",
                    algo="revised simplex"):
    p = params.copy()

    phi_s = data[:, 2]
    t_ext = data[:, 3].copy()
    if condition == "Winter":
        t_ext -= 10.0
    elif condition == "Siberia":
        t_ext -= 20.0

    # Temperatures
    T_bounds = (t_min, t_max)
    P_bounds = (0.0, p_max)
    p_3 = yield_heater / V

    # Costs
    ## Planning heures pleines / heures creuses
    h_pleines = (7, 23)

    ## Prices for heures pleines / heures creuses
    # € / kWh
    c_up, c_lo = (0.18, 0.13)
    c = c_lo * np.ones(H)
    t_index_normalized = t_index % 24
    condition =  np.where((h_pleines[0] <= t_index_normalized) &
                          (t_index_normalized <= h_pleines[1]))
    c[condition] = c_up
    # Rescale c to get cost in euros
    c /= (2 * 1000.0)

    # Write up matrix for the problem
    ## We have two decision variables: P and T
    ## We use the ordering [P; T]
    n_dim = 2*H + 1
    c_opt = np.concatenate((c, np.zeros(H+1)), axis=0)

    # Upper bounds
    bounds_up = np.zeros(n_dim)
    bounds_up[:H] = P_bounds[1]
    bounds_up[H:] = T_bounds[1]

    # Lower bounds
    bounds_lo = np.zeros(n_dim)
    bounds_lo[:H] = P_bounds[0]
    bounds_lo[H:] = T_bounds[0]

    bounds_tuple = [(lo, up) for (lo, up) in zip(bounds_lo, bounds_up)]

    # Matrix Ax = b encoding physical constraints (Euler scheme)
    beta = DELTA_T / C_AIR
    A = np.zeros((H+1, n_dim))
    b = np.zeros(H+1)
    for i in range(H):
        # T_i(k)
        A[i, H + i] = -1.0 + beta * p[0]
        # T_i(k+1)
        A[i, H + i + 1] = 1.0
        # P_i(k)
        A[i, i] = -beta * p_3
        # RHS
        b[i] = beta * (p[0] * t_ext[i] + p[1] * phi_s[i])

    # Add constraint for initial temperature
    # T_i(0) = t_0
    A[-1, H] = 1.0
    b[-1] = t_min

    tic = time.time()
    res = optimize.linprog(c_opt, A_ub=None, b_ub=None, A_eq=A, b_eq=b,
            bounds=bounds_tuple, method=algo, callback=None, options=None)
    print("Elapsed: {0:.4f} s".format(time.time() - tic))

    return res

#' Testons d'abord la résolution du problème avec les paramètres par défaut:
res = optimize_heater(p_opt, p_max=1000.0, condition="Normal", t_min=20.0)
print("Status: ", res["message"])
print("Cost: {0:.2f} €".format( res["fun"]))
# Postprocess
x_opt = res["x"]
t_sol = x_opt[H+1:]
p_sol = x_opt[:H]

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t_index, t_sol)
ax[0].set_ylabel("Temperature (°C)")
ax[1].step(t_index, p_sol)
ax[1].set_ylabel("Heating (W)")
ax[1].set_xlabel("Hours (h)")
plt.show()

#' On obtient un coût de 0.25€, ce qui est assez peu. Le chauffage n'est jamais
#' saturé. Essayons de dimiminuer la température extérieure pour observer
#' ce qui se passe.
res = optimize_heater(p_opt, p_max=1000.0, condition="Winter", t_min=20.0)
print("Status: ", res["message"])
print("Cost: {0:.2f} €".format( res["fun"]))
#' Le coût passe de 0.25€ à 0.70€, ce qui semble logique. Par contre, une facture
#' de 0.7€ pour trois jours de chauffage en hiver semble peu. Nous rencontrons
#' ici un problème : l'incertitude que nous avons sur le modèle que nous utilisons.
#' Rappelons nous la partie précédente : nous avions vu qu'idéalement la modélisation
#' renverrait un *intervalle de confiance* autour des valeurs que nous avons identifiées.
#' Cet intervalle de confiance devrait se retrouver dans l'optimisation. En effet,
#' rappelons nous que $p_1 = \dfrac{1}{V R_{eq}}$. Si nous avons une incertitude
#' de 10% sur $p_1$, nous avons un coût compris dans l'intervalle :

p_intervalle = p_opt.copy()
p_intervalle[0] = 1.1 * p_opt[0]
res = optimize_heater(p_intervalle, p_max=1000.0, condition="Winter", t_min=20.0)
c1 = res["fun"]
p_intervalle[0] = 0.9 * p_opt[0]
res = optimize_heater(p_intervalle, p_max=1000.0, condition="Winter", t_min=20.0)
c2 = res["fun"]
print("Interval: [{0:.2f}-{1:.2f}] €".format(c2, c1))
#' On obtient alors une incertitude de 10 centimes pour le coût optimal.

#' Nous voyons donc qu'il faut impérativement prendre en compte l'incertitude
#' sur les données dans notre étude. Pour la partie *modélisation*, nous pouvons
#' utiliser les techniques d'estimation bayésiennes pour avoir un intervalle
#' de confiance fiable autour de nos paramètres. Pour la partie *optimisation*,
#' la procédure est différente : nous devons utiliser des techniques d'*optimisation
#' robuste* pour optimiser au mieux notre système sachant l'incertitude que nous
#' avons sur les paramètres en entrée. Par exemple, si nous modélisons l'erreur
#' sur nos paramètres par un vecteur gaussien, le problème linéaire aura
#' pour équivalent robuste un problème d'optimisation *conique* (SOCP).
#' La modélisation et la résolution
#' de problèmes coniques ont pu être abordé pendant la partie 3 de l'examen de ce
#' cours.

#' Ici, restons dans le cadre de l'optimisation linéaire (scipy ne supporte
#' pas l'optimisation conique). Plaçons nous dans une hypothèse très défavorable,
#' où le paramètre $p_1$ est 20% plus grand qu'initialement estimé.
p_robuste = p_opt.copy()
p_robuste[0] *= 1.2
res = optimize_heater(p_robuste, p_max=1000.0, condition="Winter", t_min=20.0)
print("Status: ", res["message"])
print("Cost: {0:.2f} €".format( res["fun"]))
# Postprocess
x_opt = res["x"]
t_sol = x_opt[H+1:]
p_sol = x_opt[:H]

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t_index, t_sol)
ax[0].set_ylabel("Temperature (°C)")
ax[1].step(t_index, p_sol)
ax[1].set_ylabel("Heating (W)")
ax[1].set_xlabel("Hours (h)")
plt.show()

#' On obtient alors un coût de 0.92 euros. Si nous regardons les trajectoires,
#' nous observons que :
#'
#' - la puissance de chauffage est maximale avant le passage en heure
#' pleine (i.e. avant 7h), l'optimisation profitant du moment où l'électricité
#' est moins chère.
#' - Une fois passé en heure pleine, le chauffage est éteint. Il est rallumé
#' une fois que la température intérieure atteint la limite basse
#' (ici 20°C), pour éviter de sortir de l'intervalle de température admissible.

#' **Discussion sur les temps de calcul.**
#'
#' Regardons maintenant l'impact de l'agorithme sur les temps de calcul. Nous
#' obtenons ici une solution en environ 0.4-0.8s (sur le PC utilisé pour cette
#' étude, qui est un PC standard du commerce). Si nous utilisons à la place un algoirthme
#' de point intérieur
res = optimize_heater(p_robuste, p_max=1000.0, t_min=20.0, algo="interior-point")
#' nous remarquons que le temps de calcul diminue (d'environ un facteur 2).
#' Les performances respectives du simplexe et des points intérieurs dépendent
#' cependant grandement du problème. Notons aussi que généralement l'algorithme
#' du simplexe dual offre de meilleures performances que l'algorithme du simplexe
#' classique. Enfin, il convient de garder en mémoire que l'algorithme du simplexe
#' utilisé dans scipy est un algorithme entièrement codé en Python, qui en pratique
#' est assez peu performant (voire même pas très robuste, comme certains groupes
#' l'auront remarqué).

#' Nous verrons dans la suite du projet qu'en utilisant
#' un solveur classique nous pouvons diviser les temps de calcul d'un facteur 100.
