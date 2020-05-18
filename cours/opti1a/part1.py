#' # Projet d'optimisation - Modélisation
#' Notre système comprend un bâtiment équipé d'un radiateur électrique.
#' Pour notre modélisation, nous avons à disposition l'évolution de la température
#' intérieure, de la température extérieure et du flux solaire pendant 72h.
#' Nous cherchons dans la première partie du projet à modéliser notre
#' système sous la forme d'un problème d'optimisation.
#'
#' ## Question 1
#' Ecrivons dans un premier temps le critère associé au problème d'optimisation.
#' Nous avons un radiateur électrique, qui consomme une certaine puissance
#' pour chauffer la maison. Cette puissance est importée d'un réseau de
#' de distribution extérieur, importation qui se fait à un tarif donné. Supposons que le tarif
#' varie au cours du temps, et associons lui la function $c(t)$, qui à chaque
#' instant $t$ nous donne le prix de l'électricité en euro. Si nous notons
#' de manière similaire $P(t)$ la fonction qui à chaque instant $t$ nous donne
#' la consommation électrique du radiateur, notre critère s'écrit alors, en temps
#' continu :
#' $$
#' \int_{0}^T \; c(t) P(t) dt \;.
#' $$
#' En pratique, nous utiliserons une version discrétisée de l'intégrale
#' pour définir notre objectif. Notons $\Delta$ un pas de discrétisation
#' donné (par exemple 10mn si nous considérons un compteur linky, ou bien 30mn
#' si on utilise la même discrétisation que celle des données).
#' Nous noterons, de manière abusive: $c_k = c(k \Delta)$ (resp.
#' $P_k = P(k \Delta)$) nos variables discrétisées. Le critère se réécrit alors
#' en version discrète:
#' $$
#' \Delta \sum_{k=0}^T \; c_k P_k \;
#' $$
#'
#' ## Question 2
#' Maintenant que notre coût est explicité, nous pouvons nous intéresser à
#' la dynamique physique de notre système. La principale variable d'intérêt
#' est pour nous la température intérieure, que nous cherchons à contrôler.
#' Cette température est influencée par la température extérieure et le flux
#' solaire incident, deux grandeurs dont l'évolution nous est donnée. Nous
#' pouvons concevoir plusieurs modèles physiques pour notre modélisation, de
#' complexité décroissante :
#'
#' - Un modèle basé sur l'équation de la chaleur permettrait de modéliser
#' la température dans l'ensemble du bâtiment. L'équation différentielle
#' partielle correspondante serait alors résolue par différences finies. Ce modèle,
#' complexe, présente l'intérêt d'être exhaustif. Cependant, nous manquons de
#' données pour l'implémenter dans le cadre du problème qui nous intéresse (nous
#' ne connaissons pas les caractéristiques des murs, ni la forme précise de la pièce,
#' ni les emplacements des fenêtres...).
#' - Un modèle basé sur une analogie électrique permet de modéliser plus
#' simplement notre système, au détriment de la précision. Ce modèle suppose
#' la température homogène à l'intérieur du bâtiment (et néglige donc la présence
#' d'éventuels murs intérieurs) et assimile l'air à l'intérieur de la pièce
#' à une capacité (où on peut stocker de l'énergie sous forme de chaleur). Les murs
#' extérieurs font alors office de résistances, et la température est elle-même vue
#' comme une tension. Les différents flux de températures correspondent alors
#' à des intensités.
#'
#' Suivant le degré de précision que l'on souhaite obtenir, l'analogie électrique
#' peut s'avérer plus ou moins précise. Un modèle usuel est le R6C2 (6 résistances,
#' deux capacités) qui utilise plusieurs résistances (pour les murs, les vitres)
#' et deux capacités (une pour la température intérieure, une pour la température
#' des murs extérieurs) pour modéliser notre système. Toutefois, n'ayant pas à disposition
#' la topologie, même basique, du bâtîment, ni même la température des murs, ce modèle
#' est pour nous inabordable.
#'
#' Nous choississons dès lors de modéliser notre système par le modèle le plus
#' simple à disposition : le R1C1. Nous considérons un seul flux thermique avec
#' l'extérieur, et une capacité correspondant à la pièce intérieure du bâtiment
#' (encore une fois, nous faisons fi des pièces intérieures en regardant le dit
#' bâtiment comme une grande pièce vide).
#'
#' Dans le cadre de ce modèle, un rapide bilan d'énergie nous donne
#' l'équation différentielle suivante :
#' $$
#' \rho c_{air} V \dfrac{dT^i}{dt} = \dfrac{1}{R_{eq}} (T^e - T^i) + \kappa S \Phi_s
#' + \alpha P
#' $$
#' où nous notons $T^i$ la température intérieure, $T^e$ la température
#' extérieure, $\rho$ et $c_{air}$ la masse volumique et la capacité thermique
#' de l'air, $V$ le volume de la pièce intérieure, $R_{eq}$ la résistance équivalente
#' des murs, $\Phi_s$ le flux solaire aggrégé, $\kappa$ le coefficient de transmission
#' du flux, $S$ la surface totale d'échange, $P$ la puissance électrique
#' utilisée par le radiateur, et $\alpha$ le rendement de ce dernier.
#'
#' ## Question 3
#' Pour discrétiser cette équation différentielle, une première idée
#' est d'utiliser un schéma d'Euler explicite. Bien que rustique, ce modèle peut
#' s'avérer pertinent si la constante de temps de notre système est grande.
#' Si nous discrétisons à un pas $\Delta$ (qui pourra être judicieusement
#' choisi égal au pas de discrétisation de la question 1), nous obtenons :
#' $$
#' T^i_{k+1} = T^i_k + \dfrac{\Delta}{\rho c_{air}V} (
#' \frac 1R_{eq} (T^e_k - T^i_k) + \kappa S \Phi_s^k + \alpha P^k )
#' $$
#' en notant $k$ l'indice de discrétisation tel que $T^i_k = T^i(k \Delta)$.
#'
#' On cherche maintenant à identifier les paramètres que nous ne connaissons
#' pas dans l'équation différentielle, à savoir la résistance équivalente $R_{eq}$,
#' le coefficient $\kappa$, la surface $S$ et le volume $V$.
#' Pour ce faire, nous disposons de données observées, où le chauffage est éteint.
#' Notons $p_1 = \dfrac{1}{\rho V}$, $p_2 = \dfrac{ 1}{R_{eq}}$, $p_3 = \kappa S$.
#' On écrit alors l'équation différentielle discrétisée que nous utiliserons
#' lors de l'identification :
#' $$
#' T^i_{k+1} = T^i_k + \dfrac{\Delta\, p_1}{c_{air}} (
#' p_2 (T^e_k - T^i_k) + p_3 \Phi_s^k )
#' $$
#' les trois paramètres que nous identifierons numériquement étant notés dès lors
#' $(p_1, p_2, p_3)$.
#'
#' **Remarque :**
#' Dans notre modèle, il aurait été plus simple de définir $p_1 = \rho V$ à la place de $p_1 = \dfrac{1}{\rho V}$.
#' Cependant, numériquement, il convient de faire attention aux divisions qui
#' ont tendance à diminuer la précision, d'autant plus si $p_1$ est proche de
#' 0. Conseil : en optimisation, autant que possible, essayez de faire passer
#' vos variables d'optimisation au numérateur plutôt qu'au dénominateur.
#' [Plus d'info ici](https://xkcd.com/2295/).
#'
#' **Algorithme d'identification :**
#' Nous sommes maintenant en mesure d'écrire notre algorithme d'identification
#' de paramètres. L'idée est d'utiliser une pénalisation par moindre carré :
#'
#' - Supposons donné un vecteur de paramètre $p = (p_1, p_2, p_3)$.
#' - On calcule l'évolution de la température intérieure en utilisant notre
#' modèle. On obtient un vecteur $T^i_m$.
#' - On pénalise la différence entre $T^i_m$ et $T^i_{real}$ (ce dernier
#' correspondant aux données mesurées) par un coût quadratique $(T^i_m - T^i_{real})^2$.
#'
#' L'algorithme est implémenté dans la fonction `fit` définie dans la suite.
#' On utilise un algorithme de moindre carré non-linéaire pour trouver nos
#' paramètres $p_1, p_2, p_3$ (plus précisément l'algorithme de Levenberg-Marquardt,
#' bien adapté pour des problèmes de petite taille comme le notre).

#' On importe les paquets numériques usuels:
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#' Définissons les constantes de notre code.
# Input file
DATA_FILE = "data_temp_GR4.txt"
# Capacity of air (=rho_air * c_v_air)
C_AIR = 1.256e3
# Sampling (half-hour)
DELTA_T = 1800.0

#' Chargeons les données en mémoire
# Load dataset
data = np.loadtxt(DATA_FILE)
# Initial position

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
    opt = optimize.least_squares(resid, p0, method='lm', jac="3-point",
                                 ftol=1e-12)
    print("Optimal cost ", opt["cost"])
    print("Optimal params: ", opt["x"])
    model1 = model_euler(u0, opt["x"])
    return model1, opt

model1, _ = fit(data)
t_axis = data[:, 0]
t_int = data[:, 1]
fig, ax = plt.subplots()
ax.plot(t_axis, model1, label="model")
ax.plot(t_axis, t_int, label="observation")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Temperature (°C)")

#' Notre première tentative permet de retrouver à peu près l'évolution observée
#' dans les mesures. Cependant, on remarque que le coût final reste élevé
#' (environ 1.05) et nous observons que le modèle a du mal à suivre la variation
#' brusque de la température intérieure observée au début des mesures.
#'
#' Pour éviter ce phénomène, on choisit de se débarasser des premières mesures
#' pour ne pas avoir à traiter cette brusque variation initiale (qui ici peut
#' correspondre au fait que le chauffage ait été éteint brusquement).
#' Pour ce faire, on règle le paramètre `shift` dans la fonction `fit`.
# We remove the first hour in the dataset
shift = 0
model2, res = fit(data, shift=shift)
t_axis = data[shift:, 0]
t_int = data[shift:, 1]
fig, ax = plt.subplots()
ax.plot(t_axis, model2, label="model")
ax.plot(t_axis, t_int, label="observation")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Temperature (°C)")

#' On observe que ce second modèle est meilleur que le premier (coût final
#' de 0.08 à la place de 1.05).
#' Récupérons notre vecteur de paramètres:
p_opt = res["x"]

#' Identifions maintenant les données. En supposant que le bâtiment est un cube
#' (hum) de côté $a$, nous avons $V = a^3$, puis $S = 3 a^2$ (on suppose que
#' trois côtés du cube sont illuminés à chaque instant).
V = 1.0 / (p_opt[0])
a = np.power(V, 1/3)
R_eq = 1.0 / p_opt[1]
S = 3 * a**2
kappa = p_opt[2] / S

print("* Volume: %.3f m^3" % V)
print("* length: %.3f m" % a)
print("* resistance: %.3f K/W" % R_eq)
print("* kappa: %.3e SI" % kappa)

#' et la constante de temps de notre système:
tau_constant = C_AIR * 1.225 * V * R_eq
print("* Time constant: %.2e s" % tau_constant)


#' **Pistes d'amélioration :**
#' Cette étude permet d'identifier les paramètres de l'équation différentielle.
#' Cependant, de nombreuses pistes d'extension existent.
#'
#' - Nous pouvons utiliser des méthodes d'intégration plus précises qu'un schéma
#' d'Euler explicite pour résoudre notre équation différentielle. Par exemple, un
#' schéma de Runge et Kutta serait plus approprié.
#' - Nous n'avons aucune idée de la robustesse par rapport à notre jeu d'observation.
#' Le modèle est-il valable sur d'autres données (par exemple correspondant à une
#' saison différente) ? Il est probable que non. Pour avoir une méthode d'identification
#' plus robuste, il est usuel d'introduire un terme de pénalité L2 sur les paramètres
#' dans l'algorithme de moindre carré non-linéaire.
#' - Enfin, nous savons que les différents paramètres identifiés (volume, aire,
#' etc) ne sont qu'une approximation. Plus qu'une simple valeur numérique, nous
#' serions intéressé d'avoir un intervalle de confiance autour de ces valeurs.
#' C'est tout l'enjeu des méthodes d'identification de paramètres bayésiennes.
#' Nous invitons les étudiants intéressés par ce dernier point à suivre un cours
#' de problèmes inverses pour approfondir ces notions.

#' ## Question 4
#' Les paramètres identifiés, nous sommes maintenant en mesure de poser
#' notre problème d'optimisation. Analysons les variables d'optimisation
#' à prendre en compte :
#'
#' - le chauffage $P$ ;
#' - la température intérieure $T^i$ ;
#'
#' sous les contraintes
#'
#' - de dynamique ;
#' - de bornes sur le chauffage: $0 \leq P_k \leq \overline{P}$, supposé ici
#' constante en fonction du temps;
#' - de bornes sur la température intérieure $\underline{T}^i_{k} \leq T^i_k \leq
#' \overline{T}_k^i$, ici dépendantes du temps (on peut chauffer moins la pièce durant
#' la nuit par exemple).
#'
#' Nous n'avons pas besoin de borner les variations du chauffage entre deux
#' pas de temps, dans la mesure où un radiateur électrique a largement
#' le temps de démarrer en une demi-heure (notre pas de discrétisation $\Delta$).
#'
#' Au final, le modèle s'écrit:
#' $$
#' \begin{aligned}
#' \min_{P, T^i} \; & \Delta \sum_{k=0}^T c_k P_k \\
#'      \text{s.c.} & \quad T^i_{k+1} = f_k(T^i_k, p) & \forall k \\
#'                  & \quad 0 \leq P_k \leq \overline{P} & \forall k \\
#'                  & \quad \underline{T}^i_{k} \leq T^i_k \leq \overline{T}_k^i & \forall k
#' \end{aligned}
#' $$
#' où $f_k$ est la fonction de dynamique que nous avions identifié:
#' $$
#' f(T_k^i, p) = T^i_k + \dfrac{p_1 \,\Delta }{c_{air}} (p_2 (T^e_k - T^i_k) + p_3 \Phi_s^k )
#' $$
