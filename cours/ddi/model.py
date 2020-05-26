import numpy as np
from numpy.random import binomial
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def discrete_simulation(T, n, β, γ, s0=999, i0=1, r0=0):
    states = np.zeros((T, 3), dtype=np.int)
    states[0, :] = [s0, i0, r0]
    for t in range(T-1):
        s, i, r = states[t, :]
        contagions = binomial(s, β / n * i)
        removals = binomial(i, γ)
        states[t+1, 0] = s - contagions
        states[t+1, 1] = i + contagions - removals
        states[t+1, 2] = r + removals
    return states

states_1 = discrete_simulation(800, 1000, 0.1, 0.01)
states_2 = discrete_simulation(800, 1000, 0.1, 0.05)
states_3 = discrete_simulation(800, 1000, 0.01, 0.01)
states_4 = discrete_simulation(800, 1000, 0.02, 0.01)

nsimu = 100
for i in range(nsimu):
    states = discrete_simulation(800, 1000, 0.1, 0.01)
    plt.plot(states, lw=.2)

def continuous_simulation(T, n, beta, gamma):
    def sir(y, t):
        dy1 = - beta / n * y[0] * y[1]
        dy2 =   beta / n * y[0] * y[1] - gamma * y[1]
        dy3 = gamma * y[1]
        return [dy1, dy2, dy3]
    y0 = [n-1, 1, 0]
    t = np.linspace(0, T, T+1)
    sol = odeint(sir, y0, t)
    return sol

sol = continuous_simulation(800, 1000, 0.1, 0.01)
