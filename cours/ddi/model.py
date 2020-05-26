import numpy as np
from numpy.random import binomial
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def discrete_simulation(T, n, beta, gamma, s0=999, i0=1, r0=0):
    states = np.zeros((T, 3), dtype=np.int)
    states[0, :] = [s0, i0, r0]
    for t in range(T-1):
        s, i, r = states[t, :]
        contagions = binomial(s, beta / n * i)
        removals = binomial(i, gamma)
        states[t+1, 0] = s - contagions
        states[t+1, 1] = i + contagions - removals
        states[t+1, 2] = r + removals
    return states

beta = 0.1
gamma = 0.01
states_1 = discrete_simulation(800, 1000, beta, gamma)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(states_1[:, 0], lw=2.0, label="S")
ax.plot(states_1[:, 1], lw=2.0, label="I")
ax.plot(states_1[:, 2], lw=2.0, label="R")
ax.legend()
ax.set_ylabel("Number of cases")
ax.set_xlabel("Timesteps")
ax.set_title(f"Epidemic, beta={beta} gamma={gamma}")
fig.savefig("simu_1.pdf")


def run_simulation(nsimu, beta=0.1, gamma=0.01, outname="simu.pdf"):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_ylabel("Number of cases")
    ax.set_xlabel("Timesteps")
    ax.set_title(f"Epidemic, beta={beta} gamma={gamma}")
    for i in range(nsimu):
        states = discrete_simulation(800, 1000, beta, gamma)
        ax.plot(states[:, 0], lw=.2, c="darkblue")
        ax.plot(states[:, 1], lw=.2, c="darkred")
        ax.plot(states[:, 2], lw=.2, c="darkgreen")
    fig.savefig(outname)
    fig.clear()

nsimu = 10
run_simulation(nsimu, beta=beta, gamma=gamma, outname="simu_2.pdf")

beta = 0.01
gamma = 0.01
run_simulation(nsimu, beta=beta, gamma=gamma, outname="simu_3.pdf")

beta = 0.02
gamma = 0.01
run_simulation(nsimu, beta=beta, gamma=gamma, outname="simu_4.pdf")

beta = 0.1
gamma = 0.05
run_simulation(nsimu, beta=beta, gamma=gamma, outname="simu_5.pdf")


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

beta = 0.1
gamma = 0.01
sol = continuous_simulation(800, 1000, beta, gamma)
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(sol[:, 0], lw=2.0, label="S")
ax.plot(sol[:, 1], lw=2.0, label="I")
ax.plot(sol[:, 2], lw=2.0, label="R")
ax.legend()
ax.set_ylabel("Number of cases")
ax.set_xlabel("Timesteps")
ax.set_title(f"Epidemic, beta={beta} gamma={gamma}")
fig.savefig("simu_6.pdf")

