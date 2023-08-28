import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Praktikum Neuroprothesen
#
# 04) Aktives Kabelneuron

class HHFormulas:
    def __init__(self):
        self.Vm = None
        self.dt = None
        self.k = None
        self.T = 6.3
        self.gK = 36
        self.eK = -12
        self.gNa = 120
        self.eNa = 115
        self.gL = 0.3
        self.eL = 10.6
        self.alpha_m = None
        self.alpha_n = None
        self.alpha_h = None
        self.beta_m = None
        self.beta_n = None
        self.beta_h = None
        self.tau_m = None
        self.tau_n = None
        self.tau_h = None
        self.p_m = None
        self.p_n = None
        self.p_h = None
        self.m = None
        self.n = None
        self.h = None
        self.iK = None
        self.iNa = None
        self.iL = None

    def calc_all(self):
        self.k_fcn()
        self.alpha_m_fcn()
        self.alpha_n_fcn()
        self.alpha_h_fcn()
        self.beta_m_fcn()
        self.beta_n_fcn()
        self.beta_h_fcn()
        self.tau_m_fcn()
        self.tau_n_fcn()
        self.tau_h_fcn()
        self.p_m_fcn()
        self.p_n_fcn()
        self.p_h_fcn()

        if self.m is None:
            self.m = self.p_m
            self.n = self.p_n
            self.h = self.p_h
        else:
            self.m_fcn()
            self.n_fcn()
            self.h_fcn()

        self.ik_fcn()
        self.ina_fcn()
        self.il_fcn()

    def tau_m_fcn(self):
        self.tau_m = 1. / (self.alpha_m + self.beta_m)

    def tau_n_fcn(self):
        self.tau_n = 1. / (self.alpha_n + self.beta_n)

    def tau_h_fcn(self):
        self.tau_h = 1. / (self.alpha_h + self.beta_h)

    def p_m_fcn(self):
        self.p_m = self.alpha_m / (self.alpha_m + self.beta_m)

    def p_n_fcn(self):
        self.p_n = self.alpha_n / (self.alpha_n + self.beta_n)

    def p_h_fcn(self):
        self.p_h = self.alpha_h / (self.alpha_h + self.beta_h)

    def alpha_m_fcn(self):
        self.alpha_m = (2.5 - 0.1 * self.Vm) / (np.exp(2.5 - 0.1 * self.Vm) - 1)

    def beta_m_fcn(self):
        self.beta_m = 4 * np.exp(-self.Vm / 18)

    def alpha_n_fcn(self):
        self.alpha_n = (0.1 - 0.01 * self.Vm) / (np.exp(1 - 0.1 * self.Vm) - 1)

    def beta_n_fcn(self):
        self.beta_n = 0.125 * np.exp(-self.Vm / 80)

    def alpha_h_fcn(self):
        self.alpha_h = 0.07 * np.exp(-self.Vm / 20)

    def beta_h_fcn(self):
        self.beta_h = 1 / (np.exp((3 - 0.1 * self.Vm)) + 1)

    def m_fcn(self):
        self.m = self.m + self.dt * self.k * (self.alpha_m * (1 - self.m) - self.beta_m * self.m)

    def n_fcn(self):
        self.n = self.n + self.dt * self.k * (self.alpha_n * (1 - self.n) - self.beta_n * self.n)

    def h_fcn(self):
        self.h = self.h + self.dt * self.k * (self.alpha_h * (1 - self.h) - self.beta_h * self.h)

    def k_fcn(self):
        self.k = 3 ** (0.1 * (self.T - 6.3))

    def ik_fcn(self):
        self.iK = self.gK * self.n ** 4 * (self.Vm - self.eK)

    def ina_fcn(self):
        self.iNa = self.gNa * self.m ** 3 * self.h * (self.Vm - self.eNa)

    def il_fcn(self):
        self.iL = self.gL * (self.Vm - self.eL)


hh = HHFormulas()

# Define some parameters
hh.dt = 0.01
hh.Vm = 0

# Calculate all
hh.calc_all()

# parameters
g_Na = 120e-3  # conductance Sodium [ms/cm2]
g_K = 39e-3    # conductance Potassium [ms/cm2]
g_L = 0.3e-3   # conductance leak [ms/cm2]

v_Na = 115e-3   # voltage Sodium [mV]
v_K = -12e-3    # voltage Potassium [mV]
v_L = 10.6e-3   # voltage leak [mV]
v_Rest = -70e-3  # resting potential [mV]

cm = 1e-6        # membrane capacity [uF/cm2]

p_Axon = 0.1e3   # density axon [kOhm*cm]
r_Axon = 2e-4    # radius axon [cm]
L_comp = 0.1e-4  # length compartment [cm]

# Praktikum Neuroprothesen
#
# 03) Passives Kabelneuron

# time
dt = 0.01
t = np.arange(dt, 50 + dt, dt)

# compartments
n = 100

# membrane
Cm = 1
Rm = 10

# membrane voltage
Vm = np.zeros((n, len(t)))

# axon
p = 0.1        # conductance for the axon
L = 1e-6       # length for 1 compartment
r = 2e-4       # radius for the axon

# main script formula 2.43
Ra = (p * L) / (np.pi * r ** 2)  # resistance of the axon

# connectivity matrix
c = np.ones((n, 1)) * -2
c[0] = -1
c[-1] = -1

C = np.diag(c[:, 0], 0)
C = C + np.diag(np.ones(n - 1), 1)
C = C + np.diag(np.ones(n - 1), -1)

# part A from exercise_5_slides page 7/9
A = np.eye(n) - (dt / (Cm * Ra)) * C

# stimulus current
Istim = np.zeros((n, len(t)))

# Higher stimulation = more APs
Istim[n // 3, :100] = 30
Istim[n // 3 * 2, :100] = 30

for idx in range(len(t) - 1):
    # calculate all values from class hh_formula
    hh.calc_all()

    # calculate membrane current
    Im = hh.iNa + hh.iK + hh.iL

    # part b from exercise_5_slides page 7/9
    b = Vm[:, idx] + dt / (Cm * Ra) * (-Im + Istim[:, idx])

    # "x"
    Vm[:, idx + 1] = solve(A, b)
    hh.Vm = Vm[:, idx + 1]

# main script figure 2.11 top+botton
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(Vm, aspect='auto', cmap = "plasma")
plt.colorbar()
plt.title('Membrane potential spread [mV]', fontsize=18)
plt.xlabel('Time in ms', fontsize=16)
plt.ylabel('Compartment Nr.', fontsize=16)
#plt.xlim(0, 3000)

plt.subplot(2, 2, 2)
compartments_to_plot = [50, 60, 70, 80]
for comp in compartments_to_plot:
    plt.plot(Vm[comp, :], linewidth=2, label=f'comp # {comp}')
plt.legend()
plt.xlabel('time [ms]', fontsize=16)
plt.ylabel('Vm [mV]', fontsize=16)
plt.title('Time course of the action potentials in selected compartments', fontsize=18)
plt.grid(True)
#plt.xticks(np.arange(0, 3001, 500), np.arange(0, 51, 5))
plt.xlim(0, 3000)

plt.subplot(2, 2, 3)
plt.plot(Vm[:, 700], np.arange(1, 101), linewidth=2, color='black')
plt.xlabel('Vm [mV]', fontsize=16)
plt.ylabel('Compartment no.', fontsize=16)
plt.title('Membrane potential of all compartments at 12 ms.', fontsize=18)
plt.grid(True)

plt.tight_layout()
plt.show()
