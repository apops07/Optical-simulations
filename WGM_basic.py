import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

# parameters
m = 1.45  # Refractive index of resonator
l = 10    # Angular mode number (can be changed)
k = 2 * np.pi / 1.55  # Wavenumber for λ = 1.55 µm

# TE and TM modes
def te_mode(x):
    return x * sp.jv(l, x) - l * sp.jv(l-1, x)  # Bessel function for TE modes


def tm_mode(x):
    return x * sp.jv(l, x) - l * (sp.jv(l-1, x) - (m**2 - 1) * x * sp.jvp(l-1, x))  # TM mode

# Finding resonance
x_vals = np.linspace(0.1, 20, 1000)  # Range for roots
te_roots = np.where(np.diff(np.sign(te_mode(x_vals))))[0]
tm_roots = np.where(np.diff(np.sign(tm_mode(x_vals))))[0]

te_resonances = x_vals[te_roots]
tm_resonances = x_vals[tm_roots]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x_vals, te_mode(x_vals), label="TE Mode", linestyle="dashed")
plt.plot(x_vals, tm_mode(x_vals), label="TM Mode", linestyle="solid")
plt.scatter(te_resonances, [0]*len(te_resonances), color='red', label="TE Resonances", marker="o")
plt.scatter(tm_resonances, [0]*len(tm_resonances), color='blue', label="TM Resonances", marker="x")
plt.axhline(0, color="black", linewidth=0.5)
plt.legend()
plt.xlabel("x (size parameter)")
plt.ylabel("Mode Function Value")
plt.title("TE & TM Mode Resonances in a WGM Resonator")
plt.grid(True)
plt.show()

# Print approximate resonance points
print("TE Mode Resonances:", te_resonances)
print("TM Mode Resonances:", tm_resonances)
