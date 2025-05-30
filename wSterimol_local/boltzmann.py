# boltzmann.py
import math

def boltzmann_weights(energies):
    min_e = min(energies)
    kT = 0.593  # kcal/mol at 298K
    weights = [math.exp(-(e - min_e) / kT) for e in energies]
    total = sum(weights)
    norm_weights = [w / total * 100 for w in weights]
    return norm_weights
