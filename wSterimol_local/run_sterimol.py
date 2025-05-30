# run_sterimol.py
import os
from parse_log import extract_coordinates_and_energy
from sterimol import sterimol_from_coords  # Debes adaptar esto
from boltzmann import boltzmann_weights

def main(folder):
    results = []
    for fname in os.listdir(folder):
        if fname.endswith('.log'):
            fpath = os.path.join(folder, fname)
            coords, energy = extract_coordinates_and_energy(fpath)
            L, Bmin, Bmax = sterimol_from_coords(coords)
            results.append((fname, energy, L, Bmin, Bmax))

    energies = [r[1] for r in results]
    weights = boltzmann_weights(energies)

    print("\n=== Individual Sterimol Values ===")
    for i, (fname, energy, L, Bmin, Bmax) in enumerate(results):
        print(f"{fname:<20} E = {energy:.2f} kcal/mol  L = {L:.2f}  Bmin = {Bmin:.2f}  Bmax = {Bmax:.2f}  wt = {weights[i]:.2f}%")

    # Weighted averages
    wL = sum(r[2] * weights[i]/100 for i, r in enumerate(results))
    wBmin = sum(r[3] * weights[i]/100 for i, r in enumerate(results))
    wBmax = sum(r[4] * weights[i]/100 for i, r in enumerate(results))

    print("\n=== Boltzmann-Weighted Averages ===")
    print(f"L = {wL:.2f}  Bmin = {wBmin:.2f}  Bmax = {wBmax:.2f}")

if __name__ == '__main__':
    folder = "C:\\Linux"
    main(folder)
