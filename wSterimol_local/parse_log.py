# parse_log.py
import os
import re
import sys

def extract_coordinates_and_energy(logfile):
    coords = []
    energy = None
    with open(logfile, 'r') as f:
        lines = f.readlines()

    # Parse energy
    for line in lines:
        if 'SCF Done' in line:
            match = re.search(r'SCF Done:  E\(\w+\) =\s*(-?\d+\.\d+)', line)
            if match:
                energy = float(match.group(1))
    
    # Parse final coordinates
    start = False
    for i, line in enumerate(lines):
        if 'Standard orientation:' in line:
            coords = []
            start = True
            continue
        if start and '---------------------------------------------------------------------' in line:
            start = False
        elif start and line.strip().split() and line.strip().split()[0].isdigit():
            parts = line.strip().split()
            atom_number = int(parts[1])
            x, y, z = map(float, parts[3:6])
            coords.append((atom_number, x, y, z))
    
    return coords, energy

if __name__ == '__main__':
    coords, energy = extract_coordinates_and_energy(sys.argv[1])
    print("Energy:", energy)
    for atom in coords:
        print(atom)
