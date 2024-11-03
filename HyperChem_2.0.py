#!/usr/bin/python
# comentario

import glob
import os
import shutil
import sys
from os.path import splitext

os.chdir("C:\\Linux")

# Set global variables
start_dict = 0
end_dict = 0
conf_number = 1
number_atom = []
start_conf = 0
end_conf = 0

# Receive the HyperChem input file
conformational = "a.hcs"

# Open the conformational search from HyperChem file in read mode
with open(conformational, "r") as open_conformational:
    rline = open_conformational.readlines()

# Find the beginning and end for the dictionary that links each atom to the number
for n in range(len(rline)):
    if "mol 1" in rline[n]:
        if len(rline[n]) < 7:
            start_dict = n + 1
        elif "endmol 1" in rline[n]:
            end_dict = n
            break

# Create the dictionary linking the number to the atom in the way "number:atom"
for line in rline[start_dict:end_dict]:
    words = line.split()
    number_atom.append(words[3])

# Define the length of the molecule
len_mol = end_dict - start_dict

# Process each conformation
for n in range(end_dict, len(rline)):
    if "[Conformation " + str(conf_number) + "]" in rline[n]:
        new_conformation = str(splitext(conformational)[0]) + "_C" + str(conf_number) + ".gjc"
        with open(new_conformation, "w") as open_conformation:
            open_conformation.write("0 1 \n")
            for j, v in zip(range((n+7), (n+7+len_mol)), range(len(number_atom))):
                line = rline[j].lower()
                splitted = line.split("=")
                xyz = splitted[1].split()
                x = format(float(xyz[0]), '.8f')
                y = format(float(xyz[1]), '.8f')
                z = format(float(xyz[2]), '.8f')
                open_conformation.write(number_atom[v] + ' ')
                open_conformation.write(x + ' ')
                open_conformation.write(y + ' ')
                open_conformation.write(z + '\n')
        conf_number += 1
        n += 7 + len_mol

# Ask user for prefix to rename files
prefijo = input("Ingrese prefijo: _")

# Get the list of .gjc files
listing = glob.glob("C:\\Linux\\*.gjc")

# Rename files with the prefix
for file in listing:
    os.rename(file, file.replace("a_", prefijo + "_", 1))

# After renaming, update the listing to the new filenames
listing = glob.glob("C:\\Linux\\*.gjc")

# Ask the user for input for the XX values
nprocshared = input("%nprocshared=? ")
mem = input("%mem=? ")
comment = input("Comment (# XX)=? ")

# Now add the lines at the beginning of each .gjc file
for file in listing:
    with open(file, "r") as f:
        content = f.read()

    # Create the new header
    new_header = f"%nprocshared={nprocshared}\n"
    new_header += f"%mem={mem}\n\n"
    new_header += f"# {comment}\n\n"
    new_header += "comment\n\n"

    # Write the new header followed by the original content
    with open(file, "w") as f:
        f.write(new_header + content)

    # Add 3 blank lines at the end of the file
    with open(file, "a") as f:
        f.write("\n\n\n")

print("Proceso completado con éxito.")
