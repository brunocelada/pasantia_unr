import glob
import charge_changer
import os
import shutil
from os.path import splitext

def procesing_hcs_files(BASE_DIR):
    # Receive the HyperChem input files
    for filename in os.listdir(BASE_DIR):
        conf_number = 1  # Reset counter for each .HCS file
        number_atom = []  # Initialize list for each file
        
        # Open the conformational search from HyperChem file in read mode
        if filename.endswith(".HCS"):
            with open(os.path.join(BASE_DIR, filename), "r") as open_conformational:
                rline = open_conformational.readlines()

            # Find the beginning and end for the dictionary that links each atom to the number
            start_dict, end_dict = 0, 0
            for n, line in enumerate(rline):
                if "mol 1" in line:
                    if len(line) < 7:
                        start_dict = n + 1
                    elif "endmol 1" in line:
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
                    new_conformation = os.path.join(BASE_DIR, f"{splitext(filename)[0]}_c{conf_number}.gjc")
                    with open(new_conformation, "w") as open_conformation:
                        open_conformation.write("0 1 \n")
                        for j, v in zip(range((n + 7), (n + 7 + len_mol)), range(len(number_atom))):
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

def creating_gjc_files(base_dir):
    # Update the listing to the new filenames
    listing = glob.glob(os.path.join(base_dir, "*.gjc"))

    # Ask the user for input for the XX values
    nprocshared = input("%nprocshared= ")
    mem = input("%mem= ")
    command_line = input("command_line= ")

    # Add the lines at the beginning of each .gjc file
    for file in listing:
        with open(file, "r") as f:
            content = f.read()

        # Create the new header
        new_header = f"%nprocshared={nprocshared}\n"
        new_header += f"%mem={mem}\n"
        new_header += f"{command_line}\n\n"
        new_header += "comment\n\n"

        # Write the new header followed by the original content
        with open(file, "w") as f:
            f.write(new_header + content)

        # Add 3 blank lines at the end of the file
        with open(file, "a") as f:
            f.write("\n\n\n")

# Function to move .HCS files to the "Conformations HCS" folder
def move_hcs_files(base_dir):
    # Define the destination directory
    destination_dir = os.path.join(base_dir, "Conformations HCS")
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Move each .HCS file to the destination folder
    for filename in os.listdir(base_dir):
        if filename.endswith(".HCS"):
            source_path = os.path.join(base_dir, filename)
            destination_path = os.path.join(destination_dir, filename)
            shutil.move(source_path, destination_path)
            # print(f"Moved {filename} to {destination_dir}")

# Ejecución de funciones en orden necesario
def main():
    BASE_DIR = r"C:\Linux"

    procesing_hcs_files(BASE_DIR)
    creating_gjc_files(BASE_DIR)
    move_hcs_files(BASE_DIR)

    # Llama la función main del script "charge_changer" para cambiar las cargas que son
    # creadas incorrectamente en este script.
    charge_changer.main()

    print("\nFinalizado\n")

if __name__ == "__main__":
    main()
