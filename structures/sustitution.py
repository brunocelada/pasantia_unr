import logging 
import subprocess
import sys
import glob
import os

# Configuración de logging
logging.basicConfig(filename="./registros/script.log", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

# ------------------------------------------------------------------------------------------------------------------------------------ # 
'''Instala librerías básicas para la ejecución del programa
        rdkit: biblioteca para la química computacional y el procesamiento de información química
        pandas: biblioteca que permite utilizar archivos Excel
        py3Dmol: biblioteca para visualizar moléculas 3D
'''
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    logging.info("rdkit ya está instalado.")
except ImportError:
    print("rdkit no está instalado. Instalando...")
    logging.info("rdkit no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit"])
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw

try:
    import pandas as pd
    logging.info("pandas ya está instalado.")
except ImportError:
    print("pandas no está instalado. Instalando...")
    logging.info("pandas no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

try:
    import py3Dmol
    logging.info("py3Dmol ya está instalado.")
except ImportError:
    print("py3Dmol no está instalado. Instalando...")
    logging.info("py3Dmol no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "py3Dmol"])
    import py3Dmol

# ------------------------------------------------------------------------------------------------------------------------------------ # 
# Funciones

def get_excel_file(carpeta_base):
    """Busca el archivo Excel en la carpeta base"""
    for file in os.listdir(carpeta_base):
        if file.endswith('.xlsx'):
            return os.path.join(carpeta_base, file)
    
    logging.info(f"Error buscando excel_file.")
    return None

# Función para mostrar la molécula en 3D
def mostrar_molecula_3D(mol):
    mb = Chem.MolToMolBlock(mol)
    p = py3Dmol.view(width=400, height=400)
    p.addModel(mb, 'mol')
    p.setStyle({'stick': {}})
    p.zoomTo()
    return p.show()

def get_structures_from_mol(carpeta_base):
    """Obtiene estructuras de archivos .mol"""
    structures = {}

    for filename in os.listdir(carpeta_base):
        try:
            if filename.endswith(".mol"):
                filepath = os.path.join(carpeta_base, filename)
                
                # Leer la molécula desde el archivo .mol
                mol = Chem.MolFromMolFile(filepath, removeHs=False)
                if mol is None:
                    logging.info(f"Error al leer el archivo {filename}")
                    continue
                
                logging.info(f"Archivo {filename} procesado y optimizado correctamente.")
                structures[filename] = mol

            else:
                logging.info(f"El archivo {filename} no es de tipo .mol, se omite.")

        except Exception as e:
                logging.error(f"Error leyendo archivo .mol {filename}: {e}")

    return structures

def get_fragments_from_excel(excel_file_path):
    """Lee los fragmentos en formato SMILES desde el Excel"""
    # Leer la columna con los fragmentos en formato SMILES desde la primera columna
    df = pd.read_excel(excel_file_path)
    fragments = df.iloc[:, 0].dropna().tolist()  # Asume que la primera columna contiene los fragmentos

    logging.info(f"Se han cargado {len(fragments)} fragmentos.")
    return fragments

def modify_structures(structures, fragments):
    """Modifica las estructuras uniendo los fragmentos en el átomo marcado con '*' en SMILES."""
    modified_structures = {}

    for file_name, structure in structures.items():
        mostrar_molecula_3D(structure)
        # El átomo "marcado" es el átomo con AtomicNum 0 en RDKit
        atom_idx_base = next((atom.GetIdx() for atom in structure.GetAtoms() if atom.GetAtomicNum() == 0), None)

        if atom_idx_base is None:
            logging.error(f"No se encontró átomo marcado en la estructura: {file_name}")
            continue

        for fragment in fragments:
            try:
                # Convertir el fragmento SMILES a una molécula
                frag = Chem.MolFromSmiles(fragment) 

                if frag is None:
                    logging.error(f"Fragmento inválido: {fragment}")
                    continue

                # Obtener el átomo marcado con [*] en el SMILES
                atom_idx_fragment = next((atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0), None)
                
                if atom_idx_fragment is None:
                    logging.error(f"No se encontró átomo marcado en el fragmento: {fragment}")
                    continue

                # Combina la estructura base con el fragmento
                combined_mol = Chem.CombineMols(structure, frag)
                
                # Creación de un EditableMol para añadir el enlace entre la base y el fragmento
                combined_edmol = Chem.EditableMol(combined_mol)
                combined_edmol.AddBond(atom_idx_base, atom_idx_fragment + structure.GetNumAtoms(), Chem.BondType.SINGLE)
                
                # Obtener la nueva molécula combinada
                combined_mol = combined_edmol.GetMol()
                
                # Generar una conformación 3D
                AllChem.EmbedMolecule(combined_mol)
                
                # Aplicar restricciones: los átomos de la estructura base no se optimizan
                fixed_atoms = list(range(structure.GetNumAtoms()))  # Mantener fijos los átomos de la estructura base
                params = AllChem.UFFGetMoleculeForceField(combined_mol)
                
                # Fijar las posiciones de los átomos de la estructura base
                for idx in fixed_atoms:
                    params.AddFixedPoint(idx)

                # Optimiza solo los átomos del fragmento    
                if AllChem.UFFOptimizeMolecule(combined_mol, forcefield=params) != 0:
                    logging.error(f"Error optimizando molécula combinada: {file_name}")
                    continue

                modified_structures[file_name] = combined_mol

                logging.info(f"Fragmento {fragment} añadido a la estructura {file_name}.")

            except Exception as e:
                    logging.error(f"Error añadiendo fragmento {fragment} a {file_name}: {e}")

    return modified_structures

def write_structures_to_gjc(carpeta_base, modified_structures, header):
    """Escribe las estructuras modificadas a archivos .gjc."""
    os.chdir(carpeta_base)

    for i in range(len(modified_structures)):

        for file_name, structure in modified_structures.items():
            with open(f"{file_name}{i}.gjc", 'w') as f:
                f.write(header)
        
                conf = structure.GetConformer()
                for atom in structure.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    f.write(f"{atom.GetSymbol()} {pos.x:.9f} {pos.y:.9f} {pos.z:.9f}\n")
                
                f.write("\n\n\n\n")

# ------------------------------------------------------------------------------------------------------------------------------------ #
# Main program
def main():
    # Registrar un nuevo lanzamiento del script
    logging.info("\n\n-------NEW SUSTITUTION SCRIPT-------\n")

    carpeta_base = "C:\\Linux"

    # Pedir datos para la creación del archivo
    # nprocshared = input("nprocshared= ")
    # mem = input("mem= ")
    # command_line = input("command line= ")
    # charge = input("Cual es la carga? ")
    # mult = input("Cual es la multiplicidad? ")
    # header = (
    #             f"%nprocshared={nprocshared}\n"
    #             f"%mem={mem}\n"
    #             f"{command_line}\n\n"
    #             "comment\n\n"
    #             f"{charge} {mult}\n"
    #         )
    header = (
                f"%nprocshared=8\n"
                f"%mem=200mw\n"
                f"B3LYP\n\n"
                "comment\n\n"
                f"1 1\n"
            )

    # Obtener la ruta del archivo excel
    excel_file = get_excel_file(carpeta_base)
    if (excel_file == None):
        print("Error buscando el archivo excel con los smiles\n")
        

    # Obtener los fragmentos en SMILE de la primer columna del excel
    fragments = get_fragments_from_excel(excel_file)
    print(fragments)

    # Obtener una lista con las estructuras de archivos .mol
    structures = get_structures_from_mol(carpeta_base)
    print(structures)

    # Por cada estructura, modificarla creando una nueva por cada fragmento
    modified_structures = modify_structures(structures, fragments)
    print(modified_structures)

    # Por cada estructura modificada, crear un archivo gjc
    write_structures_to_gjc(carpeta_base, modified_structures, header)

    print("\nEnd of sustitution\n")

if __name__ == "__main__":
    main()
