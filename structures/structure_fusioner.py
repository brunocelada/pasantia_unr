# ---------------------- STRUCTURE FUSIONER -------------------------- #
# ---------------------------------------------------- by Bruno Celada #

'''Este script automatiza el reemplazo de un átomo marcado en un esqueleto
base por múltiples fragmentos señalados.

La estructura base debe estar en formato .mol, y el átomo marcado para
sustituirse debe estar marcado con el número "0".

Los fragmentos deben estar en formato SMILE, marcados con asterisco (*), 
en una lista en la primer columna de un archivo excel. Este archivo debe
ser el único excel en la carpeta base (ésta puede cambiarse, pero 
generalmente es "C:\\Linux"), y los smiles deben comenzar desde la celda 
A2 hacia abajo.

Recordar colocar correctamente los enlaces (simples, dobles, etc) en los
archivos tipo .mol para generar correctamente las estructuras.
Antes de generar los archivos nuevos, se recomienda ejecutar la pre-visualización.
'''

# More info ---------------------------------------------------------- #

# * is wildcard (any atom). The wildcard atom may also be written without brackets.


# SCRIPT ------------------------------------------------------------- #

import logging 
import subprocess
import sys
import glob
import os
from IPython.display import display

# Configuración de logging
logging.basicConfig(filename="registros/unidor_estruct.log", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

# -------------------------------------------------------------------- #
'''Instala librerías básicas para la ejecución del programa
    rdkit: biblioteca para la química computacional y el procesamiento 
        de información química
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

# -------------------------------------------------------------------- # 
# Funciones

def get_excel_file(carpeta_base):
    """Busca el archivo Excel en la carpeta base"""
    for file in os.listdir(carpeta_base):
        if file.endswith(".xlsx"):
            return os.path.join(carpeta_base, file)
    
    logging.info(f"Error buscando excel_file.")
    return None

def get_fragments_from_excel(excel_file_path):
    """Lee los fragmentos en formato SMILES desde el Excel"""
    # Leer la primer columna con los fragmentos en formato SMILES
    # LA CELDA A1 NO DEBE TENER UN FRAGMENTO
    df = pd.read_excel(excel_file_path)
    fragments_smiles = df.iloc[:, 0].dropna().tolist()

    logging.info(f"Se han cargado {len(fragments_smiles)} fragmentos.")
    return fragments_smiles

def save_3D_image(mol, prefix, idx):
    img_path = "C:\\Linux\\pre_vis"
    # Crear el directorio si no existe
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # Guardar la molécula en un archivo de imagen
    img = Draw.MolToImage(mol, size=(300, 300))
    img.save(f"{img_path}\\{prefix}_{idx}.png")
    logging.info(f"Estructura 3D guardada como fragment_{idx}.png")

def obtain_3D_fragments(smiles_list):
    """Guarda la estructura 3D de fragmentos en formato SMILES como tipo Mol.
    Args:
        smiles_list (list of str): Lista de SMILES en formato string con sitios de unión.
    """
    fragments_mol = []

    for idx, smiles in enumerate(smiles_list, start=1):
        # Convertir SMILES a una molécula RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.info(f"Error al convertir SMILES: {smiles}")
            continue

        # Verificar que el átomo de unión (*) esté en la estructura
        attachment_points = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
        if not attachment_points:
            logging.info(f"No se encontró un sitio de unión (*) en el SMILES: {smiles}")
            continue
        else:
            fragments_mol.append(mol)

        # Añadir hidrógenos y generar coordenadas 3D
        # mol = Chem.AddHs(mol)
        # if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
        #     logging.info(f"Error al generar estructura 3D para: {smiles}")
        #     continue
        # AllChem.UFFOptimizeMolecule(mol)

    return fragments_mol
        # Guardar la molécula en un archivo de imagen
        

def get_structures_from_mol(carpeta_base):
    """Obtiene estructuras de archivos .mol"""
    structures = {}

    for filename in os.listdir(carpeta_base):
        try:
            if filename.endswith(".mol"):
                filepath = os.path.join(carpeta_base, filename)
                
                # Leer la molécula desde el archivo .mol
                # removeHS=True para eliminar los hidrógenos
                # sanitize=False para desactivar la sanitización automática, permite leer átomos con valencias no estándar
                mol = Chem.MolFromMolFile(filepath, removeHs=True, sanitize=False)

                # Ajustar cargas en función del tipo de átomo y número de enlaces
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    degree = atom.GetTotalDegree()
                    
                    # Reglas para ajustar la carga en función del tipo de átomo
                    if symbol == "N" and degree == 4:
                        atom.SetFormalCharge(1)  # Nitrógeno con cuatro enlaces (carga +1)
                    elif symbol == "O" and degree == 3:
                        atom.SetFormalCharge(1)  # Oxígeno con tres enlaces (carga +1)
                    elif symbol == "P" and degree == 5:
                        atom.SetFormalCharge(1)  # Fósforo con cinco enlaces (carga +1)
                    elif symbol == "S" and degree == 6:
                        atom.SetFormalCharge(2)  # Azufre con seis enlaces (carga +2)
                    # Añadir acá otras reglas según los átomos y cargas esperadas

                if mol is None:
                    logging.info(f"Error al leer el archivo {filename}")
                    continue

                logging.info(f"Archivo {filename} leído correctamente.")
                structures[(filename.rsplit(".", 1)[0])] = mol

            else:
                logging.info(f"El archivo {filename} no es de tipo .mol, se omite.")

        except Exception as e:
                logging.error(f"Error leyendo archivo .mol {filename}: {e}")

    return structures

def generar_archivo_xyz(mol, filename, optimizacion="UFF", carpeta_destino="."):
    '''Función para generar un archivo .xyz por cada estructura tipo Mol
    Parámetros: 
        mol= estructura de tipo Mol
        filename= nombre del archivo .xyz
        optimizacion= método de optimización en RDKit. Si la molécula tiene carga, usar "None", porque seguro salta error.

    UFF: Método general, bueno para moléculas con diversos tipos de enlaces y heteroátomos.
    MMFF94: Más preciso para moléculas orgánicas, especialmente con enlaces covalentes.
    ETKDG: Utilizado para generar conformaciones 3D de moléculas orgánicas con base en datos experimentales.
    MMFF94s: Versión mejorada de MMFF94, menos utilizada y no tan documentada en RDKit.
    Conjugate Gradient (CG): Un enfoque matemático iterativo, aunque no directamente implementado en RDKit para optimización de moléculas grandes.
    '''

    # Crear el directorio si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    # Definir la ruta completa donde se guardará el archivo .xyz
    filepath = os.path.join(carpeta_destino, filename)

    # Agregar hidrógenos si no están presentes
    mol = Chem.AddHs(mol)

    # Crear coordenadas 3D para la molécula (si no tiene)
    if mol.GetNumConformers() == 0:
        if optimizacion == "ETKDG":
            AllChem.EmbedMolecule(mol, randomSeed=42, method=AllChem.ETKDG())
        else:
            AllChem.EmbedMolecule(mol)
    
    # Optimización usando UFF, MMFF94 o ETKDG
    if optimizacion == None:
        pass
    elif optimizacion == "UFF":
        AllChem.UFFOptimizeMolecule(mol)
    elif optimizacion == "MMFF94":
        # Verificar si la molécula tiene los parámetros MMFF94
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            logging.info("La molécula no tiene todos los parámetros necesarios para MMFF94. Se utilizará UFF")
            AllChem.UFFOptimizeMolecule(mol)
    elif optimizacion == "ETKDG":
        # Se puede usar directamente con EmbedMolecule
        pass

    # Escribir el archivo .xyz
    with open(filepath, "w") as f:        
        for atom in mol.GetAtoms():
            atom_type = atom.GetAtomicNum()
            conf = mol.GetConformer()
            x, y, z = conf.GetAtomPosition(atom.GetIdx())
            f.write(f" {atom_type:3d}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")

        f.write("\n\n\n")

    logging.info(f"Archivo .xyz guardado en: {filepath}")

# ------------------------------------------------------------------------------------------------------------------------------------ #
# Main program
def main():
    # Registrar un nuevo lanzamiento del script
    logging.info("\n\n------- NEW STRUCTURE FUSIONER -------\n")

    carpeta_base = "C:\\Linux"

    # Obtener la ruta del archivo excel
    excel_file = get_excel_file(carpeta_base)
    if (excel_file == None):
        print("Error buscando el archivo excel con los smiles\n")
        logging.info("Error buscando el archivo excel con los smiles\n")

    else:
        # Obtener los fragmentos en SMILE de la primer columna del excel
        fragments_smiles = get_fragments_from_excel(excel_file)
        fragments_mol = obtain_3D_fragments(fragments_smiles)
        logging.info(f"Listado de fragmentos: {fragments_smiles}. Cada fragmento es tipo string.")

        # Obtener una lista con las estructuras de archivos .mol
        structures = get_structures_from_mol(carpeta_base)

        # Opción de pre-visualización
        pre_vis = input("¿Desea realizar la pre-visualización? (guardando estructuras y fragmentos en formato png) (y/n) ").lower()
        if (pre_vis=="y"):
            for idx, structure in structures.items():
                save_3D_image(structure, "structure", idx) # Generación de imágenes de las estructuras
            for idx in range(1, len(fragments_mol)):
                save_3D_image(fragments_mol[idx - 1], "fragment", idx) # Generación de imágenes de fragmentos
        
        # Opción de unión de estructuras
        ready = input("¿Está listo para unir las estructuras y generar los .xyz? (y/n) ").lower()
        if (ready=="y"):
            # Parámetros para la creación del archivo .xyz
            opt_type=None

            for filename in structures:
                generar_archivo_xyz(structures[filename], f"{filename}.xyz", optimizacion=opt_type, carpeta_destino=carpeta_base)
            # # Por cada estructura, modificarla creando una nueva por cada fragmento
            # modified_structures = modify_structures(structures, fragments)
            # print(modified_structures)

        print("\nEnd of sustitution\n")

if __name__ == "__main__":
    main()
