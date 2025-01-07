import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import logging

# Configuración de logging
logging.basicConfig(filename="/logging/unidor_estruct.log", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

def load_base_structure(mol_file_path):
    """Cargar la molécula base desde un archivo .mol."""
    if not os.path.exists(mol_file_path):
        logging.error(f"El archivo {mol_file_path} no existe.")
        return None
    
    mol = Chem.MolFromMolFile(mol_file_path)
    if mol is None:
        logging.error("No se pudo cargar la molécula base.")
    else:
        logging.info("Molécula base cargada correctamente.")
    
    return mol

def load_fragments(excel_file_path):
    """Leer los fragmentos desde un archivo Excel."""
    if not os.path.exists(excel_file_path):
        logging.error(f"El archivo {excel_file_path} no existe.")
        return None
    
    # Leer el Excel y obtener los fragmentos
    df = pd.read_excel(excel_file_path)

    print(f"Contenido del archivo Excel:\n{df}")
    
    # Supongo que los fragmentos están en la primera columna del Excel
    fragments = df.iloc[:, 0].dropna().tolist()
    
    logging.info(f"Se han cargado {len(fragments)} fragmentos.")
    
    return fragments

# Definir rutas de archivo
mol_file = "C:\\Linux\\001-endo-E-R-TS_MDL.mol"
excel_file = "C:\\Linux\\fragments.xlsx"

# Cargar estructura base
base_mol = load_base_structure(mol_file)

# Cargar fragmentos
fragments = load_fragments(excel_file)

# Mostrar los resultados cargados
if base_mol:
    logging.info(f"Molécula base contiene {base_mol.GetNumAtoms()} átomos.")

if fragments:
    logging.info(f"Primer fragmento: {fragments[0]}")

