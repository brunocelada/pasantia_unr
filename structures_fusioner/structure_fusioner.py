# ---------------------- STRUCTURE FUSIONER -------------------------- #
# ---------------------------------------------------- by Bruno Celada #

'''Este script automatiza el reemplazo de un átomo marcado en un esqueleto
base por múltiples fragmentos señalados. Los archivos generados son de tipo
.xyz

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
import shutil
import numpy as np

# Configuración de logging
logging.basicConfig(filename="registros/unidor_estruct.log", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

# -------------------------------------------------------------------- #
'''Instala librerías básicas para la ejecución del programa
    rdkit: biblioteca para la química computacional y el procesamiento
        de información química
    pandas: biblioteca que permite utilizar archivos Excel
    py3Dmol: biblioteca para visualizar moléculas 3D
    IPython: biblioteca para visualizar imágenes
'''
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDepictor, rdMolTransforms, Draw
    from rdkit.Chem.rdMolTransforms import (
    GetBondLength, GetAngleDeg, GetDihedralDeg,
    SetBondLength, SetAngleDeg, SetDihedralDeg
    )
    logging.info("rdkit ya está instalado.")
except ImportError:
    print("rdkit no está instalado. Instalando...")
    logging.info("rdkit no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit"])
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDepictor, rdMolTransforms, Draw
    from rdkit.Chem.rdMolTransforms import (
    GetBondLength, GetAngleDeg, GetDihedralDeg,
    SetBondLength, SetAngleDeg, SetDihedralDeg
    )

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

    if df.empty or df.shape[1] < 1:
        raise ValueError("El Excel no contiene fragmentos en la primer columna.")

    fragments_smiles = df.iloc[:, 0].dropna().tolist()

    logging.info(f"Se han cargado {len(fragments_smiles)} fragmentos.")
    return fragments_smiles

def save_3D_image(mol, prefix, img_path, idx=""):
    # Crear el directorio si no existe
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # Convertir a Mol normal y calcular coords 2D
    mol2 = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol2)

    # Guardar la molécula en un archivo de imagen
    img = Draw.MolToImage(mol2, size=(300, 300))
    img.save(os.path.join(img_path, f"{prefix}_{idx}.png"))
    logging.info(f"Estructura 3D guardada como {prefix}_{idx}.png")

def get_marked_atoms(mol):
    # Devuelve la cantidad de átomos marcados
    counter = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            counter = counter + 1

    logging.info(f"N° Atomos marcados: {counter} en {mol}")
    return counter

def remove_marked_atoms(mol):
    mol_rw = Chem.RWMol(mol)
    for atom in mol_rw.GetAtoms():
        if atom.GetSymbol() == "*":
            mol_rw.RemoveAtom(atom.GetIdx())
            return mol_rw

def get_neighbors_marked_atoms(mol):
    '''
    Marca con una propiedad a los átomos vecinos a los marcados con (*) o (X)
    '''
    idxs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            # Obtiene el primer vecino del átomo marcado
            neighbors = atom.GetNeighbors()
            if neighbors:
                idxs.append(neighbors[0].GetIdx())
            else:
                print("El átomo marcado no tiene vecinos.")
                idxs.append(atom.GetIdx())

    if len(idxs) == 2:
        return idxs
    elif len(idxs) == 1:
        raise ValueError("Solo se encontró 1 átomo marcado")
    elif len(idxs) == 0:
        raise ValueError("No se encontró (*)")
    else:
        raise ValueError("Error en la función get_neighbors_marked_atoms")

# -------------------------------------------------------------------- #
# Funciones para obtener estructuras 3D de fragmentos y archivos .mol, unirlos
# y generar archivos .xyz

def fragment_info(mol):
    """Calcula información estructural de un fragmento dado como Mol."""
    frag_info = {
        'atom_types': [],
        'distances': {},
        'angles': {},
        'dihedrals': {}
    }
    
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return frag_info  # Devuelve vacío si no hay átomos

    conf = mol.GetConformer()

    # Guardamos los tipos de átomos
    for atom in mol.GetAtoms():
        frag_info['atom_types'].append(atom.GetSymbol())

    # Distancias
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            d = rdMolTransforms.GetBondLength(conf, i, j)
            frag_info['distances'][(i,j)] = d

    # Ángulos
    if num_atoms >= 3:
        for i in range(num_atoms):
            for j in range(num_atoms):
                for k in range(num_atoms):
                    if i != j and j != k and i != k:
                        try:
                            a = rdMolTransforms.GetAngleDeg(conf, i, j, k)
                            frag_info['angles'][(i,j,k)] = a
                        except:
                            pass

    # Dihedros
    if num_atoms >= 4:
        for i in range(num_atoms):
            for j in range(num_atoms):
                for k in range(num_atoms):
                    for l in range(num_atoms):
                        if len(set([i,j,k,l])) == 4:
                            try:
                                d = rdMolTransforms.GetDihedralDeg(conf, i, j, k, l)
                                frag_info['dihedrals'][(i,j,k,l)] = d
                            except:
                                pass

    return frag_info


def obtain_3D_fragments(smiles_list):
    """Guarda la estructura 3D de fragmentos en formato SMILES como tipo Mol.
    Args:
        smiles_list (list of str): Lista de SMILES en formato string con sitios de unión.
    """
    fragments_mol = []

    for idx, fragment_smile in enumerate(smiles_list, start=1):
        # Convertir SMILES a una molécula RDKit
        mol = Chem.MolFromSmiles(fragment_smile)
        if mol is None:
            logging.info(f"Error al convertir SMILES: {fragment_smile}")
            continue

        # Verificar que el átomo de unión (*) esté en la estructura
        attachment_points = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
        if not attachment_points:
            logging.info(f"No se encontró un sitio de unión (*) en el SMILES: {fragment_smile}")
            continue
        else:
            '''Este bloque se usa para optimizar la geometría de los fragmentos.
            Sin embargo, se desaconseja si los fragmentos están marcados, ya que optimiza con errores.'''
            # try:
            #     AllChem.EmbedMolecule(mol) # Generar geometría inicial
            #     AllChem.UFFOptimizeMolecule(mol)  # Optimización con UFF
            #     if AllChem.EmbedMolecule(mol) != 0:
            #         raise ValueError(f"Error embebiendo fragmento: {fragment_smile}")
            #     if AllChem.UFFOptimizeMolecule(mol) != 0:
            #         print("Warning: no se pudo optimizar el fragmento")
            #         logging.info("Warning: no se pudo optimizar el fragmento")
            #     logging.info("Fragmento optimizado correctamente.")
            # except Exception as e:
            #     print(f"Error durante la optimización del fragmento: {e}")
            #     logging.info(f"Error durante la optimización del fragmento: {e}")

            fragments_mol.append(mol)

    return fragments_mol

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
                if mol is None:
                    logging.info(f"Error al leer el archivo {filename}")
                    continue

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

def generar_archivos_xyz(mol, filename, optimizacion="UFF", carpeta_destino="."):
    '''Función para generar un archivo .xyz por cada estructura tipo Mol. También genera
        una estructura 3D optimizada de la molécula combinada.
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

    # Crear una copia editable para asegurarnos de que los cambios no afecten la estructura original
    mol_3d = Chem.RWMol(mol)
    mol_3d = Chem.AddHs(mol_3d)  # Agregar hidrógenos explícitos al final

    # Crear coordenadas 3D para la molécula (si no tiene)
    if mol_3d.GetNumConformers() == 0:
        if optimizacion == "ETKDG":
            AllChem.EmbedMolecule(mol_3d, randomSeed=42, method=AllChem.ETKDG())
        else:
            AllChem.EmbedMolecule(mol_3d)

    # Optimización usando UFF, MMFF94 o ETKDG
    if optimizacion == None:
        pass
    elif optimizacion == "UFF":
        AllChem.UFFOptimizeMolecule(mol_3d)
    elif optimizacion == "MMFF94":
        # Verificar si la molécula tiene los parámetros MMFF94
        if AllChem.MMFFHasAllMoleculeParams(mol_3d):
            AllChem.MMFFOptimizeMolecule(mol_3d)
        else:
            logging.info("La molécula no tiene todos los parámetros necesarios para MMFF94. Se utilizará UFF")
            AllChem.UFFOptimizeMolecule(mol_3d)
    elif optimizacion == "ETKDG":
        # Se puede usar directamente con EmbedMolecule
        pass

    # Embedding 3D ignorando errores de valencia
    try:
        AllChem.EmbedMolecule(mol_3d, ignoreSmoothingFailures=True)
        logging.info("Estructura 3D optimizada correctamente.")
    except Chem.AtomValenceException as e:
        logging.info(f"Error de valencia al optimizar: {e}")

    # Escribir el archivo .xyz
    with open(filepath, "w") as f:
        for atom in mol_3d.GetAtoms():
            atom_type = atom.GetAtomicNum()
            conf = mol_3d.GetConformer()
            x, y, z = conf.GetAtomPosition(atom.GetIdx())
            f.write(f" {atom_type:3d}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")

        f.write("\n\n\n")

    logging.info(f"Archivo .xyz guardado en: {filepath}")

def combine_and_link_molecules(base_mol, fragment_mol, fragment_info):
    """
    Combina base_mol y fragment_mol, añade el enlace y ajusta geometría según frag_geom.
    Aplica solo las transformaciones disponibles (distance, angle, dihedral).

    Args:
        base_mol (Chem.Mol): estructura base con conformador 3D.
        fragment_mol (Chem.Mol): fragmento con conformador 3D.
        frag_geom (dict): geometría extraída con measure_fragment_geometry().

    Returns:
        Chem.Mol: Mol combinado con transformaciones aplicadas.
    """

    test_path = "C:\\Linux\\test" # <-------------------------

    # Eliminar los hidrógenos antes de la combinación
    base = Chem.RemoveHs(base_mol, sanitize=False)
    frag = Chem.RemoveHs(fragment_mol)

    # Combinamos las moléculas sin los hidrógenos
    combo = Chem.CombineMols(base, frag)
    rw = Chem.RWMol(combo)

    save_3D_image(rw, "test", test_path, idx=0) #------------

    # Obtener los indexes de los átomos vecinos a los marcados para unirlos
    neighbor_idx_list = get_neighbors_marked_atoms(rw)

    # Crear enlace simple entre la estructua base y el fragmento
    rw.AddBond(neighbor_idx_list[0], neighbor_idx_list[1], Chem.BondType.SINGLE)

    save_3D_image(rw, "test", test_path, idx=1)#------------

    # Obtiene la cantidad de átomos marcados con (*) o (X)
    marked_atoms = get_marked_atoms(rw)

    # Elimina los átomos marcados
    for i in range (marked_atoms):
        rw = remove_marked_atoms(rw)

        save_3D_image(rw, "test", test_path, idx=2+i)#------------

    logging.info(f"Correcta combinación de {base_mol} y {fragment_mol}.")

    return rw

# ------------------------------------------------------------------------------------------------------------------------------------ #
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
        while pre_vis not in ["y", "n"]:
            pre_vis = input("Seleccione una opción válida. ¿Desea realizar la pre-visualización? (guardando estructuras y fragmentos en formato png) (y/n) ").lower()

        if (pre_vis=="y"):
            for idx, structure in structures.items():
                save_3D_image(structure, "structure", "C:\\Linux\\pre_vis", idx) # Generación de imágenes de las estructuras
            for idx in range(1, len(fragments_mol)):
                save_3D_image(fragments_mol[idx - 1], "fragment", "C:\\Linux\\pre_vis", idx) # Generación de imágenes de fragmentos
            print("\nPre-visualización completa. Las podés revisar la carpeta pre_vis antes de continuar\n")

        # Opción de unión de estructuras
        ready = input("¿Está listo para unir las estructuras y generar los archivos .xyz? (y/n) ").lower()
        while ready not in ["y", "n"]:
            ready = input("Seleccione una opción válida. ¿Está listo para unir las estructuras y generar los archivos .xyz? (y/n) ").lower()

        if (ready=="y"):
            # Parámetros para la creación del archivo .xyz
            opt_type=None # CAMBIAR A UFF O MMFF94 DEPENDIENDO EL TIPO DE ESTRUCTURA <---------------

            nombre_carpeta_destino = input("\nNombre de la carpeta destino de las nuevas combinaciones (dejar en blanco para usar la misma carpeta base): ")
            carpeta_destino = carpeta_base
            if nombre_carpeta_destino != "":
                carpeta_destino = carpeta_base + "\\" + nombre_carpeta_destino

            # Por cada estructura, modificarla creando una nueva por cada fragmento
            i=0
            for base_mol in structures:
                for fragment_mol in fragments_mol:
                    frag_info = fragment_info(fragment_mol)
                    combined_structure = combine_and_link_molecules(structures[base_mol], fragment_mol, frag_info)
                    save_3D_image(combined_structure, "combined", carpeta_destino, i) # <-----------------
                    i+=1
                    # generar_archivos_xyz(combined_structure, f"{base_mol}.xyz", optimizacion=opt_type, carpeta_destino=carpeta_destino)

            if (pre_vis=="y"):
                shutil.rmtree(f"{carpeta_base}\\pre_vis")
                print("\nSe removió la carpeta pre-vis.")
        elif (ready=="n"):
            print("\nSe mantuvo la carpeta pre-vis para revisar errores.")

        print("\nEND OF STRUCTURE_FUSIONER.PY\n")

if __name__ == "__main__":
    main()
