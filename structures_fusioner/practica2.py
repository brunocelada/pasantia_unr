from rdkit import Chem
import os
from rdkit.Chem import AllChem, Draw

def draw_molecule(mol, sufix=""):
    img_path = "C:\\Linux\\test"
    # Crear el directorio si no existe
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    img = Draw.MolToImage(mol, size=(300, 300))
    img.save(f"C:\\Linux\\test\\new_molecule_{sufix}.png")

def smiles_to_mol_3d(smiles):
    fragment_mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(fragment_mol) # Generar geometría inicial
    try:
        AllChem.UFFOptimizeMolecule(fragment_mol)  # Optimización con UFF
        print("Fragmento optimizado correctamente.")
    except Exception as e:
        print(f"Error durante la optimización del fragmento: {e}")
    return fragment_mol

def get_marked_atoms(mol):
    # Devuelve la cantidad de átomos marcados
    counter = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            counter = counter + 1
    
    print(f"N° Atomos marcados: {counter}")
    return counter

def remove_marked_atoms(mol):
    for atom in mol.GetAtoms():    
        if atom.GetSymbol() == "*": 
            mol.RemoveAtom(atom.GetIdx())
            return mol
    
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
        raise ValueError("Error en la función get_index_of_neighbors_atom")

def combine_and_link_molecules(base_mol, fragment_mol):
    # Eliminar los hidrógenos antes de la combinación
    base_mol_no_h = Chem.RemoveHs(base_mol, sanitize=False)
    fragment_mol_no_h = Chem.RemoveHs(fragment_mol)

    # Combinamos las moléculas sin los hidrógenos
    combined = Chem.CombineMols(base_mol_no_h, fragment_mol_no_h)
    combined_rw = Chem.RWMol(combined)

    draw_molecule(combined_rw, sufix=0) #------------

    # Obtener los indexes de los átomos vecinos a los marcados para unirlos
    neighbor_idx_list = get_neighbors_marked_atoms(combined_rw)
    # Agregar un enlace simple entre ellos
    combined_rw.AddBond(neighbor_idx_list[0], neighbor_idx_list[1], Chem.BondType.SINGLE)


    draw_molecule(combined_rw, sufix=1)#------------

    # Obtiene la cantidad de átomos marcados con (*) o (X)
    marked_atoms = get_marked_atoms(combined_rw)

    # Elimina los átomos marcados
    for i in range (marked_atoms):
        combined_rw = remove_marked_atoms(combined_rw)

        draw_molecule(combined_rw, sufix=2+i)#------------

    print("Correcta combinación")
    
    return combined_rw

def optimize_3d_structure(mol):
    """
    Genera una estructura 3D optimizada de la molécula combinada.
    """
    # Crear una copia editable para asegurarnos de que los cambios no afecten la estructura original
    mol_3d = Chem.RWMol(mol)
    mol_3d = Chem.AddHs(mol_3d)  # Agregar hidrógenos explícitos al final
    
    # Aquí se ajustan las cargas formales después de la modificación de la estructura.
    for atom in mol_3d.GetAtoms():
        if atom.GetSymbol() == "N" and atom.GetExplicitValence() == 4:
            atom.SetFormalCharge(1)  # Asignar una carga positiva
            print(f"Ajustada la carga formal en el nitrógeno (Índice {atom.GetIdx()})")

    # Embedding 3D ignorando errores de valencia
    try:
        AllChem.EmbedMolecule(mol_3d, ignoreSmoothingFailures=True)
        # AllChem.UFFOptimizeMolecule(mol_3d)
        print("Estructura 3D optimizada correctamente.")
    except Chem.AtomValenceException as e:
        print(f"Error de valencia al optimizar: {e}")
    
    return mol_3d

def optimize_fragment_only(mol, base_indices):
    # Convertimos a una molécula editable para realizar la optimización
    # mol = Chem.AddHs(mol)

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N" and atom.GetExplicitValence() == 4:
            atom.SetFormalCharge(1)  # Asignar una carga positiva
            print(f"Ajustada la carga formal en el nitrógeno (Índice {atom.GetIdx()})")
        elif atom.GetSymbol() == "C" and atom.GetExplicitValence() == 5:
            atom.SetNoImplicit(True)  # Evita agregar hidrógenos adicionales
            print(f"Átomo de carbono en índice {atom.GetIdx()} temporalmente ajustado para sobrevalencia.")

    mol = Chem.AddHs(mol)

    # Añadir hidrógenos explícitamente
    mol_with_hs = Chem.AddHs(mol)
    draw_molecule(combined_structure, "asdasd")#------------

    # Validar la molécula (opcional, pero recomendado)
    if not mol_with_hs:
        raise ValueError("La molécula no es válida después de añadir hidrógenos.")
    else:
        print("Bien los hidroenos")
    
    force_field = AllChem.UFFGetMoleculeForceField(mol)
    # Aplicar restricciones de posición a los átomos de la estructura base
    for idx in base_indices:
        force_field.UFFAddPositionConstraint(idx, maxDisplacement=0.0001, forceConstant=100.0)

    try:
        force_field.Minimize()  # Optimizar la molécula
        print("Optimización completada con restricciones.")
    except Exception as e:
        print(f"Error durante la optimización con restricciones: {e}")
    return mol

def define_type_structure(mol, type_value):
    for atom in mol.GetAtoms():
        atom.SetProp("type", type_value)
    return mol

def freeze_structure(mol):
    mol_rw = Chem.RWMol(mol)
    mol_rw = define_type_structure(mol_rw, "base")

    return mol_rw

def get_prop_indexes(mol, type_value):
    indexes = []
    for atom in mol.GetAtoms():
        if atom.HasProp("type") and atom.GetProp("type") == type_value:
            indexes.append(atom.GetIdx())
    print (f"Indexes: {indexes}")
    return indexes


# ----------------------------------------------------------------------
base_mol = Chem.MolFromMolFile("C:\\Linux\\001-endo-E-R-TS.mol", sanitize=False, removeHs=True)

base_mol = freeze_structure(base_mol)

fragment_mol = smiles_to_mol_3d("C1CCCC(C1)*")  # ejemplo 

combined_structure = combine_and_link_molecules(base_mol, fragment_mol)

draw_molecule(combined_structure, "a")#------------

final_structure_1 = optimize_3d_structure(combined_structure)

base_indexes = get_prop_indexes(combined_structure, "base")
final_structure_2 = optimize_fragment_only(combined_structure, base_indexes)

draw_molecule(final_structure_1, "OP1")#------------
draw_molecule(final_structure_2, "OP2")#------------



def generar_archivo_gjc(mol, filename, carpeta_destino="C:\\Linux\\test"):
    # Crear el directorio si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    # Definir la ruta completa donde se guardará el archivo .gjc
    filepath = os.path.join(carpeta_destino, filename)

    # Escribir el archivo .gjc
    with open(filepath, "w") as f:
        for atom in mol.GetAtoms():
            atom_type = atom.GetAtomicNum()
            conf = mol.GetConformer()
            x, y, z = conf.GetAtomPosition(atom.GetIdx())
            f.write(f" {atom_type:3d}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")

generar_archivo_gjc(final_structure_1, "prueba.xyz")




# def load_and_sanitize_base(base_mol):
#     """
#     Sanitiza la molécula base ignorando errores de valencia excesiva,
#     útil para moléculas con átomos cargados.
#     """
#     try:
#         # Intentamos sanitizar con una configuración flexible
#         Chem.SanitizeMol(base_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
#         print("Sanitización completada sin errores.")
#     except Chem.AtomValenceException:
#         print("Se encontró un error de valencia en la molécula. Sanitización parcial realizada.")
#         # Realizamos una sanitización básica ignorando el error de valencia
#         Chem.SanitizeMol(base_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    
#     return base_mol




# from rdkit import Chem
# from rdkit.Chem import AllChem

# def combine_molecules(base_mol, fragment_smiles):
#     # Desactiva la sanitización inicial para evitar errores de valencia
#     # base_mol = Chem.Mol(base_mol, sanitize=False)
    
#     # Sanitiza manualmente, ignorando los errores de valencia
#     Chem.SanitizeMol(base_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    
#     # Convertimos el fragmento SMILES en un Mol
#     fragment_mol = Chem.MolFromSmiles(fragment_smiles)
#     fragment_mol = Chem.AddHs(fragment_mol)
#     AllChem.EmbedMolecule(fragment_mol)
    
#     # Encuentra el átomo marcado en cada molécula
#     atom_idx_base = None
#     atom_idx_fragment = None
    
#     for atom in base_mol.GetAtoms():
#         if atom.GetSymbol() == "*":
#             atom_idx_base = atom.GetIdx()
#             break

#     for atom in fragment_mol.GetAtoms():
#         if atom.GetSymbol() == "*":
#             atom_idx_fragment = atom.GetIdx()
#             break
    
#     # Asegúrate de que los átomos marcados sean válidos
#     if atom_idx_base is None or atom_idx_fragment is None:
#         raise ValueError("No se encontró un átomo marcado (*) en la base o el fragmento.")
    
#     # Elimina los átomos marcados (*) antes de unir
#     rw_base = Chem.RWMol(base_mol)
#     rw_fragment = Chem.RWMol(fragment_mol)
#     rw_base.RemoveAtom(atom_idx_base)
#     rw_fragment.RemoveAtom(atom_idx_fragment)
    
#     # Combina las moléculas en una sola
#     combined = Chem.CombineMols(rw_base, rw_fragment)
#     combined_rw = Chem.RWMol(combined)
    
#     # Agrega el enlace de unión
#     base_idx = atom_idx_base
#     fragment_idx = atom_idx_fragment + rw_base.GetNumAtoms() - 1  # Ajusta el índice
    
#     # Verifica que no exista un enlace entre los átomos antes de agregarlo
#     if not combined_rw.GetBondBetweenAtoms(base_idx, fragment_idx):
#         combined_rw.AddBond(base_idx, fragment_idx, Chem.BondType.SINGLE)
    
#     # Optimiza la estructura 3D de la molécula combinada
#     combined_3d = Chem.MolFromSmiles(Chem.MolToSmiles(combined_rw))
#     AllChem.EmbedMolecule(combined_3d)
#     AllChem.UFFOptimizeMolecule(combined_3d)
    
#     return combined_3d

# # Ejemplo de uso
# file_path = "C:\\Linux\\001-endo-E-R-TS.mol"  # Ruta de la estructura base
# base_mol = Chem.MolFromMolFile(file_path, sanitize=False, removeHs=False)

# # Fragmento en formato SMILES
# fragment_smiles = "CC(=O)O*"  # Fragmento ejemplo con un átomo de unión marcado

# # Ensamblaje de la nueva estructura
# combined_3d = combine_molecules(base_mol, fragment_smiles)


# from rdkit.Chem import Draw
# # Guardar la imagen de la molécula en un archivo PNG
# img = Draw.MolToImage(combined_3d, size=(300, 300))
# img.save("new_molecule.png")



# from rdkit import Chem
# from rdkit.Chem import AllChem

# def generar_archivo_gjc(mol, filename='mol.gjc', optimizacion='UFF'):

#     # Agregar hidrógenos si no están presentes
#     mol = Chem.AddHs(mol)

#     # Crear coordenadas 3D para la molécula (si no tiene)
#     if mol.GetNumConformers() == 0:
#         if optimizacion == 'ETKDG':
#             AllChem.EmbedMolecule(mol, randomSeed=42, method=AllChem.ETKDG())
#         else:
#             AllChem.EmbedMolecule(mol)
    
#     # Optimización usando UFF, MMFF94 o ETKDG
#     if optimizacion == 'UFF':
#         AllChem.UFFOptimizeMolecule(mol)
#     elif optimizacion == 'MMFF94':
#         # Verificar si la molécula tiene los parámetros MMFF94
#         if AllChem.MMFFHasAllMoleculeParams(mol):
#             AllChem.MMFFOptimizeMolecule(mol)
#         else:
#             print("La molécula no tiene todos los parámetros necesarios para MMFF94")
#     elif optimizacion == 'ETKDG':
#         # Se puede usar directamente con EmbedMolecule
#         pass

#     # Escribir el archivo .gjc (igual que antes)
#     with open(filename, 'w') as f:
#         f.write('%chk=mol.chk\n')
#         f.write('#P B3LYP/6-31G(d) Opt\n\n')
#         f.write('Molecular Geometry\n\n')
#         f.write('0 1\n')
        
#         for atom in mol.GetAtoms():
#             atom_type = atom.GetAtomicNum()
#             conf = mol.GetConformer()
#             x, y, z = conf.GetAtomPosition(atom.GetIdx())
#             f.write(f' {atom_type:3d}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n')

#         f.write('\n')

# # Crear una molécula de ejemplo (por ejemplo, agua)
# mol = Chem.MolFromSmiles('O')

# # Llamar a la función para generar el archivo .gjc con optimización MMFF94
# generar_archivo_gjc(mol, 'agua_mmff94.gjc', optimizacion='MMFF94')



# Practica de generar fragmentos como imágenes
    # from rdkit import Chem
    # from rdkit.Chem import AllChem
    # from rdkit.Chem import Draw

    # def save_3D_structures(smiles_list):
    #     """Guarda la estructura 3D de fragmentos en formato SMILES como imágenes.
    #     Args:
    #         smiles_list (list of str): Lista de SMILES en formato string.
    #     """
    #     for idx, smiles in enumerate(smiles_list, start=1):
    #         # Convertir SMILES a una molécula RDKit
    #         mol = Chem.MolFromSmiles(smiles)
    #         if mol is None:
    #             print(f"Error al convertir SMILES: {smiles}")
    #             continue

    #         # Verificar que el átomo de unión (*) esté en la estructura
    #         attachment_points = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
    #         if not attachment_points:
    #             print(f"No se encontró un sitio de unión (*) en el SMILES: {smiles}")
    #             continue

    #         # Añadir hidrógenos y generar coordenadas 3D
    #         # mol = Chem.AddHs(mol)
    #         # if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
    #         #     print(f"Error al generar estructura 3D para: {smiles}")
    #         #     continue
    #         # AllChem.UFFOptimizeMolecule(mol)

    #         # Guardar la molécula en un archivo de imagen
    #         img = Draw.MolToImage(mol, size=(300, 300))
    #         img.save(f"./structures/fragment_{idx}.png")
    #         print(f"Estructura 3D guardada como fragment_{idx}.png")

    # # Ejemplo de uso
    # smiles_list = ["C1CCCC(C1)*", "c1c3c4c5c2c1c6c7c8c2", "c1cc(ccc1)*"]
    # save_3D_structures(smiles_list)

# PRACTICA PARA CAMBIAR CARGA A ÁTOMOS INDIVIDUALES DE ESTRUCTURA
# from rdkit import Chem

# def load_molecule_with_custom_charges(file_path):
#     # Cargar el archivo .mol sin sanitizar
#     mol = Chem.MolFromMolFile(file_path, sanitize=False)
    
#     # Sanitización limitada de propiedades básicas
#     Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    
#     # Ajustar cargas en función del tipo de átomo y número de enlaces
#     for atom in mol.GetAtoms():
#         symbol = atom.GetSymbol()
#         degree = atom.GetTotalDegree()
        
#         # Reglas para ajustar la carga en función del tipo de átomo
#         if symbol == "N" and degree == 4:
#             atom.SetFormalCharge(1)  # Nitrógeno con cuatro enlaces (carga +1)
#         elif symbol == "O" and degree == 3:
#             atom.SetFormalCharge(1)  # Oxígeno con tres enlaces (carga +1)
#         elif symbol == "P" and degree == 5:
#             atom.SetFormalCharge(1)  # Fósforo con cinco enlaces (carga +1)
#         elif symbol == "S" and degree == 6:
#             atom.SetFormalCharge(2)  # Azufre con seis enlaces (carga +2)
#         # Añadir aquí otras reglas según los átomos y cargas esperadas
    
#     return mol

# # Ruta al archivo .mol
# file_path = '/mnt/data/archivo.mol'
# mol = load_molecule_with_custom_charges(file_path)

# # Visualización para verificar la estructura y las cargas
# from rdkit.Chem import Draw
# display(Draw.MolToImage(mol, size=(300, 300)))
