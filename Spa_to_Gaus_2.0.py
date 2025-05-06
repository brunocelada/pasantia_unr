import os
import shutil
import gzip
import subprocess
import charge_changer
import Spa_Gaus_CreateSH_Slack_G09_16

# Constantes
BASE_DIR = r"C:\Linux"
SEVEN_ZIP_PATH = r"C:\Program Files\7-Zip\7z.exe"

# Funciones para renombrar archivos
def rename_files():
    """Renombra los archivos eliminando '.Conf.M0001' del nombre."""
    for filename in os.listdir(BASE_DIR):
        if filename.endswith(".spartan") and ".Conf.M0001" in filename:
            new_filename = filename.replace(".Conf.M0001", "")
            old_path = os.path.join(BASE_DIR, filename)
            new_path = os.path.join(BASE_DIR, new_filename)
            os.rename(old_path, new_path)

def rename_input_files():
    """Renombra los archivos 'Input.gz' según la estructura de carpetas."""
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file == "Input.gz":
                folder_hierarchy = os.path.relpath(root, BASE_DIR).split(os.sep)
                folder_prefix = "".join(folder_hierarchy[-3:]) if len(folder_hierarchy) >= 3 else "".join(folder_hierarchy)
                
                old_file_path = os.path.join(root, file)
                new_file_name = f"{folder_prefix}{file}"
                new_file_path = os.path.join(root, new_file_name)
                
                os.rename(old_file_path, new_file_path)

def rename_and_copy_gz_files():
    """Copia y renombra archivos '.gz' de acuerdo a reglas específicas."""
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.gz') and 'Input' in file:
                original_path = os.path.join(root, file)
                shutil.copy2(original_path, BASE_DIR)

                new_name = file.replace("Input", "").replace("Molecules", "_")
                new_name = new_name.replace("M000", "c").replace("M00", "c").replace("M0", "c")

                copied_path = os.path.join(BASE_DIR, file)
                new_path = os.path.join(BASE_DIR, new_name)
                os.rename(copied_path, new_path)

# Funciones para extracción y manipulación de archivos
def extract_files():
    """Extrae archivos '.spartan' en carpetas con el mismo nombre."""
    for filename in os.listdir(BASE_DIR):
        if filename.endswith(".spartan"):
            folder_name = os.path.splitext(filename)[0]
            folder_path = os.path.join(BASE_DIR, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            file_path = os.path.join(BASE_DIR, filename)
            # Redirigir stdout y stderr a subprocess.PIPE para suprimir la salida
            subprocess.run([SEVEN_ZIP_PATH, 'x', file_path, '-o' + folder_path],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def extract_and_cleanup_gz_files():
    """Extrae archivos '.gz', cambia la extensión a '.gjc' y limpia las carpetas."""
    # Eliminar archivos no '.gz'
    for file in os.listdir(BASE_DIR):
        file_path = os.path.join(BASE_DIR, file)
        if os.path.isfile(file_path) and not file.endswith('.gz'):
            os.remove(file_path)

    # Extraer archivos '.gz'
    for file in os.listdir(BASE_DIR):
        if file.endswith('.gz'):
            gz_path = os.path.join(BASE_DIR, file)
            extracted_path = os.path.join(BASE_DIR, file[:-3])  # Remueve '.gz' para obtener el nombre original
            
            with gzip.open(gz_path, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            os.remove(gz_path)
            
            # Cambiar extensión a '.gjc'
    for file in os.listdir(BASE_DIR):
        file_path = os.path.join(BASE_DIR, file)
        if os.path.isfile(file_path):
            new_path = os.path.join(BASE_DIR, os.path.splitext(file)[0] + '.gjc')
            os.rename(file_path, new_path)

    # Eliminar carpetas
    for folder in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)

def update_gjc_files():
    """Actualiza los archivos '.gjc' con nuevos valores y estructura."""
    files = [f for f in os.listdir(BASE_DIR) if f.endswith('.gjc')]

    valor_a1 = input('nprocshared= ')
    valor_a2 = input('mem=: ')

    # Menú para seleccionar la command line
    print("Seleccione una opción para la command line:")
    print("1) optb-normal")
    print("2) DP4")
    print("3) DP4+")
    print("4) ML")
    print("5) Custom")
    
    opcion = input('Opción: ').strip()

    if opcion == "1":
        valor_a3 = "# B3LYP/6-31G* opt freq=noraman"
    elif opcion == "2":
        valor_a3 = "# B3LYP/6-31G** nmr"
    elif opcion == "3":
        valor_a3 = "# mPW1PW91/6-31+G** nmr scrf=(pcm,solvent=chloroform)"
    elif opcion == "4":
        valor_a3 = "# rhf/STO-3G nmr pop=nbo"
    elif opcion == "5":
        valor_a3 = "# " + input('Custom command line: # ')
    else:
        print("Opción no válida. Usando opción Custom por defecto.")
        valor_a3 = input('Custom command line: # ')
        # B3LYP/6-31G* opt(TS,calcfc,noeigentest) freq=noraman

    for file in files:
        file_path = os.path.join(BASE_DIR, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if len(lines) >= 2:
            lines[0] = f'%nprocshared={valor_a1}\n'
            lines[1] = f'%mem={valor_a2}\n'
            lines.insert(2, f'{valor_a3}\n')
            lines.insert(3, '\n')
            lines.insert(4, 'comment\n')
            lines.insert(5, '\n')

        for i in range(1, len(lines)):
            if 'ENDCART' in lines[i]:
                lines = lines[:i]
                lines.extend(['\n', '\n', '\n'])
                break

        with open(file_path, 'w') as f:
            f.writelines(lines)

    print('Done')

# Ejecución de funciones en orden necesario
def main():
    rename_files()
    extract_files()
    rename_input_files()
    rename_and_copy_gz_files()
    extract_and_cleanup_gz_files()

    update_gjc_files()
    create_sh = input("Crear archivos .sh? (y/n): ").strip().lower()
    
    if create_sh == "y":
        # Crea los archivos .sh con el script "Spa_Gaus_CreateSH_Slack_G09_16"
        Spa_Gaus_CreateSH_Slack_G09_16.main()
        

    
    change_charge = input("Cambiar carga/multip? (y/n): ").strip().lower()
    # Llama la función main del script "charge_changer" para cambiar las cargas que son
    # creadas incorrectamente en este script.
    if change_charge == "y":
        charge_changer.main()

    # Recordatorio
    print("\nAcordate de eliminar el sufijo 'spartan' en caso de ser necesario")

    print("\nFinalizado\n")

if __name__ == "__main__":
    main()
