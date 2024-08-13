import glob
import os
import shutil

# Cambia el directorio de trabajo a C:\linux
os.chdir("C:\\linux")

# Función para mover archivos a una carpeta específica si no existe y luego eliminarlos del directorio original
def move_file_to_folder(file_path, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    shutil.copy(file_path, folder_name)
    os.remove(file_path)

# Verifica el contenido de los archivos .log y los mueve a la carpeta correspondiente si no terminan correctamente
for file in glob.glob("*.log"):
    with open(file, "r") as old:
        # Leer las últimas tres líneas del archivo
        lines = old.readlines()[-3:]

    # Verificar si alguna de las últimas tres líneas contiene "Normal termination"
    if not any("Normal termination" in line for line in lines):
        move_file_to_folder(file, "Termino Mal")

# Verifica si los archivos .log contienen frases específicas y los mueve a la carpeta correspondiente
target_phrases = ["NImag=1", "NImag\n=1", "NIma\ng=1", "NI\nmag=1"]
for file in glob.glob("*.log"):
    with open(file, "r") as old:
        content = old.read().replace("\n", "")

    if any(phrase in content for phrase in target_phrases):
        move_file_to_folder(file, "Frecuencias Negativas")

print("Terminó bien")