import glob
import os
import shutil
import re

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

# Expresión regular que permite cualquier cantidad de saltos de línea, espacios o tabulaciones entre los caracteres de "NImag=1"
pattern = re.compile(r"N\s*I\s*m\s*a\s*g\s*=\s*1")

# Verifica si los archivos .log contienen la frase objetivo y los mueve a la carpeta correspondiente
for file in glob.glob("*.log"):
    with open(file, "r") as old:
        content = old.read()

    # Buscar cualquier variante de "NImag=1" cortada por saltos de línea, espacios o tabulaciones
    if re.search(pattern, content):
        move_file_to_folder(file, "Frecuencias Negativas")
