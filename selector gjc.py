import glob
import os
import shutil

# Cambiar el directorio actual a C:/Linux
os.chdir("C:/Linux")

# Obtener la lista de archivos con extensión .out en el directorio
listing = glob.glob("*.out")

# Renombrar todos los archivos .out a .log
for file in listing:
    os.rename(file, (file.rsplit(".", 1)[0]) + ".log")

# Contar la cantidad de archivos .log
counter = len(glob.glob("*.log"))

# Crear y abrir el archivo a.txt en modo escritura
with open("a.txt", "w") as new:
    # Iterar sobre todos los archivos .log y escribir el nombre base en a.txt
    for file in glob.glob("*.log"):
        print((file.rsplit(".", 1)[0]), file=new)

# Lista para almacenar los nombres de archivos con extensión .gjc
logsgjc = []

# Leer el archivo a.txt y crear los nombres con extensión .gjc
with open("a.txt") as filter:
    filters = filter.readlines()
    for i in filters:
        logsgjc.append(i.strip() + ".gjc")

# Renombrar archivos .gjc a .gjf
for file in logsgjc:
    if os.path.exists(file):  # Asegurarse de que el archivo exista
        newfile = file.replace('.gjc', '.gjf')
        os.rename(file, newfile)

# Crear la carpeta Relanzar si no existe y mover archivos .gjc allí
for file in glob.glob("*.gjc"):
    if not os.path.exists("Relanzar"):
        os.makedirs("Relanzar")
    shutil.copy(file, 'C:/Linux/Relanzar')
    os.remove(file)

# Renombrar archivos .gjf de vuelta a .gjc
for file in glob.glob("*.gjf"):
    newfile = file.replace('.gjf', '.gjc')
    os.rename(file, newfile)

print("Terminó")