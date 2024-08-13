import os
import shutil

# Ruta de la carpeta donde se encuentran los archivos
carpeta_base = "C:\\Linux"

# Obtener una lista de todos los archivos en la carpeta base
archivos = os.listdir(carpeta_base)

# Crear un diccionario para mantener un seguimiento de las carpetas creadas
carpetas_creadas = {}

# Recorrer cada archivo
for archivo in archivos:
    # Obtener los primeros tres caracteres del nombre del archivo
    primeros_tres = archivo[:3]
    
    # Verificar si ya se ha creado una carpeta con estos tres caracteres
    if primeros_tres not in carpetas_creadas:
        # Si no existe, crear la carpeta
        nueva_carpeta = os.path.join(carpeta_base, primeros_tres)
        os.makedirs(nueva_carpeta)
        
        # Agregar la carpeta al diccionario
        carpetas_creadas[primeros_tres] = nueva_carpeta
    
    # Mover el archivo a la carpeta correspondiente
    shutil.move(os.path.join(carpeta_base, archivo), carpetas_creadas[primeros_tres])

print("Terminó")