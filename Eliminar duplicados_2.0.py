import os
import shutil

def main():
    # Ruta del directorio principal
    directorio = r'C:\Linux'

    # Crear la carpeta "eliminar" si no existe
    carpeta_eliminar = os.path.join(directorio, 'eliminar')
    if not os.path.exists(carpeta_eliminar):
        os.makedirs(carpeta_eliminar)

    # Leer los nombres de archivo a preservar desde a.txt
    nombres_a_preservar = set()
    with open(os.path.join(directorio, 'a.txt'), 'r') as archivo_a:
        for linea in archivo_a:
            nombres_a_preservar.add(linea.strip())

    # Recorrer todos los archivos en el directorio principal
    for item in os.listdir(directorio):
        ruta_item = os.path.join(directorio, item)
        if os.path.isfile(ruta_item) and item != 'a.txt' and not item.endswith('.txt'):
            nombre_archivo, extension = os.path.splitext(item)
            # Si el nombre de archivo no está en la lista de nombres a preservar, moverlo a la carpeta "eliminar"
            if nombre_archivo not in nombres_a_preservar:
                shutil.move(ruta_item, os.path.join(carpeta_eliminar, item))

    print("Proceso completado.")

if __name__ == "__main__":
    main()
