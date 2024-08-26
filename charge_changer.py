import os

def modificar_archivos_gjc(directorio, carga, multiplicidad):
    """
    Modifica los archivos .gjc en el directorio especificado para cambiar la carga y multiplicidad.

    Args:
        directorio (str): Ruta al directorio donde se encuentran los archivos .gjc.
        carga (str): Valor de la carga a ser modificado en los archivos.
        multiplicidad (str): Valor de la multiplicidad a ser modificado en los archivos.
    """
    # Obtener la lista de archivos .gjc en el directorio especificado
    archivos_gjc = [f for f in os.listdir(directorio) if f.endswith('.gjc')]

    # Recorrer cada archivo .gjc
    for archivo in archivos_gjc:
        # Ruta completa del archivo
        ruta_archivo = os.path.join(directorio, archivo)
        
        # Abrir el archivo en modo lectura
        with open(ruta_archivo, 'r') as archivo_lectura:
            # Leer las líneas del archivo
            lineas = archivo_lectura.readlines()

        # Verificar si el archivo fue procesado
        if any(linea.startswith("%nprocshared") for linea in lineas):
            # Modificar la línea que contiene la carga y multiplicidad
            lineas[6] = f"   {carga}   {multiplicidad}\n"

        # Abrir el archivo en modo escritura
        with open(ruta_archivo, 'w') as archivo_escritura:
            # Escribir las líneas modificadas en el archivo
            archivo_escritura.writelines(lineas)

    print("Proceso completado.")

def main():
    # Pedir los valores de carga y multiplicidad al usuario
    directorio = 'C:\\Linux'
    carga = input('Ingrese el valor de carga: ')
    multiplicidad = input('Ingrese el valor de multiplicidad: ')
    
    # Llamar a la función para modificar los archivos
    modificar_archivos_gjc(directorio, carga, multiplicidad)

# Estructura principal para ejecutar el script
if __name__ == "__main__":
    main()