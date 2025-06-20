import os

def pedir_lineas_criterio():
    while True:
        try:
            n = int(input("¿Cuántas líneas quiere especificar como criterio final? "))
            if n <= 0:
                raise ValueError
            break
        except ValueError:
            print("Por favor, ingrese un número entero positivo.")
    
    lineas_criterio = []
    for i in range(n):
        linea = input(f"Ingrese la línea #{i + 1} empezando desde la ANTEpenúltima hasta la última (orden cronológico): ")
        lineas_criterio.append(linea.strip())
    return lineas_criterio

def verificar_lineas_finales(ruta_archivo, lineas_objetivo):
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            contenido = f.readlines()
            contenido = [linea.strip() for linea in contenido if linea.strip()]  # eliminamos líneas vacías
            if len(contenido) < len(lineas_objetivo):
                return False
            return contenido[-len(lineas_objetivo):] == lineas_objetivo
    except Exception as e:
        print(f"Error leyendo {ruta_archivo}: {e}")
        return False

def analizar_archivos_en_carpeta(ruta_base, lineas_criterio):
    cumplen = []
    no_cumplen = []

    for archivo in os.listdir(ruta_base):
        if archivo.endswith(".txt") and archivo != "log_analysis.txt":
            ruta_completa = os.path.join(ruta_base, archivo)
            if verificar_lineas_finales(ruta_completa, lineas_criterio):
                cumplen.append(archivo)
            else:
                no_cumplen.append(archivo)

    return cumplen, no_cumplen

def guardar_resultados(ruta_salida, cumplen, no_cumplen):
    os.makedirs(ruta_salida, exist_ok=True)  # crea la carpeta si no existe
    resultado_path = os.path.join(ruta_salida, "log_analysis.txt")
    with open(resultado_path, 'w', encoding='utf-8') as f:
        f.write("Archivos que cumplen\n")
        for archivo in cumplen:
            f.write(f"{archivo}\n")
        f.write("\nArchivos que NO cumplen\n")
        for archivo in no_cumplen:
            f.write(f"{archivo}\n")
    print(f"\nResultados guardados en: {resultado_path}")

def obtener_listas_de_modelos(ruta_base, lineas_criterio):
    cumplen, no_cumplen = analizar_archivos_en_carpeta(ruta_base, lineas_criterio)
    modelos_ok = [archivo.replace(".txt", "") for archivo in cumplen]
    modelos_error = [archivo.replace(".txt", "") for archivo in no_cumplen]
    return modelos_ok, modelos_error

def main():
    ruta = input("Ingrese la ruta de la carpeta base con los archivos .txt: ").strip()
    if not os.path.isdir(ruta):
        print("La ruta ingresada no es válida.")
        return

    ruta_salida = input("Ingrese la carpeta donde se guardará el archivo log_analysis.txt: ").strip()
    if not os.path.isdir(ruta_salida):
        crear = input("La carpeta no existe. ¿Desea crearla? (s/n): ").lower()
        if crear == 's':
            os.makedirs(ruta_salida)
        else:
            print("No se creó la carpeta de salida. Cancelando...")
            return

    lineas_criterio = pedir_lineas_criterio()
    cumplen, no_cumplen = analizar_archivos_en_carpeta(ruta, lineas_criterio)

    print("\nArchivos que cumplen:")
    for c in cumplen:
        print(c)

    print("\nArchivos que NO cumplen:")
    for nc in no_cumplen:
        print(nc)

    guardar_resultados(ruta_salida, cumplen, no_cumplen)

if __name__ == "__main__":
    main()
