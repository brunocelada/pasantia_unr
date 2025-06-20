import os
import pandas as pd
from openpyxl import load_workbook
import xlwings as xw
import time

def obtener_valor_desde_excel(ruta_archivo, celda):
    try:
        app = xw.App(visible=False)
        book = app.books.open(ruta_archivo)
        sheet = book.sheets[0]
        book.app.calculate()
        time.sleep(0.1)
        valor = sheet[celda].value
        book.close()
        app.quit()
        return valor
    except Exception as e:
        print(f"Error al leer {ruta_archivo}: {e}")
        return None

def main():
    # 1. Ruta con los archivos .xlsx
    ruta_origen = input("📂 Ingrese la ruta de la carpeta con los archivos .xlsx: ").strip()

    # 2. Celda a copiar
    celda_a_copiar = input("🔍 Ingrese la celda a copiar (ej: B2): ").strip().upper()

    # 3. Ruta de salida para el Excel final
    ruta_salida = input("💾 Ingrese la ruta donde desea guardar el archivo resumen: ").strip()
    nombre_salida = input("📝 Nombre del archivo de salida (ej: resumen.xlsx): ").strip()

    # Crear diccionario
    resumen = {}

    for archivo in os.listdir(ruta_origen):
        if archivo.endswith(".xlsx") and not archivo.startswith("~$"):
            ruta_completa = os.path.join(ruta_origen, archivo)
            valor = obtener_valor_desde_excel(ruta_completa, celda_a_copiar)
            print(f"📄 Procesando {archivo}... Valor en {celda_a_copiar}: {valor}")
            resumen[archivo] = valor

    # Convertir a DataFrame
    df = pd.DataFrame(list(resumen.items()), columns=["Archivo", f"Valor en {celda_a_copiar}"])

    # Guardar en Excel
    ruta_final = os.path.join(ruta_salida, nombre_salida)
    df.to_excel(ruta_final, index=False)

    print(f"\n✅ Archivo guardado exitosamente en:\n{ruta_final}")

if __name__ == "__main__":
    main()
