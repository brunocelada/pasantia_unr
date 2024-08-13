import subprocess
import sys

# INSTALAR LIBRERIA OPENPYXL PARA CREAR ARCHIVOS EXCEL
# Verificar e instalar openpyxl si no está instalado
try:
    import openpyxl
    print("openpyxl ya está instalado.")
except ImportError:
    print("openpyxl no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl


import glob
import os
from openpyxl import Workbook

# CREAR EXCEL
# Obtener la lista de carpetas en el directorio principal
carpetas = glob.glob("C:\\linux\\*")

# Iterar sobre cada carpeta
for carpeta in carpetas:
    # Obtener el nombre de la carpeta actual
    nombre_carpeta = os.path.basename(carpeta)

    # Cambiar al directorio de la carpeta actual
    os.chdir(carpeta)

    # Listar los archivos .out en la carpeta actual
    listing = glob.glob("*.out")

    # Renombrar cada archivo .out a .log
    for file in listing:
        os.rename(file, (file.rsplit(".", 1)[0]) + ".log")

    # Crear un nuevo libro de Excel
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = nombre_carpeta

    # Escribir el encabezado en la hoja de cálculo
    sheet.append(["Compound_Name", "SCF", "ZPE", "Enthalpie", "Gibbs"])


    # Procesar cada archivo .log
    for file in glob.glob("*.log"):
        with open(file, "r") as old_file:
            rline = old_file.readlines()

            scf = 0
            gibbs = 0

            # Buscar "SCF Done:" y "Sum of electronic and thermal Free Energies="
            for i in range(len(rline)):
                if "SCF Done:" in rline[i]:
                    scf = i
            for j in range(len(rline)):
                if "Sum of electronic and thermal Free Energies=" in rline[j]:
                    gibbs = j

            # Extraer los valores necesarios y escribirlos en la hoja de cálculo
            if gibbs == 0:
                sheet.append([(file.rsplit(".", 1)[0]), rline[scf].split()[4], "-", "-", "-"])
            else:
                wzpe = rline[gibbs - 3].split()
                wentalpie = rline[gibbs - 1].split()
                wgibbs = rline[gibbs].split()
                sheet.append([(file.rsplit(".", 1)[0]), rline[scf].split()[4], wzpe[6], wentalpie[6], wgibbs[7]])

    # Ajustar el ancho de las columnas
    for col in sheet.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        adjusted_width = max_length + 2
        sheet.column_dimensions[column].width = adjusted_width

    # Guardar el archivo Excel
    workbook.save(f"{nombre_carpeta}.xlsx")

print("Terminó")