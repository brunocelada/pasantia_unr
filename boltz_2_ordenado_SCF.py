import os
import re
import math
from openpyxl import Workbook
import logging

# Configuración de logging
logging.basicConfig(filename="registros/script.log", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

# Definir constantes
K_B = 1.987e-3  # Constante de Boltzmann en kcal/(mol·K)
TEMPERATURE = 298.15  # Temperatura en Kelvin
SCF_REGEX = r'SCF Done:  E\((.*?)\) =\s*(-?\d+\.\d+)'  # Expresión regular para buscar las líneas relevantes
ENERGY_CONVERSION_FACTOR = 627.5095  # Factor de conversión de energía

def read_log_files(directory, sheettitle, filetitle, files_ignorados):
    energies = []
    filenames = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename in files_ignorados:
                continue
            if filename.endswith(".log"):
                file_path = os.path.join(root, filename)
                last_match = None
                with open(file_path, 'r') as file:
                    for line in file:
                        match = re.search(SCF_REGEX, line)
                        if match:
                            last_match = match
                if last_match:
                    functional = last_match.group(1)
                    energy = float(last_match.group(2)) * ENERGY_CONVERSION_FACTOR
                    energies.append(energy)
                    filenames.append(filename)
        if energies and filenames:
            process_and_write_xlsx(root, filenames, energies, sheettitle, filetitle)
            energies.clear()
            filenames.clear()

def calculate_boltzmann_weights(energies):
    min_energy = min(energies)
    relative_energies = [energy - min_energy for energy in energies]
    boltzmann_factors = [math.exp(-energy / (K_B * TEMPERATURE)) for energy in relative_energies]
    partition_function = sum(boltzmann_factors)
    probabilities = [factor / partition_function * 100 for factor in boltzmann_factors]  # Convertir a porcentaje
    return relative_energies, probabilities

def write_xlsx(filename, sheettitle, data):
    wb = Workbook()
    ws = wb.active
    ws.title = sheettitle
    
    # Escribir los encabezados
    ws.append(["Filename", "Energy (kcal/mol)", "Relative Energy (kcal/mol)", "Boltzmann (%)"])
    
    # Escribir los datos
    for row in data:
        ws.append(row)
    
    # Guardar el archivo
    wb.save(filename)
    logging.info(f"Archivo Excel para Boltzmann creado: {filename}")

def process_and_write_xlsx(directory, filenames, energies, sheettitle, filetitle):
    relative_energies, probabilities = calculate_boltzmann_weights(energies)
    
    # Ordenar los datos por energía
    combined_data = sorted(zip(filenames, energies, relative_energies, probabilities), key=lambda x: x[1])
    
    output_file = os.path.join(directory, filetitle)
    write_xlsx(output_file, sheettitle, combined_data)

def main():
    # Registrar un nuevo lanzamiento del script "boltz_2_ordenado_SCF.py"
    logging.info("\n\n----NEW BOLTZ_2_ORDENADO_SCF----\n")
    
    directory = r'C:\Linux'
    sheettitle = "Boltzmann Data"
    filetitle = "Boltzmann.xlsx"

    # Carpetas que no se procesarán
    patrones_ignorados = ["Frecuencias Negativas", "Fre", "Relanzar", "Rel", "Termino Mal", "Ter"]

    read_log_files(directory, sheettitle=sheettitle, filetitle=filetitle, files_ignorados=[])

    print("\nBoltzmann Finalizado\n")

if __name__ == "__main__":
    main()
