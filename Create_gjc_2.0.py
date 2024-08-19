import glob
import os
import logging
import Spa_Gaus_CreateSH_Slack

# Configuración de logging
logging.basicConfig(filename="registros/script.log", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

logging.info("\n\n-------NEW CREADOR SINGLE POINT-------\n")

carpeta_base = "C:\\Linux"
os.chdir(carpeta_base)

# Script para extraer la última estructura optimizada de archivos .log y crear archivos .gjc
# (útil para hacer un single point con solvente luego de una optb)

for file in glob.glob("*.log"):
    try:
        log_filename = os.path.splitext(file)[0]
        new_filename = f"{log_filename}.gjc"
        
        with open(file, "r") as old_file, open(new_filename, "w") as new_file:
            lines = old_file.readlines()
            start, end = 0, 0
            
            # Encuentra la última aparición de "Standard orientation:"
            for i, line in enumerate(lines):
                if "Standard orientation:" in line:
                    start = i
            
            # Encuentra el final de la estructura optimizada
            for i in range(start + 5, len(lines)):
                if "---" in lines[i]:
                    end = i
                    break
            
            # Escribe la estructura optimizada en el nuevo archivo
            for line in lines[start + 5:end]:
                words = line.split()
                if len(words) >= 6:
                    new_file.write(f"{words[1]} {words[3]} {words[4]} {words[5]}\n")
        
        logging.info(f"Estructura optimizada copiada a {new_filename}")

    except Exception as e:
            logging.error(f"Error leyendo archivo .log {file}: {e}")

# Solicitar entradas del usuario
nprocshared = input("nprocshared= ")
mem = input("mem= ")
command_line = input("command line= ")
charge = 0
multip = 1

print("¿Es la carga = 0 y multiplicidad = 1?")
opcion = input("n/no para cambiar, cualquier tecla para continuar. ")

if opcion.lower() in ["n", "no"]:
    charge = input("Carga = ")
    multip = input("Multiplicidad = ")

# Añadir encabezados a los archivos .gjc
for file in glob.glob("*.gjc"):
    try:
        with open(file, "r+") as gjc_file:
            content = gjc_file.read()
            gjc_file.seek(0, 0)
            header = (
                f"%nprocshared={nprocshared}\n"
                f"%mem={mem}\n"
                f"{command_line}\n\n"
                "comment\n\n"
                f"{charge} {multip}\n"
            )
            gjc_file.write(header + content + "\n" + "\n" + "\n")

        logging.info(f"Archivo creado {file}")

    except Exception as e:
            logging.error(f"Error creando archivo .gjc {file}: {e}")

create_sh = input("Crear archivos .sh? (y/n): ").strip().lower()

if create_sh == "y":
    Spa_Gaus_CreateSH_Slack.main()

print("\nScript finalizado\n")
