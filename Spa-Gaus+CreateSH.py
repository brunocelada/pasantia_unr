import glob
import os
import shutil
import sys

# Acá podes cambiar tu usuario <------
user = "bcelada.iquir"

# ----------------------------------------------------------------------------------------------------
# <<<------- SPARTAN TO GAUSSIAN ------->>>

# Ruta base donde se encuentran los archivos .gjc
carpeta_base = "C:\\Linux"

# Obtener la lista de archivos .gjc en la carpeta
files = [f for f in os.listdir(carpeta_base) if f.endswith(".gjc")]

# Marcador para indicar si se necesitan solicitar las variables
necesita_procesamiento = False

# Recorre cada archivo .gjc para verificar si alguno necesita ser procesado
for file in files:
    file_path = os.path.join(carpeta_base, file)
    try:
        # Abrir el archivo en modo lectura
        with open(file_path, "r") as f:
            lines = f.readlines()
            
            # Verificar si la línea "%nprocshared" ya está presente
            if not any(line.startswith("%nprocshared") for line in lines):
                necesita_procesamiento = True
                break  # No es necesario seguir verificando otros archivos
                    # Salta a sección Create SH - CAPITAN

    except Exception as e:
        print(f"Error al verificar el archivo {file}: {e}")

# Si algún archivo necesita procesamiento, solicitar las variables y procesar los archivos
if necesita_procesamiento:
    print("\n Algunos archivos necesitan procesamiento.\n")
    nproc = input("nprocshared= ")
    mem = input("mem= ")
    command_line = input("command line= ")

    # Recorre cada archivo .gjc para procesarlo
    for file in files:
        file_path = os.path.join(carpeta_base, file)
        try:
            # Abrir el archivo en modo lectura
            with open(file_path, "r") as f:
                lines = f.readlines()
                
                # Verificar si la línea "%nprocshared" ya está presente
                if any(line.startswith("%nprocshared") for line in lines):
                    print(f"El archivo {file} ya ha sido procesado. Saltando...")
                    continue  # Saltar a la siguiente iteración si ya está modificado
                
                # Si el archivo tiene al menos dos líneas, reemplaza las dos primeras
                if len(lines) >= 2:
                    lines[0] = f"%nprocshared={nproc}\n"
                    lines[1] = f"%mem={mem}\n"
                    lines.insert(2, f"{command_line}\n")
                    lines.insert(3, "\n")
                    lines.insert(4, "comment\n")
                    lines.insert(5, "\n")
                
                # Encontrar y eliminar "ENDCART" y todo lo que sigue
                for i, line in enumerate(lines):
                    if "ENDCART" in line:
                        lines = lines[:i]  # Elimina desde "ENDCART" hacia abajo
                        lines.extend(["\n"] * 3)  # Agregar tres líneas en blanco
                        break
                
            # Reabrir el archivo en modo escritura para guardar los cambios
            with open(file_path, "w") as f:
                f.writelines(lines)
            
        except Exception as e:
            print(f"Error al procesar el archivo {file}: {e}")

    print("\nModificación de archivos .gjc completada. \n")

# Si todos los archivos ya están procesados
else:
    print("\nTodos los archivos ya han sido procesados. No es necesario modificar nada. \n")


# --------------------------------------------------------------------------------------------------
# <<<------- CREATE SH - CAPITAN ------->>>

# Verificar si se desea crear los archivos .sh
print ("Desea crear los archivos .sh para cada .gjc? ")

opcion = input("n/no para cancelar, cualquier tecla para continuar. ")

if (opcion.lower() in ["n", "no"]):
    print("\nNo se crearán los archivos .sh. Se cerrará el programa.\n")
    sys.exit()  # Finaliza el script 


# ----- PEDIR DATOS -------

# Tipo de cola
ETH = int(input("\nQue tipo de colas? 1 para HI, 9 para Lo:  "))
while ETH not in [1, 9]:
        print("Tipeaste mal")
        ETH = int(input("Que tipo de colas? 1 para HI, 9 para Lo: "))

part = "eth_hi" if ETH == 1 else "eth_low"

# Numero de procesadores
nprocshared = input("Cuantos procesadores queres?: ")

# Tiempo de trabajo
jobtime = int(input("Que tiempo queres? 1 para 12 h, 5 para 24 h, 9 para 28 h: "))
while jobtime not in [1, 5, 9]:
        print("Tipeaste mal")
        jobtime = int(input("Que tiempo queres?: 1 para 12 h, 5 para 24 h, 9 para 28 h: "))

tiempo = {1: "12", 5: "24", 9: "48"}[jobtime]        

# Nombre de carpeta del cluster
folder = input("Folder: ")
fold = f"g09 /home/{user}/{folder}/" if folder else f"g09 /home/{user}/"

# Cantidad de inputs por sh
delta = int(input("Cuantos inputs por sh: "))

# ------ INGRESAR EN CADA CARPETA EL CODIGO PARA CREAR SH - Capitan_2.0 ----------

# TEMPLATE
sh_template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --partition={part}
#SBATCH --ntasks={nprocshared}
#SBATCH --time={tiempo}:00:00
#SBATCH --output={job_name}_%j.log

# ------- Defining root directory for gaussian

g09root=/opt/ohpc/pub/apps/software/Gaussian/09/AMD64.SSE4a-enabled
mkdir /local/$USER
GAUSS_SCRDIR=/local/$USER
export g09root GAUSS_SCRDIR
. $g09root/g09/bsd/g09.profile

# -------- SECTION print some infos to stdout ---------------------------------
echo " "
echo "START_TIME           = `date +'%y-%m-%d %H:%M:%S %s'`"
START_TIME=`date +%s`
echo "HOSTNAME             = $HOSTNAME"
echo "JOB_NAME             = $JOB_NAME"
echo "JOB_ID               = $JOB_ID"
echo "SGE_O_WORKDIR        = $SGE_O_WORKDIR"
echo "NSLOTS               = $NSLOTS"
echo " "

# -------- SECTION executing program ---------------------------------

echo " "
echo "Running:"
echo " "

{commands}

# -------- SECTION final cleanup and timing statistics ------------------------

echo "END_TIME (success)   = `date +'%y-%m-%d %H:%M:%S %s'`"
END_TIME=`date +%s`
echo "RUN_TIME (hours)     = `echo \"$START_TIME $END_TIME\" | awk '{{printf(\"%.4f\",($2-$1)/60.0/60.0)}}'`"

exit 0
"""

os.chdir(carpeta_base)

a = glob.glob("*.gjc")
       
for file in a:
        # Crear las variables name y job_name automaticamente
        name = f"{file}_c"
        job_name = file

        # Crear los .sh correspondientes de cada conformero
        counter = len(a)
        nsh = (counter // delta)

        for i in range(nsh):
                commands = "\n".join(fold + a[i*delta+j] for j in range(delta))

                with open(f"{name}{i+1}.sh", "w") as f:
                        f.write(sh_template.format(job_name=job_name, part=part, nprocshared=nprocshared, tiempo=tiempo, commands=commands))

        resto = counter - delta * nsh   
        if resto:
                commands = "\n".join(fold + a[nsh*delta+j] for j in range(resto))
                with open(f"{name}{nsh+1}.sh", "w") as f:
                        f.write(sh_template.format(job_name=job_name, part=part, nprocshared=nprocshared, tiempo=tiempo, commands=commands))

        
# Aviso de finalización del script
print("Finalizado.")
