import glob
import os
import shutil

# Acá podes cambiar tu usuario <------
user = "bcelada.iquir"

# ----- PEDIR DATOS -------

# Tipo de cola
ETH = int(input("Que tipo de colas? 1 para HI, 9 para Lo:  "))
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


# ------ CREAR CARPETAS SEPARADAS Y MOVER ARCHIVOS CORRESPONDIENTES ----------

# Ruta de la carpeta donde se encuentran los archivos (carpeta base)
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
        os.makedirs(nueva_carpeta, exist_ok=True)
        
        # Agregar la carpeta al diccionario
        carpetas_creadas[primeros_tres] = nueva_carpeta
    
    # Mover el archivo a la carpeta correspondiente
    shutil.move(os.path.join(carpeta_base, archivo), carpetas_creadas[primeros_tres])


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

# Obtener la lista de carpetas en el directorio principal
carpetas = glob.glob(os.path.join(carpeta_base, "*"))

# Iterar sobre cada carpeta
for carpeta in carpetas:
        # Obtener el nombre de la carpeta actual
        nombre_carpeta = os.path.basename(carpeta)

        # Cambiar al directorio de la carpeta actual
        os.chdir(carpeta)

        # Crear las variables name y job_name automaticamente
        name = f"{nombre_carpeta}_c"
        job_name = nombre_carpeta  

        # Crear los .sh correspondientes de cada conformero
        a = glob.glob("*.gjc")

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


# ------ EXTRAER ARCHIVOS Y ELIMINAR CARPETAS VACÍAS ----------

# Cambiar el directorio actual fuera de cualquier subcarpeta, por seguridad
os.chdir(carpeta_base)

for carpeta in carpetas:
    # Obtener todos los archivos en la carpeta actual
    archivos = os.listdir(carpeta)
    
    # Mover cada archivo a la carpeta base
    for archivo in archivos:
        shutil.move(os.path.join(carpeta, archivo), carpeta_base)
    
    # Intentar eliminar la carpeta si está vacía
    try:
        os.rmdir(carpeta)
    except OSError as e:
        print(f"No se pudo eliminar la carpeta {carpeta}: {e}")
        
# Aviso de finalización del script
print("Finalizado.")
