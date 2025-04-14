#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os

# Set directory to "C:\Linux" and change working directory
directory = "C:\\Linux"
os.chdir(directory)

ETH = input("Que tipo de colas? 1 para HI, 3 para epyc, 5 para IB100,6 para gpua10, 7 para colas cortas, 8 para organica, 9 para Lo:  ")
while int(ETH) not in [1, 3, 5, 6, 7, 8, 9]:
    print("Tipeaste mal")
    ETH = input("Que tipo de colas? 1 para HI, 3 para epyc, 5 para IB100,6 para gpua10 7 para colas cortas, 8 para organica, 9 para Lo:  ")

if int(ETH) == 1:
    part = "eth_hi"
elif int(ETH) == 3:
    part = "eth_epyc"
elif int(ETH) == 5:
    part = "ib100"
elif int(ETH) == 6:
    part = "gpua10"
elif int(ETH) == 7:
    part = "matcond,colisiones,colisionesNuevo,ferro,fiquin,organica"
elif int(ETH) == 8:
    part = "organica"
elif int(ETH) == 9:
    part = "eth_low"

nprocshared = input("Cuantos procesadores quieres?: ")

jobtime = input("Que tiempo quieres?: 1 para 12 h, 5 para 24 h, 9 para 28 h: ")
while int(jobtime) not in [1, 5, 9]:
    print("Tipeaste mal")
    jobtime = input("Que tiempo quieres?: 1 para 12 h, 5 para 24 h, 9 para 28 h: ")

if int(jobtime) == 1:
    tiempo = "12:00:00"
elif int(jobtime) == 5:
    tiempo = "24:00:00"
elif int(jobtime) == 9:
    tiempo = "28:00:00"

folder = input("Folder: ")
name = input("Prefijo ")

if folder == "":
    fold = "g16 /home/icortes.iquir/"
else:
    fold = "g16 /home/icortes.iquir/" + folder + "/"

script = '''\
# ------- Defining root directory for gaussian

CPU=$(lscpu | grep "Model name")

MODELO="Intel(R)"

if [[ $CPU = *$MODELO* ]]
then
    echo "micro Intel"
    g16root=/opt/ohpc/pub/apps/software/Gaussian/16/AVX-enabled
else
    echo "micro AMD"
    g16root=/opt/ohpc/pub/apps/software/Gaussian/16/SSE4-enabled
fi

# Comprobando si el directorio ya existe o GAUSS_SCRDIR no está vacío
if [ ! -d "/local/$USER" ] && [ -z "$GAUSS_SCRDIR" ]; then
    mkdir /local/$USER
fi

GAUSS_SCRDIR=/local/$USER/$SLURM_JOB_ID
export g16root GAUSS_SCRDIR
. $g16root/g16/bsd/g16.profile

# chequear si existe el directorio
if [[ ! -d "$GAUSS_SCRDIR" ]]; then
        mkdir -p "$GAUSS_SCRDIR"
fi
'''

script_time = '''\
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
'''

# Search for .gjc files in C:\Linux
a = glob.glob("*.gjc")
counter = len(a)
delta = int(input("Cuantos inputs por sh: "))
nsh = int(counter / delta)

for i in range(nsh):
    f = open(name + str(i + 1) + ".sh", "w")
    f.write(
        "#!/bin/bash\n#SBATCH --job-name=G16job\n#SBATCH --nodes=1\n#SBATCH --partition=" + part + "\n#SBATCH --ntasks=" + nprocshared + "\n#SBATCH --time=" + tiempo + "\n#SBATCH --output=G16job_%j.log\n\n\n")
    f.write(script)
    f.write(script_time)
    f.write("#-------- SECTION executing program ---------------------------------\n\necho \" \"\necho \"Running:\"\necho \" \"\n\n")

    for j in range(delta):
        f.write(fold + a[i * delta + j] + "\n")

    f.write(
        "\n\n# -------- SECTION final cleanup and timing statistics ------------------------\n\necho \"END_TIME (success)  = `date +\'%y-%m-%d %H:%M:%S %s\'`\"\nEND_TIME=`date +%s`\necho \"RUN_TIME (hours)   = \"`echo \"$START_TIME $END_TIME\" | awk \'{printf(\"%.4f\",($2-$1)/60.0/60.0)}\'`\n")
    
    f.write("\nrm -rf $GAUSS_SCRDIR\n")
    f.write("\nexit 0")
    f.close()

resto = int(counter - delta * nsh)

if resto != 0:
    f = open(name + str(nsh + 1) + ".sh", "w")
    f.write(
        "#!/bin/bash\n#SBATCH --job-name=G16job\n#SBATCH --nodes=1\n#SBATCH --partition=" + part + "\n#SBATCH --ntasks=" + nprocshared + "\n#SBATCH --time=" + tiempo + "\n#SBATCH --output=G16job_%j.log\n\n\n")
    f.write(script)
    f.write(script_time)
    f.write("#-------- SECTION executing program ---------------------------------\n\necho \" \"\necho \"Running:\"\necho \" \"\n\n")

    for j in range(resto):
        f.write(fold + a[nsh * delta + j] + "\n")

    f.write(
        "\n\n# -------- SECTION final cleanup and timing statistics ------------------------\n\necho \"END_TIME (success)   = `date +\'%y-%m-%d %H:%M:%S %s\'`\"\nEND_TIME=`date +%s`\necho \"RUN_TIME (hours)     = \"`echo \"$START_TIME $END_TIME\" | awk \'{printf(\"%.4f\",($2-$1)/60.0/60.0)}\'`\n\n\nexit 0")
    f.close()




import os

# Ruta de la carpeta "C:\Linux"
carpeta_linux = r'C:\Linux'

# Preguntar al usuario por el valor de reemplazo
nuevo_valor = input("job-name?")

# Recorrer todos los archivos en la carpeta
for archivo in os.listdir(carpeta_linux):
    if archivo.endswith(".sh"):
        ruta_archivo = os.path.join(carpeta_linux, archivo)
        # Leer el contenido del archivo
        with open(ruta_archivo, 'r') as f:
            contenido = f.read()
        
        # Reemplazar "G16job" por el nuevo valor
        nuevo_contenido = contenido.replace("G16job", nuevo_valor)
        
        # Escribir el nuevo contenido de vuelta al archivo
        with open(ruta_archivo, 'w') as f:
            f.write(nuevo_contenido)

print("Reemplazo completado.")
