import glob
import os

os.chdir ("C:\\linux")

ETH=input("Que tipo de colas? 1 para HI, 3 para epyc, 5 para IB100,6 para gpua10, 7 para colas cortas, 8 para organica, 9 para Lo:  ")
while (int(ETH)!=1) and (int(ETH)!=9) and (int(ETH)!=3) and (int(ETH)!=5) and (int(ETH)!=6) and (int(ETH)!=7)and (int(ETH)!=8) :  
        print("Tipeaste mal")
        ETH=input("Que tipo de colas? 1 para HI, 3 para epyc, 5 para IB100,6 para gpua10 7 para colas cortas, 8 para organica, 9 para Lo:  ")

if int(ETH)==1:
        part="eth_hi"
if int(ETH)==3:
        part="eth_epyc"
if int(ETH)==5:
        part="ib100"
if int(ETH)==6:
        part="gpua10"
if int(ETH)==7:
        part="matcond,colisiones,colisionesNuevo,ferro,fiquin,organica"
if int(ETH)==8:
        part="organica"
if int(ETH)==9:
        part="eth_low"

nprocshared=input("Cuantos procesadores queres?: ")

jobtime=input("Que tiempo queres?: 1 para 12 h, 5 para 24 h, 9 para 48 h: ")
while (int(jobtime)!=1) and (int(jobtime)!=5) and (int(jobtime)!=9):
        print("Tipeaste mal")
        jobtime=input("Que tiempo queres?: 1 para 12 h, 5 para 24 h, 9 para 48 h: ")
        
if int(jobtime)==1:
        tiempo="12"
if int(jobtime)==5:
        tiempo="24"
if int(jobtime)==9:
        tiempo="48"
        


folder = input("Folder: ")

name = input("Sufijo ")



if folder=="":
        fold="g09 /home/bcelada.iquir/"
else:
        fold="g09 /home/bcelada.iquir/"+folder+"/"



script = '''\
# ------- Defining root directory for gaussian

CPU=$(head /proc/cpuinfo | grep "model name")

MODELO="Intel(R)"

if [[ $CPU = *$MODELO* ]]
then
    echo "micro Intel"
    g09root=/opt/ohpc/pub/apps/software/Gaussian/09/EM64T.SSE4.2-enabled
else
    echo "micro AMD"
    g09root=/opt/ohpc/pub/apps/software/Gaussian/09/AMD64.SSE4a-enabled
fi

mkdir /local/$USER
GAUSS_SCRDIR=/local/$USER
export g09root GAUSS_SCRDIR
. $g09root/g09/bsd/g09.profile


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



a = glob.glob("*.gjc")

counter = len(glob.glob("*.gjc"))

delta = int(input("Cuantos inputs por sh: "))

nsh = int(counter/delta)

for i in range(0,nsh):
        f = open(name+str(i+1)+".sh","w")
        f.write("#!/bin/bash\n#SBATCH --job-name=G09job\n#SBATCH --nodes=1\n#SBATCH --partition="+part+"\n#SBATCH --ntasks="+nprocshared+"\n#SBATCH --time="+tiempo+":00:00\n#SBATCH --output=G09job_%j.log\n\n\n")
        f.write(script)
        f.write(script_time)
        f.write("-------- SECTION executing program ---------------------------------\n\necho \" \"\necho \"Running:\"\necho \" \"\n\n")

        for j in range(0,int(delta)):
                f.write(fold+a[i*delta+j]+"\n")
        
        f.write("\n\n# -------- SECTION final cleanup and timing statistics ------------------------\n\necho \"END_TIME (success)   = `date +\'%y-%m-%d %H:%M:%S %s\'`\"\nEND_TIME=`date +%s`\necho \"RUN_TIME (hours)     = \"`echo \"$START_TIME $END_TIME\" | awk \'{printf(\"%.4f\",($2-$1)/60.0/60.0)}\'`\n\n\nexit 0")
        f.close()

resto = int(counter-delta*nsh)

if resto!=0:
        f = open(name+str(nsh+1)+".sh","w")
        f.write("#!/bin/bash\n#SBATCH --job-name=G09job\n#SBATCH --nodes=1\n#SBATCH --partition="+part+"\n#SBATCH --ntasks="+nprocshared+"\n#SBATCH --time="+tiempo+":00:00\n#SBATCH --output=G09job_%j.log\n\n\n")
        f.write(script)
        f.write(script_time)
        f.write("-------- SECTION executing program ---------------------------------\n\necho \" \"\necho \"Running:\"\necho \" \"\n\n")       
        
        for j in range(0,resto):
                f.write(fold+a[nsh*delta+j]+"\n")

        f.write("\n\n# -------- SECTION final cleanup and timing statistics ------------------------\n\necho \"END_TIME (success)   = `date +\'%y-%m-%d %H:%M:%S %s\'`\"\nEND_TIME=`date +%s`\necho \"RUN_TIME (hours)     = \"`echo \"$START_TIME $END_TIME\" | awk \'{printf(\"%.4f\",($2-$1)/60.0/60.0)}\'`\n\n\nexit 0")
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
        
        # Reemplazar "G09job" por el nuevo valor
        nuevo_contenido = contenido.replace("G09job", nuevo_valor)
        
        # Escribir el nuevo contenido de vuelta al archivo
        with open(ruta_archivo, 'w') as f:
            f.write(nuevo_contenido)

print("Reemplazo completado.")

           
           



