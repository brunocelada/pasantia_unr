import os
import glob
import tkinter as tk
from tkinter import messagebox
import sys
import logging  

# Configuración de logging
logging.basicConfig(filename="registros/script.log", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

# Acá podes cambiar tu usuario <------
user = "bcelada.iquir"

# Ruta base donde se encuentran los archivos .gjc <------
carpeta_base = "C:\\Linux"

# Plantilla de archivos .sh
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

{calc_notif}

{commands}


# -------- SECTION final cleanup and timing statistics ------------------------

status=$?

echo "END_TIME (success)   = `date +'%y-%m-%d %H:%M:%S %s'`"
END_TIME=`date +%s`
echo "RUN_TIME (hours)     = `echo \"$START_TIME $END_TIME\" | awk '{{printf(\"%.4f\",($2-$1)/60.0/60.0)}}'`"

{final_notif}

exit $status
"""

def obtener_archivos_gjc(carpeta_base):
    return [f for f in os.listdir(carpeta_base) if f.endswith(".gjc")]

def verificar_procesamiento(files, carpeta_base):
    for file in files:
        file_path = os.path.join(carpeta_base, file)
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                if not any(line.startswith("%nprocshared") for line in lines):
                    logging.info(f"No se necesita procesar ningún archivo")
                    return True
        except Exception as e:
            logging.info(f"Error al verificar el archivo {file}: {e}")
    return False

def procesar_archivos(files, carpeta_base, nproc, mem, command_line):
    for file in files:
        file_path = os.path.join(carpeta_base, file)
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                
                if any(line.startswith("%nprocshared") for line in lines):
                    logging.info(f"El archivo {file} ya ha sido procesado. Saltando...")
                    continue
                
                if len(lines) >= 2:
                    lines[0] = f"%nprocshared={nproc}\n"
                    lines[1] = f"%mem={mem}\n"
                    lines.insert(2, f"{command_line}\n")
                    lines.insert(3, "\n")
                    lines.insert(4, "comment\n")
                    lines.insert(5, "\n")
                
                for i, line in enumerate(lines):
                    if "ENDCART" in line:
                        lines = lines[:i]
                        lines.extend(["\n"] * 4)
                        break

            with open(file_path, "w") as f:
                logging.info(f"El archivo {file} se procesó correctamente.")
                f.writelines(lines)
            
        except Exception as e:
            logging.info(f"Error al procesar el archivo {file}: {e}")

def crear_archivo_sh(file, part, nprocshared, tiempo, fold, send_notif):
    os.chdir(carpeta_base)
    job_name = os.path.splitext(file)[0]
    commands = fold + file

    # Generar las secciones de Slack solo si se seleccionó enviar mensajes
    final_notif = ""
    calc_notif = ""

    if send_notif:
        calc_notif = f"""
# -------- Send Slack Join Calc Notification -----------------------------------------

curl -X POST -H 'Content-type: application/json; charset=utf-8' \
    --data '{{\"text\":\"El cálculo del archivo {job_name} ha ingresado al cluster.\"}}' \
    https://hooks.slack.com/services/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
"""
        final_notif = f"""
# -------- Send Slack Termination Notification -----------------------------------------

echo "END_TIME (success)   = `date +'%y-%m-%d %H:%M:%S %s'`"
END_TIME=`date +%s`
RUN_TIME_SECONDS=$(($END_TIME - $START_TIME))

# Calcular horas, minutos y segundos
if [ $RUN_TIME_SECONDS -ge 3600 ]; then
    RUN_TIME=$(awk "BEGIN {{printf \\"%.3f\\", $RUN_TIME_SECONDS/3600}}")
    TIME_UNIT="horas"
elif [ $RUN_TIME_SECONDS -ge 60 ]; then
    RUN_TIME=$(awk "BEGIN {{printf \\"%.3f\\", $RUN_TIME_SECONDS/60}}")
    TIME_UNIT="minutos"
else
    RUN_TIME=$RUN_TIME_SECONDS
    TIME_UNIT="segundos"
fi

if [ $status -eq 0 ]; then
    curl -X POST -H 'Content-type: application/json; charset=utf-8' \
    --data "{{\\"text\\":\\"El cálculo para el archivo {job_name} ha terminado con éxito. Tiempo total: $RUN_TIME $TIME_UNIT.\\"}}" \
    https://hooks.slack.com/services/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
else
    curl -X POST -H 'Content-type: application/json; charset=utf-8' \
    --data "{{\\"text\\":\\"¡Error en el cálculo para el archivo {job_name}! Tiempo total: $RUN_TIME $TIME_UNIT.\\"}}" \
    https://hooks.slack.com/services/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
fi
"""

    with open(f"{job_name}.sh", "w", encoding="utf-8") as f:
        f.write(sh_template.format(job_name=job_name, part=part, nprocshared=nprocshared, 
                                   tiempo=tiempo, commands=commands, calc_notif=calc_notif, final_notif=final_notif))

    logging.info(f"Archivo {job_name}.sh creado con éxito.")

# Función que se ejecuta cuando se hace clic en el botón de "Crear archivos .sh"
def crear_archivos():
    eth = entry_eth.get()
    jobtime = entry_jobtime.get()
    folder = entry_folder.get()
    
    log("Creando archivos .sh con ETH={}, jobtime={}, folder={}".format(eth, jobtime, folder))
    # Aquí va la lógica real para la creación de los archivos .sh, utilizando las variables eth, jobtime, folder
    
    if messagebox.askyesno("Notificaciones", "¿Deseas crear notificaciones por Slack?"):
        log("Notificaciones por Slack activadas.")
        # Código para enviar notificaciones por Slack
    else:
        log("Notificaciones por Slack desactivadas.")
    
def verificar_procesamiento_logic():
    if verificar_procesamiento():  # Si la función retorna True
        log("Procesamiento verificado. Solicitando parámetros adicionales.")
        mostrar_parametros(True)
    else:
        log("Procesamiento no verificado. Preguntando por la creación de archivos .sh.")
        mostrar_pregunta()

def mostrar_parametros(es_verificado):
    if es_verificado:
        label_procs.pack(pady=5)
        entry_procs.pack(pady=5)
        label_memoria.pack(pady=5)
        entry_memoria.pack(pady=5)
        label_cmd.pack(pady=5)
        entry_cmd.pack(pady=5)
        btn_continuar.pack(pady=10)
    else:
        label_eth.pack(pady=5)
        entry_eth.pack(pady=5)
        label_jobtime.pack(pady=5)
        entry_jobtime.pack(pady=5)
        label_folder.pack(pady=5)
        entry_folder.pack(pady=5)
        btn_crear_archivos.pack(pady=10)

def mostrar_pregunta():
    frame_pregunta.pack(pady=10)

def continuar():
    procs = entry_procs.get()
    memoria = entry_memoria.get()
    cmd = entry_cmd.get()
    
    log("Parámetros recibidos: procesadores={}, memoria={}, línea de comando={}".format(procs, memoria, cmd))
    logging.info("Algunos archivos necesitan procesamiento.")
    nproc = procs
    mem = memoria
    command_line = cmd

    files = obtener_archivos_gjc(carpeta_base)

    procesar_archivos(files, carpeta_base, nproc, mem, command_line)
    logging.info("Modificación de archivos .gjc completada. ")

def log(mensaje):
    text_log.insert(tk.END, mensaje + "\n")
    text_log.see(tk.END)
    logging.info(mensaje)

def finalizar_programa():
    root.quit()


# Crear la ventana principal
root = tk.Tk()
root.title("Programa de Creación de Archivos .sh")
root.geometry("400x600")
root.configure(bg="white")

# Título del programa
label_titulo = tk.Label(root, text="Creación de Archivos .sh", font=("Arial", 16), bg="white", fg="black")
label_titulo.pack(pady=20)

# Sección de loggings
text_log = tk.Text(root, height=10, bg="black", fg="white")
text_log.pack(pady=20)

# Entradas para cuando verificar_procesamiento es True
label_procs = tk.Label(root, text="Número de Procesadores:", bg="white", fg="black")
entry_procs = tk.Entry(root)

label_memoria = tk.Label(root, text="Memoria:", bg="white", fg="black")
entry_memoria = tk.Entry(root)

label_cmd = tk.Label(root, text="Línea de Comando:", bg="white", fg="black")
entry_cmd = tk.Entry(root)

btn_continuar = tk.Button(root, text="Continuar", command=continuar, bg="black", fg="white")

# Pregunta de "Sí" o "No" para crear archivos .sh
frame_pregunta = tk.Frame(root, bg="white")
label_pregunta = tk.Label(frame_pregunta, text="¿Deseas crear los archivos .sh?", bg="white", fg="black")
label_pregunta.pack(pady=5)
btn_si = tk.Button(frame_pregunta, text="Sí", command=lambda: mostrar_parametros(False), bg="black", fg="white")
btn_si.pack(side=tk.LEFT, padx=10)
btn_no = tk.Button(frame_pregunta, text="No", command=finalizar_programa, bg="black", fg="white")
btn_no.pack(side=tk.LEFT, padx=10)

# Entradas para cuando se elige "Sí"
label_eth = tk.Label(root, text="ETH:", bg="white", fg="black")
entry_eth = tk.Entry(root)

label_jobtime = tk.Label(root, text="Job Time:", bg="white", fg="black")
entry_jobtime = tk.Entry(root)

label_folder = tk.Label(root, text="Folder:", bg="white", fg="black")
entry_folder = tk.Entry(root)

btn_crear_archivos = tk.Button(root, text="Crear Archivos .sh", command=crear_archivos, bg="black", fg="white")

# Iniciar la lógica de verificación
verificar_procesamiento_logic()

root.mainloop()



logging.info("\n\n-------NEW SPA TO GAUSS + CREATE SH-------\n")


if verificar_procesamiento(files, carpeta_base):
    print("\nAlgunos archivos necesitan procesamiento.\n")
    nproc = input("nprocshared= ")
    mem = input("mem= ")
    command_line = input("command line= ")

    procesar_archivos(files, carpeta_base, nproc, mem, command_line)
    print("\nModificación de archivos .gjc completada. \n")
else:
    print("\nTodos los archivos ya han sido procesados. No es necesario modificar nada. \n")

if input("Desea crear los archivos .sh para cada .gjc? (n/no para cancelar): ").lower() in ["n", "no"]:
    print("\nNo se crearán los archivos .sh. Se cerrará el programa.\n")
    sys.exit()

ETH = int(input("\nQue tipo de colas? 1 para HI, 9 para Lo: "))
while ETH not in [1, 9]:
    print("Tipeaste mal")
    ETH = int(input("Que tipo de colas? 1 para HI, 9 para Lo: "))
part = "eth_hi" if ETH == 1 else "eth_low"

nprocshared = input("Cuantos procesadores queres?: ")

jobtime = int(input("Que tiempo queres? 1 para 12 h, 5 para 24 h, 9 para 48 h: "))
while jobtime not in [1, 5, 9]:
    print("Tipeaste mal")
    jobtime = int(input("Que tiempo queres?: 1 para 12 h, 5 para 24 h, 9 para 48 h: "))
tiempo = {1: "12", 5: "24", 9: "48"}[jobtime]        

folder = input("Folder: ")
fold = f"g09 /home/{user}/{folder}/" if folder else f"g09 /home/{user}/"

# Aclaración
print("\nRecordar que se trabaja con 1 input por sh.\n")

# Preguntar si se desea enviar notificaciones por Slack de los .sh finalizados
send_notif = input("¿Deseas enviar una notificación por Slack cuando el cálculo termine? (s/n): ").lower() in ["s", "si"]

if send_notif:       
    logging.info(f"Se enviarán notificaciones por Slack")
else:
    logging.info(f"No se enviarán notificaciones por slack.")

for file in files:
    crear_archivo_sh(file, part, nprocshared, tiempo, fold, send_notif)

