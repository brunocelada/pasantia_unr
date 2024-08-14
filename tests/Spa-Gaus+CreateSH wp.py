import os
import glob
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

{commands}

# -------- SECTION final cleanup and timing statistics ------------------------

if [ $status -eq 0 ]; then
    status_msg="Éxito"
else
    status_msg="Error"
fi

echo "END_TIME (success)   = `date +'%y-%m-%d %H:%M:%S %s'`"
END_TIME=`date +%s`
echo "RUN_TIME (hours)     = `echo \"$START_TIME $END_TIME\" | awk '{{printf(\"%.4f\",($2-$1)/60.0/60.0)}}'`"

{twilio_section}

exit 0
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

def crear_archivo_sh(file, part, nprocshared, tiempo, fold, send_whatsapp, account_sid="", auth_token="", 
                     from_number="", to_number=""):
    os.chdir(carpeta_base)
    job_name = os.path.splitext(file)[0]
    commands = fold + file

    # Generar la sección de Twilio solo si se seleccionó enviar mensajes
    twilio_section = ""
    if send_whatsapp:
        twilio_section = f"""
# -------- Send WhatsApp Notification -----------------------------------------

if [ $status -eq 0 ]; then
    curl -X POST https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json \\
    --data-urlencode "To=whatsapp:{to_number}" \\
    --data-urlencode "From=whatsapp:{from_number}" \\
    --data-urlencode "Body=Cálculo finalizado con éxito para {job_name}.sh" \\
    -u {account_sid}:{auth_token}
else
    curl -X POST https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json \\
    --data-urlencode "To=whatsapp:{to_number}" \\
    --data-urlencode "From=whatsapp:{from_number}" \\
    --data-urlencode "Body=Cálculo fallido para {job_name}.sh" \\
    -u {account_sid}:{auth_token}
fi
"""

    with open(f"{job_name}.sh", "w") as f:
        f.write(sh_template.format(job_name=job_name, part=part, nprocshared=nprocshared, 
                                   tiempo=tiempo, commands=commands, twilio_section=twilio_section))

    logging.info(f"Archivo {job_name}.sh creado con éxito.")


# Función main
def main():
    logging.info("\n\n-------NEW SPA TO GAUSS + CREATE SH-------\n")
    files = obtener_archivos_gjc(carpeta_base)

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

    jobtime = int(input("Que tiempo queres? 1 para 12 h, 5 para 24 h, 9 para 28 h: "))
    while jobtime not in [1, 5, 9]:
        print("Tipeaste mal")
        jobtime = int(input("Que tiempo queres?: 1 para 12 h, 5 para 24 h, 9 para 28 h: "))
    tiempo = {1: "12", 5: "24", 9: "48"}[jobtime]        

    folder = input("Folder: ")
    fold = f"g09 /home/{user}/{folder}/" if folder else f"g09 /home/{user}/"

    # Aclaración
    print("Recordar que se trabaja con 1 input por sh.")

    # Preguntar si se desea enviar notificaciones por WhatsApp de los .sh finalizados
    send_whatsapp = input("¿Deseas enviar una notificación por WhatsApp cuando el cálculo termine? (s/n): ").lower() in ["s", "si"]
    
    account_sid = auth_token = from_number = to_number = ""

    if send_whatsapp:
        # Solicitar detalles de Twilio
        account_sid = "AC5db83e00c19e144836600178b72e00e8"
        auth_token = "630b4ea4668d7d0f8c8670fe9d60f181"
        from_number = "+18173857401"
        to_number = input("Ingresa el número de destino (formato: +543419999999): ")
        
        logging.info(f"Se enviarán notificaciones al Whatsapp: {to_number}")
    else:
        print("\nNo se enviarán notificaciones por whatsapp.\n")
        logging.info(f"Sin notificaciones por whatsapp.")

    for file in files:
        crear_archivo_sh(file, part, nprocshared, tiempo, fold, send_whatsapp, account_sid, auth_token, from_number, to_number)

    print("\nFinalizado.")

if __name__ == "__main__":
    main()
