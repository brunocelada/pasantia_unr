#!/bin/bash
dos2unix "$0" # Convierte el script actual a formato Unix

# --------- PLANTILLA PARA ENVIAR MUCHOS CÁLCULOS AL CLUSTER -------------------------

# 1- Completar correctamente los datos de este archivo .sh
# 2- Guardar el archivo .sh en una carpeta conocida del cluster (por ejemplo "scripts")
# 3- Hacer que el script sea ejecutable escribiendo en la consola lo siguiente:
    # chmod +x /home/$USER/scripts/send_calcs.sh
    # (tener en cuenta que "send_calcs" es el nombre del archivo .sh presente)

# 4A -Opción 1: Útil si los cálculos son cortos y no es necesario cerrar el terminal, ya que
            # los bucles se ejecutan con el programa abierto
#       4A- Ejecutar el script utilizando en la consola el siguiente comando:
            # ./send_calcs.sh

# 4B -Opción 2: En caso de que los cálculos tarden más tiempo, se puede ejecutar
                # periódicamente el script con cron jobs
#       4B- Ejecutar el script utilizando en la consola el siguiente comando:
            # crontab -e
            # Pegá lo siguiente (tene en cuenta que n son min, cambialo por el valor): 
            #   */n * * * * /home/$USER/scripts/send_calcs.sh
            # O pegá lo siguiente (tene en cuenta que n son horas, cambialo por el valor): 
            #   0 * * * * /home/$USER/scripts/send_calcs.sh
            # Guardar los cambios y sali con: :wq

    # Configuración del cron job:
        # Los primeros 5 asteriscos son:
        #     Minuto (0-59)
        #     Hora (0-23)
        #     Día del mes (1-31)
        #     Mes (1-12)
        #     Día de la semana (0-7, donde 0 y 7 representan el domingo)
        # Para salir sin guardar los cambios en cron:
        #     :qa!
        # Para guardar los cambios y salir:
        #     :wq
        # Para ver el listado de cron jobs:
        #     crontab -l
        # Para remover todos los cron jobs (!precaución, es irreversible):
        #     crontab -r

# ------------------------------------------------------------------------------------

# Webhook de Slack
slack_webhook_url="https://hooks.slack.com/services/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"

# Función para enviar notificaciones a Slack
send_slack_notification() {
    local message="$1"
    curl -X POST -H 'Content-type: application/json; charset=utf-8'     --data "{\"text\":\"$message\"}"     "$slack_webhook_url"
}

# Definir el límite máximo de trabajos simultáneos
max_jobs=50

# Archivo de texto que contiene las rutas de los directorios
directories_file="/home/$USER/scripts/directories.txt"

# Archivo para almacenar la cantidad total de archivos
total_files_file="/home/$USER/scripts/total_files.txt"
# Archivo para almacenar los nombres de archivos enviados
sent_files_file="/home/$USER/scripts/sent_files.txt"

# Inicializar archivos txt si no existen
if [ ! -f "$total_files_file" ]; then
    echo "0" > "$total_files_file"
fi
if [ ! -f "$sent_files_file" ]; then
    > "$sent_files_file" # Inicializar como archivo vacío
fi

# Verificar si el archivo de directorios existe y no está vacío
if [ ! -f "$directories_file" ] || [ ! -s "$directories_file" ]; then
    send_slack_notification "El archivo de directorios ($directories_file) no existe o está vacío. Verifique la configuración. (Se eliminará el cron job)"
    (crontab -l | grep -v '/home/$USER/scripts/send_calcs.sh') | crontab -
    exit 1  # Salir del script si no hay directorios válidos
fi

# Leer los directorios del archivo
mapfile -t directorios < "$directories_file"

# Expandir las variables de entorno en las rutas
for i in "${!directorios[@]}"; do
    directorios[$i]=$(eval echo "${directorios[$i]}" | tr -d '\r\n' | xargs)  # Eliminar saltos de línea y espacios en blanco
done

# Contador de archivos .sh que se enviarán
total_scripts=0

# Primero, contar los archivos .sh
for dir in "${directorios[@]}"; do
  # Verificar que el directorio existe
  if [ -d "$dir" ]; then
    # Contar los archivos .sh en el directorio actual
    script_count=$(find "$dir" -maxdepth 1 -name "*.sh" | wc -l)
    total_scripts=$((total_scripts + script_count))
  fi
done

# Guardar el total de archivos en el archivo de marcador
echo "$total_scripts" > "$total_files_file"
echo "Total de archivos a enviar: $total_scripts"

# Contar cuántos archivos ya se han enviado (número de líneas en sent_files.txt)
archivos_enviados=$(wc -l < "$sent_files_file")

# Iterar sobre cada directorio
for dir in "${directorios[@]}"; do
  # Verificar que el directorio existe
  if [ -d "$dir" ]; then
    # Iterar sobre cada archivo .sh en el directorio actual
    for script in "$dir"*.sh; do
      # Verificar si el archivo existe
      if [ -e "$script" ]; then
        # Verificar si el script ya ha sido enviado
        if grep -q "$(basename "$script")" "$sent_files_file"; then
          echo "El trabajo $(basename "$script") ya ha sido enviado, saltando..."
          continue  # Saltar al siguiente archivo
        fi

        # Convertir el archivo a formato Unix antes de enviarlo
        dos2unix "$script"

        # Verificar si el script ya está en la cola
        if squeue -u $USER | grep -q "$(basename "$script")"; then
          echo "El trabajo $(basename "$script") ya está en la cola, saltando..."
        else
          # Controlar cuántos trabajos tienes en la cola
          while [ "$(squeue -u $USER | tail -n +2 | wc -l)" -ge "$max_jobs" ]; do
            echo "El número máximo de trabajos ($max_jobs) está en la cola, esperando que se liberen trabajos..."
            sleep 3600  # Esperar n segundos antes de volver a revisar
          done

          # Enviar el archivo al clúster
          sbatch "$script"
          echo "Trabajo $(basename "$script") enviado."
          echo "$(basename "$script")" >> "$sent_files_file"  # Guardar el nombre en el archivo de enviados
          archivos_enviados=$((archivos_enviados + 1))  # Aumentar el contador de archivos enviados

          # Pausa breve (1 seg) para no enviar trabajos demasiado rápido
          sleep 1
        fi
      else
        echo "No se encontraron archivos .sh en $dir."
      fi
    done
  else
    echo "El directorio $dir no existe."
  fi
done

# Verificar si se enviaron todos los archivos
if [ "$archivos_enviados" -eq "$total_scripts" ]; then
  echo "Se han enviado todos los $total_scripts archivos. Eliminando cron job..."
  send_slack_notification "Se han enviado todos los $total_scripts archivos. El cron job ha sido eliminado correctamente."
  (crontab -l | grep -v '/home/$USER/scripts/send_calcs.sh') | crontab -
else
  echo "Se enviaron $archivos_enviados de $total_scripts archivos. El cron job seguirá activo."
fi