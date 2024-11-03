#!/bin/bash
dos2unix "$0" # Convierte el script actual a formato Unix

# ------------------------- ADVERTENCIA DE USO!!! -----------------------------------
  # El uso de cronjobs, tal como se lo plantea en este script, puede resultar muy cómodo
  # pero potencialmente dificil de cancelar en caso de que haya errores en bucles y el
  # script se ejecute infinitamente, llenando el cluster de ordenes y ralentizando futuros cálculos.
  # Se recomienda utilizar scripts posteriores a este formato. Consultar a Bruno Celada.


# --------- PLANTILLA PARA ENVIAR MUCHOS CÁLCULOS AL CLUSTER -------------------------

  # 1- Completar correctamente los datos de este archivo .sh
  # 2- Guardar el archivo .sh en una carpeta conocida del cluster (por ejemplo "scripts")
  # 3- Crear un archivo llamado directories.txt y escribir una línea por cada ubicación
  #   de una carpeta para enviar los cálculos. Por ejemplo:
  #   /home/$USER/test
  # 4- Hacer que el script sea ejecutable escribiendo en la consola lo siguiente:
      # chmod +x /home/$USER/scripts/send_calcs.sh
      # (tener en cuenta que "send_calcs" es el nombre del archivo .sh presente)

  # 5A -Opción 1: Útil si los cálculos son cortos y no es necesario cerrar el terminal, ya que
              # los bucles se ejecutan con el programa abierto
  #       5A- Ejecutar el script utilizando en la consola el siguiente comando:
              # ./send_calcs.sh

  # 5B -Opción 2: En caso de que los cálculos tarden más tiempo, se puede ejecutar
                  # periódicamente el script con cron jobs
  #       5B- Ejecutar el script utilizando en la consola el siguiente comando:
              # crontab -e
              # Pegá lo siguiente (tene en cuenta que n son min, cambialo por el valor): 
              #   */n * * * * /home/$USER/scripts/send_calcs.sh
              # O pegá lo siguiente (tene en cuenta que n son horas, cambialo por el valor): 
              #   n * * * * /home/$USER/scripts/send_calcs.sh
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

# Webhook de Slack - Código para el canal de Bruno Celada
slack_webhook_url="https://hooks.slack.com/services/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"

# Función para enviar notificaciones a Slack
send_slack_notification() {
    local message="$1"
    curl -X POST -H 'Content-type: application/json; charset=utf-8'     --data "{\"text\":\"$message\"}" "$slack_webhook_url" || echo "Error al enviar notificación a Slack."
}

# Definir el límite máximo de trabajos simultáneos
max_jobs=15
# Actualizar la cantidad actual de trabajos en cada iteración del while
check_current_jobs() {
    actual_jobs=$(squeue -u "$USER" | tail -n +2 | wc -l)
}

# Archivo de texto que contiene las rutas de los directorios
directories_file="/home/$USER/scripts/directories.txt"

# Archivo para almacenar la cantidad total de archivos
total_files_file="/home/$USER/scripts/total_files.txt"
# Archivo para almacenar los nombres de archivos enviados
sent_files_file="/home/$USER/scripts/sent_files.txt"

# Inicializar archivos txt si no existen
> "$total_files_file"  # Limpiar o crear archivo
> "$sent_files_file"  # Limpiar o crear archivo

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

# Función para mapear archivos .sh y .log
map_files() {
    local ext="$1"  # Extensión de archivo (.sh o .log)
    local output_file="$2"
    
    for dir in "${directorios[@]}"; do
        # Verificar que el directorio existe
        if [ -d "$dir" ]; then
            for file in "$dir"/*"$ext"; do
                # Verificar si el archivo existe
                if [ -f "$file" ]; then
                    basename "${file%$ext}" >> "$output_file"  # Agregar nombre sin extensión
                fi
            done
        fi
    done
}

# Función para contar la cantidad de archivos dentro de las carpetas
count_files() {
  local -n total_count_ref="$1" # Usar "-n" para referenciar la variable pasada por nombre
  local total_count_file="$2"

  total_count_ref=$(wc -l < "$total_count_file")
}

# FUNCION GLOBAL para actualizar datos
actualizar_datos() {
  # Mapear archivos .sh a total_files_file y archivos .log a sent_files_file
  map_files ".sh" "$total_files_file"
  map_files ".log" "$sent_files_file"


  # Leer la cantidad total de archivos .sh a enviar
  count_files total_scripts "$total_files_file"
  echo "Total de archivos a enviar: $total_scripts"

  # Contar cuántos archivos ya se han enviado
  count_files archivos_enviados "$sent_files_file"
  echo "Cantidad de archivos ya enviados: $archivos_enviados"

  # Verificar si se enviaron todos los archivos
  if [ "$archivos_enviados" -eq "$total_scripts" ]; then
    echo "Se han enviado todos los $total_scripts archivos. Eliminando cron job..."
    send_slack_notification "Se han enviado todos los $total_scripts archivos. El cron job se eliminará."
    (crontab -l | grep -v '/home/$USER/scripts/send_calcs.sh') | crontab -
    exit 1  # Salir del script
  else
    echo "Se enviaron $archivos_enviados de $total_scripts archivos. El cron job seguirá activo."
  fi
}

# Actualizar toda la información necesaria
check_current_jobs
actualizar_datos 

# Controlar cuántos trabajos hay en cola
if [ "$actual_jobs" -ge "$max_jobs" ]; then
  echo "El número máximo de trabajos ($max_jobs) está en la cola, esperando que se liberen trabajos..."
  exit 1  # Salir del script
fi

# Iterar sobre cada directorio
for dir in "${directorios[@]}"; do
  # Verificar que el directorio existe
  if [ -d "$dir" ]; then
    # Iterar sobre cada archivo .sh en el directorio actual
    for script in "$dir"/*.sh; do
      # Verificar si el archivo existe
      if [ -f "$script" ]; then
        # Nombre del script sin extensión
        script_name=$(basename "${script%.*}")

        # Verificar si el script ya está en la cola
        if squeue -u "$USER" | grep -q "$script_name"; then
          echo "El trabajo $(basename "$script") ya está en cola, saltando..."

        # Verificar si el script ya se envió o si ya comenzó a calcularse (está en sent_files_file.txt)
        elif grep -Fxq "$script_name" "$sent_files_file"; then
          echo "El trabajo $(basename "$script") ya fue enviado, saltando..."
        else
          check_current_jobs
          while [ "$max_jobs" -gt "$actual_jobs" ]; do
            dos2unix "$script" # Convertir el archivo a formato Unix antes de enviarlo
 
            # Enviar el archivo al clúster con el directorio de trabajo
            sbatch "$script"
            echo "Trabajo $(basename "$script") enviado."

            # Actualizar archivos y variables para el siguiente ciclo
            echo "$script_name" >> "$sent_files_file"
            check_current_jobs

            sleep 1 # Pausa breve (1 seg) para no enviar trabajos demasiado rápido
          done
        fi
      else
        echo "No se encontraron archivos .sh en $dir."
      fi
    done
  else
    echo "El directorio $dir no existe."
  fi
done

# Evitar overhead de los archivos temporales y eliminarlos si existen
if [ -f "$total_files_file" ] || [ -f "$sent_files_file" ]; then
    rm -f "$total_files_file" "$sent_files_file"
fi

exit 1  # Salir del script