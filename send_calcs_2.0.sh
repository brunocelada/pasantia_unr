#!/bin/bash
dos2unix "$0" # Convierte el script actual a formato Unix

# ACLARACIÓN 1: Este script sirve para enviar cálculos simplemente a través del comando:
#   sbatch archivo.sh. Si se desea enviar a algun nodo en particular, como en las computadoras
#   de orgánica, debe modificarse. (aún no me tomé el tiempo de hacerlo)

# ---------------------- PLANTILLA PARA ENVIAR MUCHOS CÁLCULOS AL CLUSTER ----------------------

  # SCRIPT HECHO POR BRUNO CELADA - 4TO AÑO LIC. EN QUÍMICA | FBIOYF - UNR | Pasantía en IQUIR-Computacional
  # Este script funciona correctamente al mandar los cálculos con el mismo nombre en el job_name, archivo .sh y archivo .log,
  # ya que verifica de ésta manera si se enviaron los calculos correctamente. En caso de necesitarse, pedir a Bruno el 
  # script "Spa_Gaus_Create_SH_Slack.py" que realiza esta creación de archivos .sh apropiados para este script automáticamente.

  # PASOS -------------------------------------------------------------------------------------
    # 1- Completar correctamente los datos de este archivo .sh:
      # - Webhook del canal de Slack para enviar notificaciones

    # 2- Guardar este archivo .sh en una carpeta conocida del cluster (por ejemplo "scripts")
    #   Los siguientes archivos .txt, mencionados en los puntos 3 y 4, crearlos y guardalos
    #   en esa misma carpeta.
    
    # 3- Crear un archivo llamado directories.txt y escribir una línea por cada ubicación
    #   de una carpeta para enviar los cálculos. Por ejemplo:
    #   /home/$USER/test
    
    # 4- Crear un archivo llamado max_jobs.txt y escribir solamente el número máximo de
    #   trabajos que desea que el script verifique antes de enviar más cálculos. De esta
    #   manera, solo se envían nuevos cálculos si hay menos que el máximo establecido en el txt.
    
    # 5- Hacer que el script sea ejecutable escribiendo en la consola lo siguiente:
        # chmod +x /home/$USER/scripts/send_calcs_2.0.sh
        # (tener en cuenta que "send_calcs_2.0" es el nombre del archivo .sh presente)
    
    # 6 Ejecutar el script utilizando en la consola el siguiente comando:
      #   nohup /home/$USER/scripts/send_calcs_2.0.sh &
      # Esto permite que se ejecute autónomamente en segundo plano, incluso si se cierra la terminal. 
      # Para verificar si el script está corriendo y evitar ejecuciones simultáneas,
      # ejecutar el siguiente código en la terminal:
      #   ps aux | grep send_calcs_2.0.sh
    
    # 7 Para detener todos los procesos, utiliza:
      #   pkill -f send_calcs_2.0.sh
      # Para detener solo 1 proceso en específico, ubica el PID y eliminalo:
      #   kill -9 PID     <----(reemplazá PID por el número correspondiente)

  # Extra:
    # cat "$temp_dir/script_log_$(date '+%Y-%m-%d').txt"  <--- Para mostrar el log del día actual

# ------------------------------------------------------------------------------------------------------------------------------------

# SECTION: Logging --------------------------------------------------------------------------------------------------------------

# Carpeta de archivos temporales (la crea en caso de que no exista, pero no borra si ya está creada)
temp_dir="/home/$USER/scripts/temp_logs"
mkdir -p "$temp_dir"

# Función para registrar eventos en un archivo de log con fecha (1 log por cada día distinto)
log_event() {
    local message="$1"
    local log_file="$temp_dir/script_log_$(date '+%Y-%m-%d').txt"  # Nombre del archivo de log basado en la fecha
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $message" >> "$log_file"
}

find "$temp_dir" -name "script_log_*.txt" -mtime +7 -exec rm {} \;  # Elimina logs más antiguos de 7 días

# SECTION: Webhook --------------------------------------------------------------------------------------------------------------

# Webhook de Slack - Código para el canal de Bruno Celada
slack_webhook_url="https://hooks.slack.com/services/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"

# Función para enviar notificaciones a Slack
send_slack_notification() {
  local message="$1"
  if curl -X POST -H 'Content-type: application/json; charset=utf-8'     --data "{\"text\":\"$message\"}" "$slack_webhook_url"; then
        log_event "Notificación enviada a Slack: $message"
    else
        log_event "Error al enviar notificación a Slack: $message"
  fi
}

# SECTION: Max and actual Jobs --------------------------------------------------------------------------------------------------------------

# Archivo de texto que contiene la cantidad de trabajos máximos para enviar al cluster
max_jobs_file="/home/$USER/scripts/max_jobs.txt"
max_jobs=100

# Definir el límite máximo de trabajos simultáneos
check_max_jobs() {
  # Verificar si el archivo de trabajos máximos existe y no está vacío
  if [ ! -f "$max_jobs_file" ] || [ ! -s "$max_jobs_file" ]; then
    send_slack_notification "El archivo de trabajos máximos ($max_jobs_file) no existe o está vacío. Se utilizará un máximo de 100 trabajos por usuario."
    log_event "El archivo de trabajos máximos no existe o está vacío. Se utilizará un máximo de 100 trabajos por usuario."
    max_jobs=100
  else
    max_jobs=$(head -n 1 "$max_jobs_file" | tr -d '[:space:]')
    log_event "Límite máximo de trabajos simultáneos: $max_jobs"
  fi
}
check_max_jobs

# SECTION: Directories --------------------------------------------------------------------------------------------------------------

# Archivo de texto que contiene las rutas de los directorios
directories_file="/home/$USER/scripts/directories.txt"
directorios=()

map_directories() {
  # Verificar si el archivo de directorios existe y no está vacío
  if [ ! -f "$directories_file" ] || [ ! -s "$directories_file" ]; then
    send_slack_notification "El archivo de directorios ($directories_file) no existe o está vacío. Verifique la configuración."
    log_event "El archivo de directorios ($directories_file) no existe o está vacío. Verifique la configuración."
    exit 1  # Salir del script si no hay directorios válidos
  fi

  # Leer los directorios del archivo
  mapfile -t directorios < "$directories_file"
  # Expandir las variables de entorno en las rutas
  for i in "${!directorios[@]}"; do
    directorios[$i]=$(eval echo "${directorios[$i]}" | tr -d '\r\n' | xargs)  # Eliminar saltos de línea y espacios en blanco
    log_event "Directorio añadido: ${directorios[$i]}"
  done
}
map_directories

# Inicializar arrays para almacenar los nombres de archivos enviados, totales y en cola
declare -a archivos_enviados_array
declare -a archivos_totales_array
declare -a archivos_en_cola_array

# Función para mapear archivos .sh y .log
map_files() {
  local ext="$1"  # Extensión de archivo (.sh o .log)
  declare -n output_array="$2"  # Variable referencial para asignar al array correcto
    
  for dir in "${directorios[@]}"; do
    # Verificar que el directorio existe
    j=0
    if [ -d "$dir" ]; then
      # Encuentra archivos con la extensión especificada y los añade al array
      found_files=("$dir"/*"$ext")
      for file in "${found_files[@]}"; do
        # Verificar si el archivo existe
        if [ -f "$file" ]; then
          output_array+=("$(basename "${file%$ext}")") # Agregar solo el nombre sin ruta ni extensión
          j=$((j + 1))
        fi
      done
    fi
    log_event "($j) archivos encontrados en $dir con extensión $ext: $(IFS=';'; echo "${output_array[*]}")"
  done
}

# SECTION: Functions --------------------------------------------------------------------------------------------------------------

# Función para verificar si un elemento está en un array
contains_element() {
  local element="$1"
  shift
  for item in "$@"; do
    if [[ "$item" == "$element" ]]; then
      return 0  # Elemento encontrado
    fi
  done
  return 1  # Elemento no encontrado
}

# FUNCION GLOBAL para actualizar datos
actualizar_datos() {
  directorios=() # Limpia el arreglo
  map_directories # Llama a la función para rellenar el arreglo nuevamente

  # Reiniciar las array a un estado vacío
  archivos_enviados_array=()
  archivos_totales_array=()
  archivos_en_cola_array=()

  # Mapear archivos .sh a archivos_totales_array y archivos .log a archivos_enviados_array
  map_files ".sh" "archivos_totales_array"
  map_files ".log" "archivos_enviados_array"

  # Lee los nombres de trabajos en cola y los categoriza
  mapfile -t archivos_en_cola_array < <(squeue -u "$USER" --noheader --format="%j") 

  # Estas lineas son importantes porque el programa reconoce 2 tipos de archivos en cola: 
    # (1) Archivos enviados pero que no empezaron a calcularse (no tienen un .log asociado)
    # (2) Archivos enviados y comenzaron su cálculo (ya tienen un .log asociado)
  for job_name in "${archivos_en_cola_array[@]}"; do
    # Agregar a enviados si es archivo total y aún no tiene .log asociado
    if ! contains_element "$job_name" "${archivos_enviados_array[@]}" && contains_element "$job_name" "${archivos_totales_array[@]}"; then
      archivos_enviados_array+=("$job_name") # Agregar el script a la lista de enviados
    fi
  done

  log_event "Lista de archivos ya enviados (${#archivos_enviados_array[@]}): $(IFS=';'; echo "${archivos_enviados_array[*]}")"
  log_event "Lista de trabajos totales (${#archivos_totales_array[@]}): $(IFS=';'; echo "${archivos_totales_array[*]}")"
  log_event "Lista de trabajos actualmente en cola (${#archivos_en_cola_array[@]}): $(IFS=';'; echo "${archivos_en_cola_array[*]}")"

  # Revisa cuántos trabajos hay actualmente en cola.
  actual_jobs=${#archivos_en_cola_array[@]}

  # Contar la cantidad total de archivos .sh a enviar
  total_scripts=${#archivos_totales_array[@]}

  # Contar cuántos archivos ya se han enviado
  archivos_enviados=${#archivos_enviados_array[@]}
}

# SECTION: Cycles of send_calcs --------------------------------------------------------------------------------------------------------------

sleep_time=3600 # Pausa de n segundos antes de volver a verificar las condiciones
last_check_time=$(date +%s)  # Inicializamos con el tiempo actual

while true; do
  actualizar_datos
  # Llamar a check_max_jobs periódicamente solo si es necesario
  if [[ $(date +%s) -gt $((last_check_time + sleep_time / 2)) ]]; then
    check_max_jobs  # Solo actualiza max_jobs si han pasado n/2 segundos
    last_check_time=$(date +%s) # Actualiza el tiempo de última revisión
  fi

  # Verificar si se enviaron todos los archivos
  if [ "$archivos_enviados" -eq "$total_scripts" ]; then
    send_slack_notification "<<< Se han enviado todos los $total_scripts archivos. Finalizó el script. >>>"
    break
  else
    log_event "Se enviaron $archivos_enviados de $total_scripts archivos. El script seguirá activo."

    # Si tengo menos cálculos en cola que el máximo establecido, enviar nuevos cálculos hasta alcanzar el máximo.
    if [ "$actual_jobs" -lt "$max_jobs" ]; then
      # Iterar sobre cada directorio
      for dir in "${directorios[@]}"; do
        if [ "$actual_jobs" -ge "$max_jobs" ]; then
          log_event "Se alcanzó el límite de trabajos permitidos (no se revisarán más carpetas). Pausa de "$sleep_time" segundos y re-inicio del bucle..."
          sleep "$sleep_time"
          break
        fi
        # Verificar que el directorio existe
        if [ -d "$dir" ]; then
          log_event "El directorio "$dir" será examinado"
          # Iterar sobre cada archivo .sh en el directorio actual
          for script in "$dir"/*.sh; do
            # Verificar si el archivo existe
            if [ -f "$script" ]; then
              log_event "El script "$script" será examinado"
              # Nombre del script sin extensión
              script_name=$(basename "${script%.*}")

              # Verificar si el script ya está en la cola (está en archivos_en_cola_array)
              if contains_element "$script_name" "${archivos_en_cola_array[@]}"; then
                log_event "El trabajo $(basename "$script") ya está en cola, saltando..."

              # Verificar si el script ya se envió o si ya comenzó a calcularse (está en archivos_enviados_array)
              elif contains_element "$script_name" "${archivos_enviados_array[@]}"; then
                log_event "El trabajo $(basename "$script") ya fue enviado, saltando..."

              else
                if [ "$max_jobs" -gt "$actual_jobs" ]; then
                  dos2unix "$script" # Convertir el archivo a formato Unix antes de enviarlo
      
                  # Enviar el archivo al clúster con el directorio de trabajo
                  if sbatch "$script"; then
                    log_event "Trabajo $(basename "$script") enviado exitosamente."
                  else
                    log_event "Error al enviar el trabajo $(basename "$script")."
                  fi

                  # Actualizar archivos y variables para el siguiente ciclo
                  archivos_enviados_array+=("$script_name") # Agregar el script a la lista de enviados
                  actual_jobs=$((actual_jobs + 1))

                  sleep 5 # Pausa breve (5 seg) para no enviar trabajos demasiado rápido
                else
                  log_event "Se alcanzó el límite de trabajos permitidos (no se revisarán más los archivos .sh de "$dir")."
                  break
                fi
              fi
            else
              log_event "No se encontraron archivos .sh en $dir."
            fi
          done
        else
          log_event "El directorio $dir no existe."
        fi
      done
    else
      # Al tener igual o más cálculos en cola que el máximo establecido, se espera 1 hora hasta volver a checkear.
      log_event "Hay más trabajos en cola que el máximo permitido. Revisando nuevamente en 1 hora..."
      sleep "$sleep_time" # Pausa
    fi
  fi
done

# Crea un logging de archivos temporales y guarda las array en la carpeta de logging.
timestamp=$(date +%Y%m%d_%H%M%S)
sent_files_file="$temp_dir/sent_files_$timestamp.txt"
total_files_file="$temp_dir/total_files_$timestamp.txt"
printf "%s\n" "${archivos_enviados_array[@]}" > "$sent_files_file"
printf "%s\n" "${archivos_totales_array[@]}" > "$total_files_file"

# --------------------------------------------------------------
exit 0  # Salir del script