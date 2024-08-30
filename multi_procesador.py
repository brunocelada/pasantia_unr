import glob
import os
import shutil
import subprocess
import sys
import logging 
from openpyxl import Workbook 

# Configuración de logging
logging.basicConfig(filename="registros/script.log", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

# Función para verificar e instalar openpyxl si no está instalado
def verificar_instalar_openpyxl():
    try:
        import openpyxl
        from openpyxl import Workbook
        logging.info("openpyxl ya está instalado.")
    except ImportError:
        print("openpyxl no está instalado. Instalando...")
        logging.info("openpyxl no está instalado. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
        import openpyxl
        from openpyxl import Workbook

# Funcion que elimina los archivos .sh
def eliminar_archivos_sh(carpeta_base):
    try:
        archivos_sh = glob.glob(os.path.join(carpeta_base, "*.sh"))
        for archivo in archivos_sh:
            os.remove(archivo)
            logging.info(f"Archivo eliminado: {archivo}")
    except Exception as e:
        logging.error(f"Error eliminando archivos .sh: {e}")

# Funcion que mueve archivos determinados a una carpeta determinada
def move_file_to_folder(file_path, folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            logging.info(f"Carpeta creada: {folder_name}")
        shutil.copy(file_path, folder_name)
        os.remove(file_path)
        logging.info(f"Archivo movido: {file_path} a {folder_name}")
    except Exception as e:
        logging.error(f"Error moviendo archivo {file_path}: {e}")

# Funcion que verifica la correcta terminación de los archivos .log
def verificar_terminacion_log(carpeta_base, patrones_conocidos, patrones_ignorados):
    os.chdir(carpeta_base)
    
    for file in glob.glob("*.log"):
        try:
            # Ignorar archivos que contengan cualquiera de los patrones en su nombre
            if any(patron in file for patron in patrones_ignorados):
                logging.info(f"Archivo {file} ignorado porque contiene un patrón de la lista ignorada: {patrones_ignorados}")
                continue

            with open(file, "r") as old:
                lines = old.readlines()[-3:]
    
            if not any("Normal termination" in line for line in lines):
                logging.info(f"Relanzar archivo: {file} - No finalizó correctamente")
                print(f"Relanzar archivo: {file} - No finalizó correctamente")
                move_file_to_folder(file, "Termino Mal")
                continue
    
            # Key phrases para buscar frecuencias negativas
            target_phrases = ["NImag=1", "NImag=\n1", "NIm\nag=1", "N\nImag=1", "NImag\n=1", "NIma\ng=1", "NI\nmag=1"]

            # Verificar si el archivo tiene un patrón en su nombre que debe evitarse
            if not any(patron in file for patron in patrones_conocidos):
                with open(file, "r") as old:
                    content = old.read().replace("\n", "")

                # Si no tiene un prefijo a evitar, evaluar si tiene frecuencias negativas
                if any(phrase in content for phrase in target_phrases):
                    logging.info(f"Relanzar archivo: {file} - Frecuencias negativas")
                    print(f"Relanzar archivo: {file} - Frecuencias negativas")
                    move_file_to_folder(file, "Frecuencias Negativas")

        except Exception as e:
            logging.error(f"Error procesando archivo {file}: {e}")

# Funcion que renombra archivos .out a .log
def renombrar_archivos_out_a_log(carpeta_base):
    os.chdir(carpeta_base)
    try:
        for file in glob.glob("*.out"):
            os.rename(file, (file.rsplit(".", 1)[0]) + ".log")
            logging.info(f"Archivo renombrado: {file}")
    except Exception as e:
        logging.error(f"Error renombrando archivo {file}: {e}")

# Funcion que verifica si hay archivos .gjc para relanzar
def procesar_archivos_gjc(carpeta_base):
    os.chdir(carpeta_base)
    gjc_list = []
    log_list = []

    for file in glob.glob("*.log"):
        log_list.append(file)
        gjc_list.append(file.rsplit(".", 1)[0] + ".gjc")

    logging.info(f"Archivos .log encontrados: {gjc_list}")

    for file in glob.glob("*.gjc"):
        try:
            if file not in gjc_list:
                logging.info(f"No se encontró el archivo .log correspondiente a: {file}")
                move_file_to_folder(file, "Relanzar")
        except Exception as e:
            logging.error(f"Error procesando archivo .gjc: {e}")

# Funcion que crea archivos Excel para los .log correctamente procesados
def crear_excel(carpeta_base, patrones_ignorados):
    carpetas = glob.glob(os.path.join(carpeta_base, "*"))
    
    for carpeta in carpetas:
        nombre_carpeta = os.path.basename(carpeta)

        if nombre_carpeta in patrones_ignorados:
            continue

        os.chdir(carpeta)
        renombrar_archivos_out_a_log(carpeta)
        
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = nombre_carpeta
        sheet.append(["Compound_Name", "SCF", "ZPE", "Enthalpie", "Gibbs"])

        for file in glob.glob("*.log"):
            try:
                with open(file, "r") as old_file:
                    rline = old_file.readlines()

                    # Encontrar la última línea con "SCF Done:"
                    scf_line = next((line for line in reversed(rline) if "SCF Done:" in line), None)
                    # Encontrar la última línea con "Sum of electronic and thermal Free Energies="
                    gibbs_line_index = next((i for i, line in enumerate(reversed(rline)) if "Sum of electronic and thermal Free Energies=" in line), None)

                    if scf_line is None:
                        logging.error(f"No se encontró la energía SCF en el archivo {file}")
                        continue

                    # Extraer el valor de SCF de la línea encontrada
                    try:
                        scf_value = scf_line.split()[4]  # Extraemos el valor SCF
                    except IndexError:
                        logging.error(f"Error extrayendo el valor SCF en el archivo {file}")
                        continue

                    if gibbs_line_index is None:
                        # Si no se encuentra la línea de Gibbs, escribir solo SCF
                        sheet.append([(file.rsplit(".", 1)[0]), scf_value, "-", "-", "-"])
                    else:
                        # Extraer ZPE, entalpía y Gibbs
                        gibbs_line_index = len(rline) - gibbs_line_index - 1  # Convertimos el índice de reversed a índice normal
                        try:
                            wzpe = rline[gibbs_line_index - 3].split()
                            wentalpie = rline[gibbs_line_index - 1].split()
                            wgibbs = rline[gibbs_line_index].split()
                            sheet.append([(file.rsplit(".", 1)[0]), scf_value, wzpe[6], wentalpie[6], wgibbs[7]])
                        except IndexError:
                            logging.error(f"Error extrayendo ZPE, entalpía o Gibbs en el archivo {file}")

                logging.info(f"Datos añadidos del archivo {file} al Excel '{nombre_carpeta}'")
            except Exception as e:
                logging.error(f"Error procesando archivo {file} para Excel {nombre_carpeta}: {e}")


        # Ajustar el ancho de las columnas
        for col in sheet.columns:
            max_length = max(len(str(cell.value)) for cell in col if cell.value)
            adjusted_width = max_length + 2
            sheet.column_dimensions[col[0].column_letter].width = adjusted_width

        workbook.save(f"{nombre_carpeta}.xlsx")
        logging.info(f"Archivo Excel guardado: {nombre_carpeta}.xlsx")

# Funcion que organiza los archivos en carpetas con prefijos seleccionados
def organizar_archivos(carpeta_base, patrones_conocidos, patrones_ignorados):
    archivos = os.listdir(carpeta_base)
    carpetas_creadas = {}

    for archivo in archivos:
        try:
            # Ignorar archivos que contengan cualquiera de los patrones en su nombre
            if any(patron in archivo for patron in patrones_ignorados):
                logging.info(f"Archivo {archivo} ignorado porque contiene un patrón de la lista ignorada: {patrones_ignorados}")
                continue

            for patron in patrones_conocidos:
                if patron in archivo:
                    nombre_carpeta = archivo.split("-")[0] + "-" + patron
                    break
            else:
                nombre_carpeta = archivo.split("-")[0]

            if nombre_carpeta not in carpetas_creadas and nombre_carpeta not in patrones_ignorados:
                nueva_carpeta = os.path.join(carpeta_base, nombre_carpeta)
                os.makedirs(nueva_carpeta)
                carpetas_creadas[nombre_carpeta] = nueva_carpeta

            if nombre_carpeta not in patrones_ignorados:
                shutil.move(os.path.join(carpeta_base, archivo), carpetas_creadas[nombre_carpeta])
                logging.info(f"Archivo {archivo} movido a: {carpetas_creadas[nombre_carpeta]}")

        except Exception as e:
            logging.error(f"Error organizando archivo {archivo}: {e}")

# Funcion principal
def main():
    # Registrar un nuevo lanzamiento del script
    logging.info("\n\n-------NEW MULTI_PROCESADOR FINAL-------\n")

    carpeta_base = "C:\\Linux"
    
    # Lista de patrones conocidos
    patrones_conocidos = ["TS"]
    
    # Carpetas que no se procesarán
    patrones_ignorados = ["Frecuencias Negativas", "Fre", "Relanzar", "Rel", "Termino Mal", "Ter"]

    verificar_instalar_openpyxl()
    eliminar_archivos_sh(carpeta_base)
    verificar_terminacion_log(carpeta_base, patrones_conocidos, patrones_ignorados)
    renombrar_archivos_out_a_log(carpeta_base)
    procesar_archivos_gjc(carpeta_base)
    organizar_archivos(carpeta_base, patrones_conocidos, patrones_ignorados)
    crear_excel(carpeta_base, patrones_ignorados)

    print("\nProcesamiento finalizado\n")

if __name__ == "__main__":
    main()
