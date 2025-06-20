import os
import logging
from docx import Document
from docx.shared import Pt
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

'''
No olvidar instalar la biblioteca python-docx
'''

def solicitar_direccion_carpeta(mensaje):
    ruta = input(mensaje)
    if not os.path.isdir(ruta):
        raise NotADirectoryError(f"La ruta proporcionada no es válida: {ruta}")
    return os.path.abspath(ruta)

def buscar_archivos_log(carpeta_base):
    archivos_log = []
    for root, _, files in os.walk(carpeta_base):
        for file in files:
            if file.endswith(".log"):
                archivos_log.append(os.path.join(root, file))
    return archivos_log

def verificar_terminacion_correcta(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for line in reversed(lines[-50:]):
            if "Normal termination" in line:
                return True, lines
        return False, lines
    except Exception as e:
        logging.error(f"Error leyendo archivo {path}: {e}")
        return False, []

def extraer_cita(lines):
    try:
        for i in range(len(lines)-1, -1, -1):
            if "Normal termination" in lines[i]:
                index = i
                break
        else:
            return None

        # Buscar cita hacia arriba desde 10 líneas antes
        block = lines[max(0, index-15):index]
        quote_lines = []
        in_quote = False
        for line in reversed(block):
            if line.strip() == '':
                if in_quote:
                    break
                continue
            in_quote = True
            quote_lines.insert(0, line.strip())
        return '\n'.join(quote_lines).strip()
    except Exception as e:
        logging.error(f"Error extrayendo cita: {e}")
        return None

def guardar_citas_en_docx(citas, base_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    doc_count = 1
    for i in range(0, len(citas), 50):
        doc = Document()
        chunk = citas[i:i+50]
        for cita in chunk:
            p = doc.add_paragraph()
            run = p.add_run(cita + "\n\n")
            run.font.size = Pt(11)
        filename = os.path.join(output_dir, f"{base_name}_{doc_count:02}.docx")
        doc.save(filename)
        logging.info(f"Guardado: {filename}")
        doc_count += 1

def main():
    try:
        carpeta_busqueda = solicitar_direccion_carpeta("Ingrese la carpeta a recorrer: ")
        carpeta_salida = solicitar_direccion_carpeta("Ingrese la carpeta donde guardar los DOCX: ")
        nombre_base = input("Ingrese el nombre base para los archivos Word: ").strip()

        archivos_log = buscar_archivos_log(carpeta_busqueda)
        logging.info(f"Se encontraron {len(archivos_log)} archivos .log")

        citas = []
        for archivo in archivos_log:
            terminado, lines = verificar_terminacion_correcta(archivo)
            if not terminado:
                logging.info(f"Archivo sin terminación normal: {archivo}")
                continue
            cita = extraer_cita(lines)
            if cita:
                citas.append(cita)
                logging.info(f"Cita extraída de {archivo}")
            else:
                logging.warning(f"No se encontró cita en {archivo}")

        if citas:
            guardar_citas_en_docx(citas, nombre_base, carpeta_salida)
        else:
            logging.info("No se encontraron citas válidas para guardar.")

    except Exception as e:
        logging.error(f"Error general: {e}")

if __name__ == "__main__":
    main()
