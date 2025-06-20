import os

def convertir_a_unix(filepath):
    try:
        with open(filepath, 'rb') as f:
            content = f.read()

        new_content = content.replace(b'\r\n', b'\n')

        if content != new_content:
            with open(filepath, 'wb') as f:
                f.write(new_content)
            print(f"[✓] Convertido a formato Unix: {filepath}")
        else:
            print(f"[=] Ya estaba en formato Unix: {filepath}")
    except Exception as e:
        print(f"[!] Error procesando {filepath}: {e}")
        
# Agregar más extensiones si se desea
def recorrer_y_convertir(directorio, extensiones=('.sh'), incluir_subcarpetas=True):
    for root, dirs, files in os.walk(directorio):
        for file in files:
            if file.lower().endswith(extensiones):
                filepath = os.path.join(root, file)
                convertir_a_unix(filepath)
        if not incluir_subcarpetas:
            break

if __name__ == "__main__":
    ruta_objetivo = input("📁 Ingresá la ruta de la carpeta a limpiar: ").strip()
    incluir_sub = input("¿Incluir subcarpetas? (s/n): ").strip().lower() == 's'

    if os.path.isdir(ruta_objetivo):
        recorrer_y_convertir(ruta_objetivo, incluir_subcarpetas=incluir_sub)
    else:
        print("❌ La ruta especificada no es una carpeta válida.")
