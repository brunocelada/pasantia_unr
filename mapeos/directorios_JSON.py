import os
import json
import re
from unicodedata import normalize

def leer_description(ruta):
    """Lee el archivo !Description.txt si existe y devuelve su contenido.
    No sensible a mayúsculas/minúsculas."""
    pattern = re.compile(r'^!?desc.*\.txt$', re.IGNORECASE)

    for filename in os.listdir(ruta):
        if pattern.match(filename):
            try:
                with open(os.path.join(ruta, filename), encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                continue
    return ""

def construir_arbol(ruta):
    """Función principal que construye la estructura"""
    nodo = {
        "description": leer_description(ruta),
        "_archivos": []
    }
    
    try:
        with os.scandir(ruta) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    nodo[entry.name] = construir_arbol(entry.path)
                elif not entry.name.startswith('!'):  # Ignorar archivos especiales
                    nodo["_archivos"].append(entry.name)
    except PermissionError:
        nodo["_error"] = "Permiso denegado"
    except Exception as e:
        nodo["_error"] = f"Error: {str(e)}"
    
    return nodo

def guardar_json(arbol, archivo_salida):
    """Guarda el JSON asegurando encoding UTF-8"""
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(arbol, f, indent=2, ensure_ascii=False)

def flatten_dict(d, parent_key='', sep='.'):
    """Helper para debug: aplana el diccionario"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    ruta = input("📂 Ruta de la carpeta a mapear: ").strip()
    salida = input("💾 Archivo JSON de salida: ").strip()

    print(f"\n🧭 Mapeando {ruta}...")
    arbol = construir_arbol(ruta)
    
    # Debug: muestra estadísticas
    desc_vacias = sum(1 for k, v in flatten_dict(arbol).items() 
                     if k.endswith('description') and not v)
    print(f"\n📊 Carpetas con descripción vacía: {desc_vacias}")
    
    guardar_json(arbol, salida)
    print(f"\n✅ JSON guardado en {salida}")



if __name__ == "__main__":
    main()