import os
import json

def leer_description(ruta):
    """Lee el archivo !Description.txt si existe y devuelve su contenido."""
    desc_path = os.path.join(ruta, "!Description.txt")
    if os.path.isfile(desc_path):
        try:
            with open(desc_path, encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""

def construir_arbol(ruta):
    """
    Recorre la carpeta y devuelve un dict con esta estructura:
    {
      "description": <texto de !Description.txt o "">,
      "_archivos": [lista de archivos],
      "<subcarpeta1>": { … recursivo … },
      "<subcarpeta2>": { … }
    }
    """
    nodo = {
        "description": leer_description(ruta),
        "_archivos": []
    }
    try:
        with os.scandir(ruta) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    nodo[e.name] = construir_arbol(e.path)
                else:
                    nodo["_archivos"].append(e.name)
    except PermissionError:
        nodo["_error"] = "Permiso denegado"
    return nodo

def guardar_json(arbol, archivo_salida):
    with open(archivo_salida, "w", encoding="utf-8") as f:
        json.dump(arbol, f, indent=2, ensure_ascii=False)

def main():
    ruta = input("📂 Ruta de la carpeta a mapear: ").strip()
    salida = input("💾 Archivo JSON de salida (p. ej. /home/usuario/estructura.json): ").strip()

    print(f"\n🧭 Mapeando {ruta} …")
    arbol = construir_arbol(ruta)
    guardar_json(arbol, salida)
    print(f"\n✅ JSON guardado en {salida}")

if __name__ == "__main__":
    main()
