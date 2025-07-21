import json
import os
import html
import re

'''
Crea un archivo JSON con la estructura de carpetas y archivos de una ruta dada.
- Incluye descripciones de carpetas desde !Description.txt
- Agrupa archivos en un solo elemento por carpeta
- Maneja errores de permisos
- Genera un JSON estructurado para uso posterior

Convierte un JSON de estructura de carpetas a HTML interactivo.
- Muestra carpetas y archivos con iconos
- Permite copiar rutas al portapapeles
- Agrupa archivos en un solo elemento por carpeta
- Incluye descripciones de carpetas
- Maneja errores de permisos
- Utiliza JavaScript para interactividad
'''

# --------------------------------------------------------------------------------
# FUNCIONES PARA CREAR JSON DE ESTRUCTURA DE CARPETAS
# --------------------------------------------------------------------------------

def leer_description(ruta):
    """Lee el archivo !Description.txt si existe y devuelve su contenido.
    No sensible a mayúsculas/minúsculas."""
    pattern = re.compile(r'^!descripci[oó]n\.txt$', re.IGNORECASE)

    for filename in os.listdir(ruta):
        if pattern.match(filename):
            try:
                with open(os.path.join(ruta, filename), encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                continue
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

# --------------------------------------------------------------------------------
# FUNCIONES PARA CREAR HTML DE ESTRUCTURA DE CARPETAS
# --------------------------------------------------------------------------------

def render_node(name, node, path):
    """
    Renderiza un nodo de carpeta completo:
    - icono de copy
    - icono de tipo (carpeta/archivo)
    - nombre + conteos + tooltip de description
    - hijos (subcarpetas y archivos) en un <ul> colapsable
    """
    # Conteos
    subfolders = [k for k in node if k not in ("_archivos", "description", "_error")]
    n_sub = len(subfolders)
    n_arch = len(node.get("_archivos", []))
    conteo = f" ({n_sub} subcarpetas, {n_arch} archivos)" if (n_sub or n_arch) else ""

    # Tooltip de description
    desc = node.get("description", "").strip()
    title_attr = f' title="{html.escape(desc)}"' if desc else ""

    # Rutas escapadas
    safe_name = html.escape(name)
    safe_path = html.escape(path).replace('\\', '\\\\')
    # print(f"Procesando: {safe_name} -> {safe_path}")

    # Iconos (📁 para carpeta, 📄🗎 para archivo)
    folder_icon = '📁'
    file_icon = '🗎' # Icono genérico (ahora se usará el dinámico)

    # Inicio de este nodo (carpeta)
    out = (
        f'<li>\n'
        f'  <span class="copy-icon" onclick="copyPath(\'{safe_path}\')">📋</span> '
        f'  <span class="type-icon">{folder_icon}</span> '
        f'  <span class="carpeta"{title_attr} onclick="toggle(this)">'
        f'{safe_name}{conteo}'
        f'</span>\n'
        f'  <ul>\n'
    )

    # Archivos - ahora agrupados en un solo elemento
    if n_arch > 0:
        out += (
            f'    <li>\n'
            f'      <span class="type-icon">{file_icon}</span> '
            f'      <span class="archivo-grupo" onclick="toggle(this)"> >> Archivos ({n_arch})</span>\n'
            f'      <ul class="archivos-contenedor">\n'
        )
        for archivo in node.get("_archivos", []):
            file_path = os.path.join(path, archivo)
            safe_file_path = html.escape(file_path).replace('\\', '\\\\')
            file_icon = f'<span class="file-icon" data-ext="{html.escape(archivo.split(".")[-1].lower())}">🗎</span>'
            # print(f"Procesando: {file_path} -> {safe_file_path}")
            out += (
                f'        <li>'
                f'<span class="copy-icon" onclick="copyPath(\'{html.escape(safe_file_path)}\')">📋</span> '
                f'{file_icon} '
                f'<span class="archivo">{html.escape(archivo)}</span>'
                f'</li>\n'
            )
        out += '      </ul>\n    </li>\n'

    # Subcarpetas (recursión)
    for sub in subfolders:
        sub_path = os.path.join(path, sub)
        out += render_node(sub, node[sub], sub_path)

    # Errores de permiso
    if "_error" in node:
        out += f'    <li class="error">[ERROR]: {html.escape(node["_error"])}</li>\n'

    # Cierre del nodo
    out += '  </ul>\n</li>\n'
    return out

def main():
    base_path = input("📂 Ruta de la carpeta a mapear: ").strip().rstrip(os.sep)
    json_path = input("💾 Archivo JSON de salida (p. ej. /home/usuario/estructura.json): ").strip()
    html_path = input("💾 Archivo HTML de salida(p. ej. /home/usuario/estructura.html): ").strip()

    print(f"\n🧭 Mapeando {base_path} …")
    arbol = construir_arbol(base_path)
    guardar_json(arbol, json_path)
    print(f"\n✅ JSON guardado en {json_path}")

    with open(json_path, encoding="utf-8") as f:
        tree = json.load(f)

    folder_name = os.path.basename(base_path) or base_path

    # Generar HTML DINÁMICO (que carga el JSON desde URL)
    html_page = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <title>Estructura de {html.escape(folder_name)}</title>
  <link rel="stylesheet" href="">

  <body>
    <div id="contenedor">
      <h2>Estructura de """ + html.escape(folder_name) + """</h2>
      <div id="estructura-raiz"></div>
    </div>
    <div id="toast"></div>

  <script src="scripts/getDates.js"></script>

</head>
<body>
  <h2>Estructura de {html.escape(folder_name)}</h2>
  <div id="estructura-raiz">
    {body}
  </div>
  <div id="toast"></div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_page)

    print(f"\n✅ HTML generado en {html_path}")

if __name__ == "__main__":
    main()