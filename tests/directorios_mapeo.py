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

    # Construir cuerpo UL completo con el nodo raíz
    body = '<ul>\n'
    body += render_node(folder_name, tree, base_path)
    body += '</ul>\n'

    html_page = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <title>Estructura de {html.escape(folder_name)}</title>
  <style>
    :root {{
      --bg-color: #c4bfbf;      /* Fondo claro*/
      --text-color: #464646;    /* Texto oscuro */
      --accent-color: #4a80f0;  /* Azul moderno */
      --folder-color: #6bb9f0;  /* Azul más claro para carpetas */
      --file-color: #a0a0a0;    /* Color archivos */
    }}

    body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background-color: var(--bg-color);
    color: var(--text-color);
    line-height: 1.6;
    margin: 0;
    padding: 20px;
    }}

    #contenedor {{
      max-width: 800px;
      margin: 0 auto;
      background-color: #3a3a3a;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }}

    ul {{ list-style: none; margin:0; padding:0 0 0 20px; border-left:1px solid #ccc; }}
    li {{ margin:4px 0; }}
    .carpeta, .archivo-grupo {{ font-weight: bold; cursor: pointer; }}
    .carpeta {{ color: #2b4b84; }}
    .archivo-grupo {{ color: #555; }}
    .archivo {{ color: #464646; }}
    .error {{ color:red; font-style:italic; }}
    .type-icon {{ margin-right: 5px; display: inline-block; width: 1.2em; }}
    ul ul ul {{ display: none; }}  /* Solo oculta los ul anidados a partir del segundo nivel */
    #estructura-raiz {{ background-color: #edebeb; border-radius: 8px; padding: 20px; 
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); }}  /* Muestra el primer nivel */
    ul.archivos-contenedor {{ padding-left: 10px; border-left: 1px dashed #ddd; }}
    .copy-icon {{ cursor: pointer; margin-right:4px; position: relative; }}
    #toast {{
      visibility: hidden;
      min-width: 120px;
      background-color: #333;
      color: #fff;
      text-align: center;
      border-radius: 4px;
      padding: 8px;
      position: fixed;
      z-index: 1;
      bottom: 30px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 0.9em;
    }}
    #toast.show {{
      visibility: visible;
      animation: fadein 0.3s, fadeout 0.5s 1.2s;
    }}

    /* Tooltip */
    .copy-icon::after {{
      content: "Click para copiar";
      position: absolute;
      bottom: 80%;
      left: 50%;
      transform: translateX(-50%) scale(0.9);
      opacity: 0;
      background-color: #333;
      color: white;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 14px;
      transition: all 0.2s ease;
      pointer-events: none;
      white-space: nowrap;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); /* Sombra más pronunciada */
      z-index: 1000;
      border: 1px solid #444; /* Borde sutil */
    }}
    .copy-icon:hover::after {{
      opacity: 1;
      transform: translateX(-50%) scale(1);
      bottom: 100%;
      text-decoration: none; /* Sin subrayado */
    }}
    /* Flechita inferior opcional (más visible) */
    .copy-icon::before {{
      content: "";
      position: absolute;
      bottom: calc(100% - 8px);
      left: 50%;
      transform: translateX(-50%);
      border-width: 8px 6px 0 6px; /* Más grande */
      border-style: solid;
      border-color: #222 transparent transparent transparent;
      opacity: 0;
      transition: opacity 0.3s ease;
    }}
    .copy-icon:hover::before {{
      opacity: 1;
    }}

    /* Iconos con colores por tipo */
    .file-icon[data-ext="pdf"] {{ color: #ff4d4d; }}
    .file-icon[data-ext="xlsx"], 
    .file-icon[data-ext="csv"] {{ color: #4CAF50; }}
    .file-icon[data-ext="docx"] {{ color: #2196F3; }}
    .file-icon[data-ext="js"] {{ color: #FFD600; }}
    .file-icon[data-ext="rar"],
    .file-icon[data-ext="zip"] {{ color: #FF9800; }}

    /* Efectos hover */
    .carpeta:hover, .archivo-grupo:hover, .copy-icon:hover {{ 
      text-decoration: underline;
      opacity: 0.5;
    }}
    .copy-icon:hover {{
      transform: scale(1.1);
      color: var(--accent-color);
      opacity: 0.7;
      text-decoration: none;
    }}

    /* Transiciones suaves */
    li, span {{
      transition: all 0.2s ease;
    }}

    /* Tooltip para descripciones */
    .carpeta[title] {{
      position: relative;
      cursor: help; /* Cambia el cursor a ? */
    }}
    .carpeta[title]::after {{
      content: attr(title);
      position: absolute;
      bottom: 100%;
      left: 50%;
      transform: translateX(-50%);
      background-color: #333;
      color: white;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 14px;
      white-space: pre-wrap; /* Permite saltos de línea */
      max-width: 300px; /* Ancho máximo */
      width: max-content;
      opacity: 0;
      transition: opacity 0.15s ease; /* Más rápido (0.15s vs 0.3s original) */
      pointer-events: none;
      z-index: 1000;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
      text-align: left;
    }}
    /* Mostrar tooltip inmediatamente al hover */
    .carpeta[title]:hover::after {{
      opacity: 1;
      transition-delay: 0.1s; /* Retraso mínimo */
    }}
    /* Flechita inferior del tooltip */
    .carpeta[title]::before {{
      content: "";
      position: absolute;
      bottom: calc(100% - 8px);
      left: 50%;
      transform: translateX(-50%);
      border-width: 8px 6px 0 6px;
      border-style: solid;
      border-color: #333 transparent transparent transparent;
      opacity: 0;
      transition: opacity 0.15s ease;
    }}
    .carpeta[title]:hover::before {{
      opacity: 1;
      transition-delay: 0.1s;
    }}

    @keyframes fadein {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    @keyframes fadeout {{ from {{ opacity: 1; }} to {{ opacity: 0; }} }}
  </style>


  <body>
    <div id="contenedor">
      <h2>Cargando estructura de archivos...</h2>
      <div id="estructura-raiz"></div>
    </div>

    <div id="toast"></div>


  <script>
    // Configuración
      const JSON_URL = "https://raw.githubusercontent.com/tu_usuario/tu_repo/main/ruta/archivo.json";

      
    function getFileIcon(filename) {{
    const extension = filename.split('.').pop().toLowerCase();
    const iconMap = {{
      // Documentos
      'pdf': '📕',
      'doc': '📘', 'docx': '📘',
      'xls': '📊', 'xlsx': '📊', 'csv': '📊',
      'ppt': '📑', 'pptx': '📑',
      'txt': '📄',
      
      // Código
      'js': '📜', 'py': '📜', 'html': '📜', 'css': '📜',
      
      // Imágenes
      'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️', 'gif': '🖼️',
      
      // Archivos comprimidos
      'zip': '🗜️', 'rar': '🗜️', '7z': '🗜️',
      
      // Default
      'default': '🗎'
      }};
      return iconMap[extension] || iconMap['default'];
    }}
    // Procesar descripciones largas al cargar la página
    function formatearDescripciones() {{
      document.querySelectorAll('.carpeta[title]').forEach(element => {{
        const desc = element.getAttribute('title');
        if (desc.length > 100) {{ // Si es muy larga
          element.setAttribute('title', 
            desc.slice(0, 100) + '\n\n... (click para ver completo)');
          
          // Mostrar completo al hacer click
          element.addEventListener('click', (e) => {{
            if (e.target === element) {{
              alert(`Descripción completa:\n\n${{desc}}`);
            }}
          }});
        }}
      }});
    }}


    // Función toggle global correctamente definida
    window.toggle = function(elem) {{
      const nextUl = elem.nextElementSibling;
      if (nextUl && nextUl.tagName === 'UL') {{
        nextUl.style.display = nextUl.style.display === 'block' ? 'none' : 'block';
      }}
    }};
    
    // Función para decodificar paths
    window.copyPath = function(encodedPath) {{
      // Usamos el método decodeURIComponent para manejar caracteres especiales
      let path = encodedPath
        .replace(/\\\\x5c/g, '\\\\')
        .replace(/\\\\x2f/g, '/');
      
      console.log("Copiando ruta:", path);
      
      // Usamos el API del clipboard con fallback
      if (navigator.clipboard) {{
        navigator.clipboard.writeText(path).then(() => showToast());
      }} else {{
        // Fallback para navegadores antiguos
        const textarea = document.createElement('textarea');
        textarea.value = path;
        document.body.appendChild(textarea);
        textarea.select();
        try {{
          document.execCommand('copy');
          showToast();
        }} catch (err) {{
          console.error('Error al copiar:', err);
        }}
        document.body.removeChild(textarea);
      }}
      
      function showToast() {{
        const toast = document.getElementById('toast');
        toast.textContent = 'Ruta copiada';
        toast.className = 'show';
        setTimeout(() => toast.className = '', 1500);
      }}
    }};

  formatearDescripciones();
  </script>

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