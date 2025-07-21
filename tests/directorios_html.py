import json
import os
import html

'''
Convierte un JSON de estructura de carpetas a HTML interactivo.
- Muestra carpetas y archivos con iconos
- Permite copiar rutas al portapapeles
- Agrupa archivos en un solo elemento por carpeta
- Incluye descripciones de carpetas
- Maneja errores de permisos
- Utiliza JavaScript para interactividad
'''

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
    file_icon = '🗎'

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
            # print(f"Procesando: {file_path} -> {safe_file_path}")
            out += (
                f'        <li>'
                f'<span class="copy-icon" onclick="copyPath(\'{html.escape(safe_file_path)}\')">📋</span> '
                f'<span class="type-icon">{file_icon}</span> '
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
    json_path = input("📄 Ruta del JSON a convertir: ").strip()
    base_path = input("📁 Ruta base de la carpeta original: ").strip().rstrip(os.sep)
    html_path = input("💾 Archivo HTML de salida: ").strip()

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
    body {{ font-family: Arial, sans-serif; }}
    ul {{ list-style: none; margin:0; padding:0 0 0 20px; border-left:1px solid #ccc; }}
    li {{ margin:4px 0; }}
    .carpeta, .archivo-grupo {{ font-weight: bold; cursor: pointer; }}
    .carpeta {{ color: #2b4b84; }}
    .archivo-grupo {{ color: #555; }}
    .archivo {{ color: #777; }}
    .error {{ color:red; font-style:italic; }}
    .type-icon {{ margin-right: 5px; display: inline-block; width: 1.2em; }}
    ul ul ul {{ display: none; }}  /* Solo oculta los ul anidados a partir del segundo nivel */
    #estructura-raiz > ul {{ display: block; }}  /* Muestra el primer nivel */
    ul.archivos-contenedor {{ padding-left: 10px; border-left: 1px dashed #ddd; }}
    .copy-icon {{ cursor: pointer; margin-right:4px; }}
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
    @keyframes fadein {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    @keyframes fadeout {{ from {{ opacity: 1; }} to {{ opacity: 0; }} }}
  </style>

  <script>
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