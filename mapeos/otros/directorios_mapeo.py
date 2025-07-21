import json
import os
import html
import re
import shutil

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

def generar_html(folder_name, json_path, html_path):
    """Genera el archivo HTML completo con CSS y JS integrados."""
    # Obtener solo el nombre del archivo JSON (sin ruta completa)
    json_filename = os.path.basename(json_path)
    
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Estructura de {html.escape(folder_name)}</title>
  <link rel="stylesheet" href="https://raw.githubusercontent.com/brunocelada/pasantia_unr/refs/heads/main/tests/mapeos/styles/style.css">

</head>
<body>
  <input type="file" id="jsonFileInput" accept=".json" />
  <div id="contenedor">
    <h2>Estructura de {html.escape(folder_name)}</h2>
    <div id="estructura-raiz"></div>
  </div>
  <div id="toast"></div>

  <script>
    document.getElementById('jsonFileInput').addEventListener('change', function(e) {{
            const file = e.target.files[0];
            const reader = new FileReader();
            
            reader.onload = function(e) {{
                try {{
                    const data = JSON.parse(e.target.result);
                    renderizarEstructura(data);
                }} catch (error) {{
                    console.error("Error al parsear JSON:", error);
                }}
            }};
            
            reader.readAsText(file);
        }});
   
    // Función para cargar el JSON
    async function cargarEstructura() {{
      try {{
        const response = await fetch(JSON_URL);
        if (!response.ok) throw new Error("Error al cargar el JSON");
        const data = await response.json();
        renderizarEstructura(data);
      }} catch (error) {{
        document.getElementById("estructura-raiz").innerHTML = `
          <div class="error">
            <p>Error al cargar la estructura: ${{error.message}}</p>
            <p>Verifica que la URL del JSON sea correcta:</p>
            <code>${{JSON_URL}}</code>
          </div>
        `;
      }}
    }}

    // Función para renderizar la estructura
    function renderizarEstructura(data, basePath = "") {{
      const contenedor = document.getElementById("estructura-raiz");
      contenedor.innerHTML = generarHTML(data, basePath);
      agregarEventListeners();
      formatearDescripciones();
    }}

    // Genera el HTML recursivamente (similar a tu función Python original)
    function generarHTML(node, path, name = "Raíz") {{
      let html = `<ul><li>
        <span class="copy-icon" onclick="copyPath('${{path}}')">📋</span>
        <span class="type-icon">📁</span>
        <span class="carpeta" onclick="toggle(this)">${{name}}</span>
        <ul>`;
      
      // Subcarpetas
      Object.keys(node).forEach(key => {{
        if (!key.startsWith('_') && key !== 'description') {{
          const subPath = path ? `${{path}}/${{key}}` : key;
          html += generarHTML(node[key], subPath, key);
        }}
      }});
      
      // Archivos agrupados
      if (node._archivos && node._archivos.length > 0) {{
        html += `
          <li>
            <span class="type-icon">🗎</span>
            <span class="archivo-grupo" onclick="toggle(this)">Archivos (${{node._archivos.length}})</span>
            <ul class="archivos-contenedor">`;
        
        node._archivos.forEach(archivo => {{
          const filePath = path ? `${{path}}/${{archivo}}` : archivo;
          html += `
            <li>
              <span class="copy-icon" onclick="copyPath('${{filePath}}')">📋</span>
              <span class="type-icon">${{getFileIcon(archivo)}}</span>
              <span class="archivo">${{archivo}}</span>
            </li>`;
        }});
        
        html += `</ul></li>`;
      }}
      
      // Descripción (tooltip)
      if (node.description) {{
        html = html.replace('class="carpeta"', `class="carpeta" title="${{node.description}}"`);
      }}
      
      html += `</ul></li></ul>`;
      return html;
    }}

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
            desc.slice(0, 100) + '... (click para ver completo)');
          
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

  // Iniciar carga al abrir la página
  document.addEventListener('DOMContentLoaded', cargarEstructura);

  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    base_path = input("📂 Ruta de la carpeta a mapear: ").strip().rstrip(os.sep)
    json_path = input("💾 Archivo JSON de salida (p. ej. /home/estructura.json): ").strip()
    html_path = input("💾 Archivo HTML de salida(p. ej. /home/estructura.html): ").strip()

    print(f"\n🧭 Mapeando {base_path} …")
    arbol = construir_arbol(base_path)
    guardar_json(arbol, json_path)
    print(f"\n✅ JSON guardado en {json_path}")

    folder_name = os.path.basename(base_path) or base_path
    generar_html(folder_name, json_path, html_path)
    print(f"\n✅ HTML generado en {html_path}")

if __name__ == "__main__":
    main()