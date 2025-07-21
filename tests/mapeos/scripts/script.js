    // Configuración
      const JSON_URL = "https://raw.githubusercontent.com/brunocelada/pasantia_unr/refs/heads/main/tests/mapeos/downloads.json";

    // Función para cargar el JSON
    async function cargarEstructura() {
      try {
        const response = await fetch(JSON_URL);
        if (!response.ok) throw new Error("Error al cargar el JSON");
        const data = await response.json();
        renderizarEstructura(data);
      } catch (error) {
        document.getElementById("estructura-raiz").innerHTML = `
          <div class="error">
            <p>Error al cargar la estructura: ${error.message}</p>
            <p>Verifica que la URL del JSON sea correcta:</p>
            <code>${JSON_URL}</code>
          </div>
        `;
      }
    }

    // Función para renderizar la estructura
    function renderizarEstructura(data, basePath = "") {
      const contenedor = document.getElementById("estructura-raiz");
      contenedor.innerHTML = generarHTML(data, basePath);
      agregarEventListeners();
      formatearDescripciones();
    }

    // Genera el HTML recursivamente (similar a tu función Python original)
    function generarHTML(node, path, name = "Raíz") {
      let html = `<ul><li>
        <span class="copy-icon" onclick="copyPath('${path}')">📋</span>
        <span class="type-icon">📁</span>
        <span class="carpeta" onclick="toggle(this)">${name}</span>
        <ul>`;
      
      // Subcarpetas
      Object.keys(node).forEach(key => {
        if (!key.startsWith('_') && key !== 'description') {
          const subPath = path ? `${path}/${key}` : key;
          html += generarHTML(node[key], subPath, key);
        }
      });
      
      // Archivos agrupados
      if (node._archivos && node._archivos.length > 0) {
        html += `
          <li>
            <span class="type-icon">🗎</span>
            <span class="archivo-grupo" onclick="toggle(this)">Archivos (${node._archivos.length})</span>
            <ul class="archivos-contenedor">`;
        
        node._archivos.forEach(archivo => {
          const filePath = path ? `${path}/${archivo}` : archivo;
          html += `
            <li>
              <span class="copy-icon" onclick="copyPath('${filePath}')">📋</span>
              <span class="type-icon">${getFileIcon(archivo)}</span>
              <span class="archivo">${archivo}</span>
            </li>`;
        });
        
        html += `</ul></li>`;
      }
      
      // Descripción (tooltip)
      if (node.description) {
        html = html.replace('class="carpeta"', `class="carpeta" title="${node.description}"`);
      }
      
      html += `</ul></li></ul>`;
      return html;
    }

    function getFileIcon(filename) {
    const extension = filename.split('.').pop().toLowerCase();
    const iconMap = {
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
      };
      return iconMap[extension] || iconMap['default'];
    }
    // Procesar descripciones largas al cargar la página
    function formatearDescripciones() {
      document.querySelectorAll('.carpeta[title]').forEach(element => {
        const desc = element.getAttribute('title');
        if (desc.length > 100) { // Si es muy larga
          element.setAttribute('title', 
            desc.slice(0, 100) + '\n\n... (click para ver completo)');
          
          // Mostrar completo al hacer click
          element.addEventListener('click', (e) => {
            if (e.target === element) {
              alert(`Descripción completa:\n\n${desc}`);
            }
          });
        }
      });
    }


    // Función toggle global correctamente definida
    window.toggle = function(elem) {
      const nextUl = elem.nextElementSibling;
      if (nextUl && nextUl.tagName === 'UL') {
        nextUl.style.display = nextUl.style.display === 'block' ? 'none' : 'block';
      }
    };
    
    // Función para decodificar paths
    window.copyPath = function(encodedPath) {
      // Usamos el método decodeURIComponent para manejar caracteres especiales
      let path = encodedPath
        .replace(/\\\\x5c/g, '\\\\')
        .replace(/\\\\x2f/g, '/');
      
      console.log("Copiando ruta:", path);
      
      // Usamos el API del clipboard con fallback
      if (navigator.clipboard) {
        navigator.clipboard.writeText(path).then(() => showToast());
      } else {
        // Fallback para navegadores antiguos
        const textarea = document.createElement('textarea');
        textarea.value = path;
        document.body.appendChild(textarea);
        textarea.select();
        try {
          document.execCommand('copy');
          showToast();
        } catch (err) {
          console.error('Error al copiar:', err);
        }
        document.body.removeChild(textarea);
      }
      
      function showToast() {
        const toast = document.getElementById('toast');
        toast.textContent = 'Ruta copiada';
        toast.className = 'show';
        setTimeout(() => toast.className = '', 1500);
      }
    };

  // Iniciar carga al abrir la página
  document.addEventListener('DOMContentLoaded', cargarEstructura);

  