// Variables globales
let currentStructure = null;

// Esperar a que el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
  // Configurar el input de archivo
  const fileInput = document.getElementById('jsonFileInput');
  const fileNameDisplay = document.getElementById('file-name');
  
  fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    fileNameDisplay.textContent = file.name;
    
    const reader = new FileReader();
    reader.onload = function(e) {
      try {
        currentStructure = JSON.parse(e.target.result);
        renderizarEstructura(currentStructure);
        mostrarToast('✅ Estructura cargada correctamente');
      } catch (error) {
        mostrarError('Error al parsear el JSON: ' + error.message);
      }
    };
    reader.onerror = function() {
      mostrarError('Error al leer el archivo');
    };
    reader.readAsText(file);
  });
});

// Función para renderizar la estructura
function renderizarEstructura(data, basePath = "", parentElement = null) {
  const contenedor = parentElement || document.getElementById('estructura-raiz');
  contenedor.innerHTML = '';
  
  if (!data || typeof data !== 'object') {
    contenedor.innerHTML = '<div class="error">El JSON no contiene una estructura válida</div>';
    return;
  }
  
  // Crear elemento raíz
  const rootName = basePath ? basePath.split('/').pop() || 'Raíz' : 'Raíz';
  const rootElement = document.createElement('ul');
  rootElement.className = 'estructura-root';
  
  // Función recursiva para generar HTML
  function generarHTML(node, path, name) {
    // Conteo de elementos
    const subfolders = Object.keys(node).filter(k => !k.startsWith('_') && k !== 'description');
    const nSub = subfolders.length;
    const nArch = node._archivos ? node._archivos.length : 0;
    const conteo = (nSub || nArch) ? ` (${nSub} subcarpetas, ${nArch} archivos)` : '';
    
    // Tooltip de descripción
    const desc = node.description || '';
    const titleAttr = desc ? ` title="${htmlEscape(desc)}"` : '';
    
    // Crear elemento
    let html = `
      <li class="nodo-carpeta">
        <span class="copy-icon" onclick="copiarRuta('${escapeRuta(path)}')">📋</span>
        <span class="type-icon">📁</span>
        <span class="carpeta"${titleAttr} onclick="toggle(this)">${htmlEscape(name)}${conteo}</span>
        <ul class="contenido-carpeta">`;
    
    // Subcarpetas
    subfolders.forEach(key => {
      const subPath = path ? `${path}/${key}` : key;
      html += generarHTML(node[key], subPath, key);
    });
    
    // Archivos agrupados
    if (nArch > 0) {
      html += `
        <li class="grupo-archivos">
          <span class="type-icon">🗎</span>
          <span class="archivo-grupo" onclick="toggle(this)">Archivos (${nArch})</span>
          <ul class="lista-archivos">`;
      
      node._archivos.forEach(archivo => {
        const filePath = path ? `${path}/${archivo}` : archivo;
        const extension = archivo.split('.').pop().toLowerCase();
        html += `
          <li class="archivo-item">
            <span class="copy-icon" onclick="copiarRuta('${escapeRuta(filePath)}')">📋</span>
            <span class="file-icon" data-ext="${extension}">${getFileIcon(extension)}</span>
            <span class="archivo-nombre">${htmlEscape(archivo)}</span>
          </li>`;
      });
      
      html += `</ul></li>`;
    }
    
    html += `</ul></li>`;
    return html;
  }
  
  rootElement.innerHTML = generarHTML(data, basePath, rootName);
  contenedor.appendChild(rootElement);
  
  // Formatear descripciones largas
  formatearDescripciones();
}

// Función para obtener icono según extensión
function getFileIcon(extension) {
  const iconMap = {
    'pdf': '📕',
    'doc': '📘', 'docx': '📘',
    'xls': '📊', 'xlsx': '📊', 'csv': '📊',
    'ppt': '📑', 'pptx': '📑',
    'txt': '📄',
    'js': '📜', 'py': '📜', 'html': '📜', 'css': '📜',
    'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️', 'gif': '🖼️',
    'zip': '🗜️', 'rar': '🗜️', '7z': '🗜️',
    'default': '🗎'
  };
  return iconMap[extension] || iconMap['default'];
}

// Función para mostrar/ocultar elementos
window.toggle = function(elem) {
  const content = elem.parentElement.querySelector('.contenido-carpeta, .lista-archivos');
  if (content) {
    content.style.display = content.style.display === 'none' ? 'block' : 'none';
  }
};

// Función para copiar rutas
window.copiarRuta = function(ruta) {
  navigator.clipboard.writeText(ruta).then(() => {
    mostrarToast('Ruta copiada: ' + ruta);
  }).catch(err => {
    mostrarToast('Error al copiar: ' + err);
  });
};

// Función para mostrar notificaciones
function mostrarToast(mensaje) {
  const toast = document.getElementById('toast');
  toast.textContent = mensaje;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3000);
}

// Función para mostrar errores
function mostrarError(mensaje) {
  const contenedor = document.getElementById('estructura-raiz');
  contenedor.innerHTML = `<div class="error">${mensaje}</div>`;
}

// Funciones auxiliares
function htmlEscape(str) {
  return str.toString()
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escapeRuta(ruta) {
  return ruta.replace(/\\/g, '/').replace(/'/g, "\\'");
}

// Procesar descripciones largas
function formatearDescripciones() {
  document.querySelectorAll('.carpeta[title]').forEach(element => {
    const desc = element.getAttribute('title');
    if (desc.length > 100) {
      element.setAttribute('title', desc.slice(0, 100) + '...');
    }
  });
}