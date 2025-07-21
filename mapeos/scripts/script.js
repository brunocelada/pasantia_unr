// Variables globales
let currentStructure = null;
let basePath = '';  // Nueva variable global

// Esperar a que el DOM esté listo
document.addEventListener('DOMContentLoaded', function () {
  // Configurar el input de archivo
  const fileInput = document.getElementById('jsonFileInput');
  const fileNameDisplay = document.getElementById('file-name');

  fileInput.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;

    fileNameDisplay.textContent = file.name;

    const reader = new FileReader();
    reader.onload = function (e) {
      try {
        currentStructure = JSON.parse(e.target.result);
        basePath = currentStructure.__base_path || '';
        const nombreRaiz = basePath.split(/[\\/]/).filter(Boolean).pop();
        delete currentStructure.__base_path;
        renderizarEstructura(currentStructure, '', null, nombreRaiz);
        mostrarToast('✅ Estructura cargada correctamente');
      } catch (error) {
        mostrarError('Error al parsear el JSON: ' + error.message);
      }
    };
    reader.onerror = function () {
      mostrarError('Error al leer el archivo');
    };
    reader.readAsText(file);
  });
});

// Función para renderizar la estructura
function renderizarEstructura(data, basePath = "", parentElement = null, nombreRaiz = 'Root') {
  const contenedor = parentElement || document.getElementById('estructura-raiz');
  contenedor.innerHTML = '';

  if (!data || typeof data !== 'object') {
    contenedor.innerHTML = '<div class="error">El JSON no contiene una estructura válida</div>';
    return;
  }

  // Crear elemento raíz
  const rootElement = document.createElement('ul');
  rootElement.id = 'estructura-raiz';

  // 🌱 usar nombreRaiz (nuevo parámetro)
  rootElement.innerHTML = generarHTML(data, basePath, nombreRaiz, true);
  contenedor.appendChild(rootElement);

  // Función recursiva para generar HTML
  function generarHTML(node, path, name, isRoot = false) {
    // Conteo de elementos
    const subfolders = Object.keys(node).filter(k => !k.startsWith('_') && k !== 'description');
    const nSub = subfolders.length;
    const nArch = node._archivos ? node._archivos.length : 0;
    const conteo = (nSub || nArch) ? ` (${nSub} subcarpetas, ${nArch} archivos)` : '';

    // Tooltip de descripción
    const desc = node.description || '';
    const titleAttr = desc ? ` title="${htmlEscape(desc)}"` : '';

    let descHtml = '';
    if (desc.length > 0) {
      const displayStyle = isRoot ? 'block' : 'none';  // Mostrar la descripción solo si es raíz
      let descContent = '';
      if (desc.length > 80) {
        const shortDesc = htmlEscape(desc.slice(0, 80)) + '...';
        const fullDesc = htmlEscape(desc)
          .replace(/\n/g, '<br>')
          .replace(/\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;'); // reemplaza tabulaciones con 4 espacios
        descContent = `
          <span class="desc-corta">${shortDesc}</span>
          <span class="desc-larga" style="display:none;">${fullDesc}</span>
          <button class="ver-mas-btn">Ver más</button>
        `;
      } else {
        descContent = htmlEscape(desc)
          .replace(/\n/g, '<br>')
          .replace(/\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;'); // reemplaza tabulaciones con 4 espacios
      }

      // Ocultar la descripción hasta que se expanda la carpeta
      const descClass = desc.length > 80 ? 'descripcion expandible' : 'descripcion';
      descHtml = `<div class="${descClass}" style="display: ${displayStyle};">${descContent}</div>`;
    }


    // Crear elemento
    let html = `
      <li class="nodo-carpeta">
        <span class="copy-icon" onclick="copiarRuta('${escapeRuta(path)}')">📋</span>
        <span class="type-icon">📁</span>
        <span class="carpeta"${titleAttr} onclick="toggle(this)">${htmlEscape(name)}${conteo}</span>
        ${descHtml}
        <ul class="contenido-carpeta" style="display: ${path === basePath ? 'block' : 'none'};">`;

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
          <ul class="lista-archivos" style="display:none;">`;

      node._archivos.forEach(archivo => {
        const relativeFilePath = `${path}/${archivo}`.replace(/^\/+/, '');
        const extension = archivo.split('.').pop().toLowerCase();
        html += `
          <li class="archivo-item">
            <span class="copy-icon" onclick="copiarRuta('${escapeRuta(relativeFilePath)}')">📋</span>
            <span class="file-icon" data-ext="${extension}">${getFileIcon(extension)}</span>
            <span class="archivo-nombre">${htmlEscape(archivo)}</span>
          </li>`;
      });

      html += `</ul></li>`;
    }

    html += `</ul></li>`;
    return html;
  }

  rootElement.innerHTML = generarHTML(data, basePath, nombreRaiz, true);
  contenedor.appendChild(rootElement);

  // Formatear descripciones largas
  formatearDescripciones();

  if (data.__base_path) {
    delete data.__base_path;
  }
}

// Función para obtener icono según extensión
function getFileIcon(extension) {
  const iconMap = {
    'pdf': '📕',
    'doc': '📘', 'docx': '📘',
    'xls': '📗', 'xlsx': '📗', 'csv': '📗',
    'ppt': '📙', 'pptx': '📙',
    'txt': '📄',
    'js': '📜', 'py': '📜', 'html': '📜', 'css': '📜',
    'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️', 'gif': '🖼️',
    'zip': '🗜️', 'rar': '🗜️', '7z': '🗜️',
    'default': '🗎'
  };
  return iconMap[extension] || iconMap['default'];
}

// Función para mostrar/ocultar elementos
window.toggle = function (elem) {
  const content = elem.parentElement.querySelector('.contenido-carpeta, .lista-archivos');
  const descripcion = elem.parentElement.querySelector('.descripcion');
  if (content) {
    const isHidden = content.style.display === 'none' || content.style.display === '';
    content.style.display = isHidden ? 'block' : 'none';

    if (descripcion) {
      descripcion.style.display = isHidden ? 'block' : 'none';
    }
  }
};

// Función para copiar rutas
window.copiarRuta = function (rutaRelativa) {
  const rutaAbsoluta = basePath ? `${basePath}/${rutaRelativa}`.replace(/\/+/g, '/') : rutaRelativa;
  navigator.clipboard.writeText(rutaAbsoluta).then(() => {
    mostrarToast('Ruta copiada: ' + rutaAbsoluta);
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

document.addEventListener('click', function (e) {
  if (e.target.classList.contains('ver-mas-btn')) {
    const container = e.target.closest('.descripcion.expandible');
    const corta = container.querySelector('.desc-corta');
    const larga = container.querySelector('.desc-larga');

    const expanded = larga.style.display === 'inline';

    if (expanded) {
      corta.style.display = 'inline';
      larga.style.display = 'none';
      e.target.textContent = 'Ver más';
    } else {
      corta.style.display = 'none';
      larga.style.display = 'inline';
      e.target.textContent = 'Ver menos';
    }
  }
});
