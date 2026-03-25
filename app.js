// ============================================================
//  Doctor Vid AI — app.js
//  Inferencia 100% en dispositivo con ONNX Runtime Web
// ============================================================

// ── Importar ONNX Runtime Web (cargado vía CDN en index.html) ──
// Se inyecta dinámicamente para garantizar compatibilidad PWA
(function injectORT() {
  const s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js';
  s.onload = iniciarApp;
  document.head.appendChild(s);
})();

// ── Constantes del modelo ──
const CLASES = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight'];

const INFO_CLASES = {
  'Black Rot': {
    icono: '🟤',
    color: '#c0392b',
    descripcion: 'Podredumbre negra causada por el hongo Guignardia bidwellii. Produce manchas circulares marrones con borde oscuro. Tratamiento: fungicidas con captan o mancozeb en periodo vegetativo.'
  },
  'ESCA': {
    icono: '🟡',
    color: '#d4ac0d',
    descripcion: 'Enfermedad fúngica compleja (Phaeomoniella, Phaeoacremonium). Provoca decoloración interveinal en "atigrado". Sin cura definitiva; podar y quemar madera afectada.'
  },
  'Healthy': {
    icono: '🟢',
    color: '#27ae60',
    descripcion: '¡Hoja sana! No se detectan signos de enfermedad. Continúa con las prácticas de cultivo habituales y monitoriza periódicamente.'
  },
  'Leaf Blight': {
    icono: '🟠',
    color: '#e67e22',
    descripcion: 'Tizón foliar causado por Pseudopezicula tracheiphila. Manchas angulares limitadas por nervios. Tratamiento: cobre o folpet en primavera.'
  }
};

// Media y std de ImageNet (igual que en el entrenamiento)
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

// ── Estado global ──
let session   = null;   // sesión ONNX
let imagenPIL = null;   // ImageBitmap de la imagen seleccionada

// ── Registrar Service Worker (PWA offline) ──
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').then(reg => {
      console.log('SW registrado:', reg.scope);
      mostrarBadgeOffline();
    }).catch(err => console.warn('SW error:', err));
  });
}

// ── Inicializar app: cargar modelo ONNX ──
async function iniciarApp() {
  setEstado('Cargando modelo de IA en el dispositivo…');
  try {
    // Configurar ONNX Runtime para usar WebAssembly
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd       = true;

    session = await ort.InferenceSession.create('./model/doctor_vid.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    setEstado('✅ Modelo listo. Selecciona una hoja para analizar.');
    console.log('Modelo ONNX cargado. Entradas:', session.inputNames);
  } catch (err) {
    setEstado('❌ Error al cargar el modelo. Comprueba que doctor_vid.onnx está en /model/');
    console.error(err);
  }
}

// ── Abrir cámara ──
function abrirCamara() {
  const input = document.getElementById('file-input');
  input.setAttribute('capture', 'environment');   // cámara trasera
  input.click();
}

// ── Abrir galería ──
function abrirGaleria() {
  const input = document.getElementById('file-input');
  input.removeAttribute('capture');
  input.click();
}

// ── Cargar imagen seleccionada ──
async function cargarImagen(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Mostrar preview
  const url       = URL.createObjectURL(file);
  const preview   = document.getElementById('preview');
  const placeholder = document.getElementById('placeholder');

  preview.src     = url;
  preview.style.display  = 'block';
  placeholder.style.display = 'none';

  // Crear ImageBitmap para procesado
  imagenPIL = await createImageBitmap(file);

  // Activar botón de análisis
  document.getElementById('btn-analizar').disabled = false;
  document.getElementById('resultado').style.display = 'none';
  setEstado('Imagen cargada. Pulsa "Analizar hoja".');

  // Reset input para permitir seleccionar la misma foto otra vez
  event.target.value = '';
}

// ── Preprocesado: ImageBitmap → Float32Array [1,3,224,224] ──
function preprocesar(imageBitmap) {
  const SIZE   = 224;
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx    = canvas.getContext('2d');
  ctx.drawImage(imageBitmap, 0, 0, SIZE, SIZE);

  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);  // RGBA Uint8

  // Separar canales y normalizar (igual que torchvision.Normalize)
  const r = new Float32Array(SIZE * SIZE);
  const g = new Float32Array(SIZE * SIZE);
  const b = new Float32Array(SIZE * SIZE);

  for (let i = 0; i < SIZE * SIZE; i++) {
    r[i] = (data[i * 4]     / 255 - MEAN[0]) / STD[0];
    g[i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    b[i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }

  // Concatenar en orden CHW [R, G, B]
  const tensor_data = new Float32Array(3 * SIZE * SIZE);
  tensor_data.set(r, 0);
  tensor_data.set(g, SIZE * SIZE);
  tensor_data.set(b, SIZE * SIZE * 2);

  return new ort.Tensor('float32', tensor_data, [1, 3, SIZE, SIZE]);
}

// ── Softmax ──
function softmax(logits) {
  const max  = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

// ── Analizar imagen ──
async function analizar() {
  if (!session || !imagenPIL) return;

  const btn = document.getElementById('btn-analizar');
  btn.disabled = true;
  setEstado('⚡ Analizando… (puede tardar unos segundos)');

  try {
    // Preprocesar
    const tensor = preprocesar(imagenPIL);

    // Inferencia ONNX
    const feeds   = { imagen: tensor };
    const results = await session.run(feeds);
    const logits  = Array.from(results.logits.data);

    // Probabilidades
    const probs = softmax(logits);
    const maxIdx = probs.indexOf(Math.max(...probs));

    // Mostrar resultado
    mostrarResultado(probs, maxIdx);
    setEstado('');
  } catch (err) {
    setEstado('❌ Error durante el análisis: ' + err.message);
    console.error(err);
  } finally {
    btn.disabled = false;
  }
}

// ── Mostrar resultado en la UI ──
function mostrarResultado(probs, idxMax) {
  const clase    = CLASES[idxMax];
  const info     = INFO_CLASES[clase];
  const confianza = (probs[idxMax] * 100).toFixed(1);

  // Diagnóstico principal
  document.getElementById('diag-icono').textContent    = info.icono;
  document.getElementById('diag-nombre').textContent   = clase;
  document.getElementById('diag-confianza').textContent = `Confianza: ${confianza}%`;

  // Barras de probabilidad (ordenadas de mayor a menor)
  const orden = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
  const barras = document.getElementById('barras');
  barras.innerHTML = '';

  orden.forEach(i => {
    const pct   = (probs[i] * 100).toFixed(1);
    const color = INFO_CLASES[CLASES[i]].color;
    barras.innerHTML += `
      <div class="barra-row">
        <div class="barra-label">
          <span>${INFO_CLASES[CLASES[i]].icono} ${CLASES[i]}</span>
          <span>${pct}%</span>
        </div>
        <div class="barra-bg">
          <div class="barra-fill" style="width:${pct}%; background:${color};"></div>
        </div>
      </div>`;
  });

  // Descripción
  document.getElementById('descripcion').textContent = info.descripcion;

  // Mostrar tarjeta
  const card = document.getElementById('resultado');
  card.style.display = 'block';
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Helpers ──
function setEstado(msg) {
  document.getElementById('estado').textContent = msg;
}

function mostrarBadgeOffline() {
  const b = document.getElementById('badge-offline');
  b.classList.add('show');
  setTimeout(() => b.classList.remove('show'), 3000);
}
