const CACHE = 'doctor-vid-v1';
const ASSETS = [
  './',
  './index.html',
  './app.js',
  './manifest.json',
  './model/doctor_vid.onnx',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm',
];

// Instalación: cachea todos los assets
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

// Activación: borra cachés antiguas
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch: sirve desde caché, con fallback a red
self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
