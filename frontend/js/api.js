// Thin wrapper over the Bhoomi3D REST API.

const BASE = '';

async function get(path, params) {
  const url = new URL(BASE + path, window.location.origin);
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
  }
  const res = await fetch(url);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* keep status */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

async function post(path, body, params) {
  const url = new URL(BASE + path, window.location.origin);
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
  }
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    // FastAPI puts validation failures in `detail`; ours are human-readable
    // sentences meant to be shown to the user verbatim
    throw new Error((data && (data.detail || data.error)) || `${res.status}: ${res.statusText}`);
  }
  return data;
}

export const api = {
  health: () => get('/api/health'),
  site: () => get('/api/site'),
  terrain: (product = 'dem') => get('/api/terrain', { product }),
  groundImage: (params) => get('/api/ground-image', params),
  objects: (params) => get('/api/objects', params),
  object: (id) => get('/api/objects/' + id.split('/').map(encodeURIComponent).join('/')),
  tree: () => get('/api/tree'),
  ulpin: (u) => get('/api/ulpin/' + encodeURIComponent(u)),
  column: (params) => get('/api/column', params),
  at: (params) => get('/api/at', params),
  search: (q) => get('/api/search', { q }),
  validation: (params) => get('/api/validation', params),
  samplePlan: (kind) => get('/api/sample-plan', { kind }),
  previewPlan: (plan, level) => post('/api/preview-plan', plan, { level }),
  buildFromPlan: (plan) => post('/api/build-from-plan', plan),
  metrics: () => get('/api/metrics'),
  pipeline: () => get('/api/pipeline'),

  /** Stream a pipeline re-run. Returns a function that aborts it. */
  runPipeline(onEvent, onDone, onError) {
    const src = new EventSource('/api/pipeline/run');
    src.onmessage = (e) => {
      try { onEvent(JSON.parse(e.data)); } catch { /* ignore malformed frame */ }
    };
    src.addEventListener('end', () => { src.close(); onDone && onDone(); });
    src.onerror = () => {
      // EventSource fires onerror on normal close too, so only report the
      // failure if the stream never reached its explicit end event
      if (src.readyState !== EventSource.CLOSED) {
        src.close();
        onError && onError(new Error('pipeline stream interrupted'));
      }
    };
    return () => src.close();
  },
};
