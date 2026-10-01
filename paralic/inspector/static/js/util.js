// Small helpers shared by the panels: building DOM, formatting numbers and
// times, the colours, sharp canvases on high-DPI screens and the tooltip.

// Categorical series colours, in their fixed order (checked for colour-vision
// deficiencies on the plot surface). An entity keeps its colour in every panel.
export const S = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];

export const C = {
  plot: '#101a35',
  text: '#eef2ff',
  muted: '#a4b0d4',
  faint: '#6f7ba3',
  grid: 'rgba(148, 163, 214, 0.10)',
  axis: 'rgba(148, 163, 214, 0.30)',
  accent: '#5eead4',
  good: '#0ca30c',
  warning: '#fab219',
  serious: '#ec835a',
  critical: '#d03b3b',
  noface: '#3a4566',
};

export const EYE = { left: S[0], right: S[1] };
export const GAZE = { cursor: S[0], raw: S[1], target: S[2], click: S[4] };
export const HEAD = { yaw: S[0], pitch: S[1], roll: S[2] };
export const PHASE = {
  cal: S[0], head: S[1], val: S[2], adjust: S[3], ft: S[4],      // eye mode
  hspan: S[0], hpoint: S[1], hpinch: S[2],                        // hand setup
};
export const PHASE_NAMES = {
  cal: 'calibration dots', head: 'head movements', val: 'validation dots', adjust: 'quick-adjust dots',
  ft: 'learning samples', hspan: 'hand size', hpoint: 'pointing dots', hpinch: 'pinch',
};

/** Build an element: h('div', {class: 'x', onclick}, 'text', child...). Strings become text nodes. */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') {
      for (const [prop, value] of Object.entries(v)) {
        if (prop.startsWith('--')) el.style.setProperty(prop, value);     // custom properties need setProperty
        else el.style[prop] = value;
      }
    } else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children) {
    if (c == null || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

/** SVG element with attributes. */
export function svg(tag, attrs = {}, ...children) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  for (const c of children) if (c != null) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

/** Replace an element's children (null and false are left out, as in h()). */
export function fill(el, ...children) {
  append(clear(el), children);
  return el;
}

// -- numbers and times ------------------------------------------------------------

export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export function num(v, digits = 1) {
  if (!isNum(v)) return '—';
  return v.toFixed(digits);
}

export function signed(v, digits = 1) {
  if (!isNum(v)) return '—';
  return (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(digits);
}

/** A value of any size, readably: tiny values in scientific notation rather than 0.0. */
export function smart(v) {
  if (!isNum(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.001) return v.toFixed(3);
  return v.toExponential(1).replace(/-/g, '−');
}

export function px(v, digits = 0) {
  return isNum(v) ? `${v.toFixed(digits)} px` : '—';
}

export function xy(p, digits = 0) {
  return Array.isArray(p) && isNum(p[0]) ? `${p[0].toFixed(digits)}, ${p[1].toFixed(digits)}` : '—';
}

/** Seconds as m:ss.mmm (h:mm:ss.mmm past an hour). */
export function clock(sec, ms = true) {
  if (!isNum(sec)) return '—';
  const neg = sec < 0;
  let s = Math.abs(sec);
  const hrs = Math.floor(s / 3600);
  s -= hrs * 3600;
  const min = Math.floor(s / 60);
  s -= min * 60;
  const whole = Math.floor(s);
  const frac = ms ? '.' + String(Math.floor((s - whole) * 1000 + 1e-6)).padStart(3, '0') : '';
  const body = hrs ? `${hrs}:${String(min).padStart(2, '0')}:${String(whole).padStart(2, '0')}`
    : `${min}:${String(whole).padStart(2, '0')}`;
  return (neg ? '−' : '') + body + frac;
}

export function duration(sec) {
  if (!isNum(sec)) return '—';
  if (sec < 60) return `${sec.toFixed(1)} s`;
  if (sec < 3600) return `${Math.floor(sec / 60)} min ${Math.round(sec % 60)} s`;
  return `${Math.floor(sec / 3600)} h ${Math.round((sec % 3600) / 60)} min`;
}

export function bytes(n) {
  if (!isNum(n)) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let k = 0;
  while (n >= 1000 && k < units.length - 1) {
    n /= 1000;
    k++;
  }
  return `${n.toFixed(n >= 100 || k === 0 ? 0 : 1)} ${units[k]}`;
}

export function count(n) {
  return isNum(n) ? n.toLocaleString('en-US') : '—';
}

/** Index of the last element <= x in a sorted array (-1 if none). */
export function bisect(arr, x) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Nice tick step for a span (1, 2, 5 x 10^k). */
export function niceStep(span, target = 5) {
  const raw = span / Math.max(target, 1);
  const p = 10 ** Math.floor(Math.log10(raw || 1));
  const f = raw / p;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
}

export function ticks(lo, hi, target = 5) {
  if (!(hi > lo)) return [lo];
  const step = niceStep(hi - lo, target);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

// -- colours ---------------------------------------------------------------------

function hexRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export function alpha(hex, a) {
  const [r, g, b] = hexRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

// Diverging scale for signed values: blue (negative) - neutral grey - red (positive).
const NEG = hexRgb('#6da7ec');
const MID = hexRgb('#383b47');
const POS = hexRgb('#e66767');

export function diverging(v) {
  if (!isNum(v)) return '#202840';
  const t = clamp(v, -1, 1);
  const end = t < 0 ? NEG : POS;
  const f = Math.abs(t) ** 0.8;
  const c = MID.map((m, k) => Math.round(m + (end[k] - m) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

// -- canvases ---------------------------------------------------------------------

/** Size a canvas for its CSS box on this screen; returns the 2-D context in CSS pixels. */
export function fitCanvas(canvas, width, height) {
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const w = Math.max(1, Math.round(width));
  const hgt = Math.max(1, Math.round(height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(hgt * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(hgt * dpr);
  }
  canvas.style.width = `${w}px`;
  canvas.style.height = `${hgt}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** Call fn at most once per animation frame (with the latest arguments). */
export function perFrame(fn) {
  let queued = false;
  let args = [];
  return (...a) => {
    args = a;
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      fn(...args);
    });
  };
}

export function debounce(fn, ms) {
  let timer = null;
  return (...a) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...a), ms);
  };
}

// -- tooltip ----------------------------------------------------------------------

let tipEl = null;

/** Show the tooltip near (x, y) (page coordinates) with rows [{key colour?, value, label}] or nodes. */
export function showTip(x, y, content) {
  tipEl = tipEl || document.getElementById('tooltip');
  if (!tipEl) return;
  clear(tipEl);
  if (Array.isArray(content)) {
    for (const row of content) {
      if (row instanceof Node) {
        tipEl.append(row);
        continue;
      }
      if (row.title) {
        tipEl.append(h('div', { class: 'tip-title' }, row.title));
        continue;
      }
      tipEl.append(h('div', { class: 'tip-row' },
        row.color ? h('span', { class: 'tip-key', style: { background: row.color } }) : h('span', { class: 'tip-key none' }),
        h('strong', {}, row.value),
        h('span', { class: 'tip-label' }, row.label || '')));
    }
  } else {
    tipEl.append(content);
  }
  tipEl.hidden = false;
  const r = tipEl.getBoundingClientRect();
  let left = x + 14;
  let top = y + 14;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
  tipEl.style.left = `${Math.max(4, left)}px`;
  tipEl.style.top = `${Math.max(4, top)}px`;
}

export function hideTip() {
  tipEl = tipEl || document.getElementById('tooltip');
  if (tipEl) tipEl.hidden = true;
}

/** A small legend: [{label, color, kind: 'line'|'dash'|'box'|'dot'}]. */
export function legend(items) {
  return h('div', { class: 'legend' }, items.map((it) => h('span', { class: 'legend-item' },
    h('span', { class: `legend-key ${it.kind || 'line'}`, style: { '--c': it.color } }), it.label)));
}

/** Compact one-line text for an event's data (untrusted: always used as text). */
export function summarize(data, max = 160) {
  if (data == null || typeof data !== 'object') return data == null ? '' : String(data);
  const parts = [];
  for (const [k, v] of Object.entries(data)) {
    if (k === 'type' || k === 'personal') continue;
    let s;
    if (v == null) s = 'null';
    else if (typeof v === 'number') s = Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) >= 100 ? 1 : 3);
    else if (typeof v === 'object') s = Array.isArray(v) && v.length <= 4 && v.every((x) => typeof x !== 'object')
      ? `[${v.map((x) => (typeof x === 'number' && !Number.isInteger(x) ? x.toFixed(1) : x)).join(', ')}]`
      : (Array.isArray(v) ? `[${v.length}]` : '{…}');
    else s = String(v);
    parts.push(`${k}=${s}`);
  }
  const text = parts.join('  ');
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
