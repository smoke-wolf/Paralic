// Small DOM helpers shared by all modules.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Escape text for use inside an HTML template (names typed by people, etc.). */
export function esc(text) {
  return String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
export const lerp = (a, b, t) => a + (b - a) * t;
export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Create an element: h('button', {class: 'btn', onclick: fn}, 'Label', child)
 * Attributes starting with "on" become event listeners; `html` sets innerHTML.
 */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key === 'html') el.innerHTML = value;
    else if (key === 'style' && typeof value === 'object') Object.assign(el.style, value);
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, '');
    else el.setAttribute(key, value);
  }
  appendChildren(el, children);
  return el;
}

function appendChildren(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

let toastSeq = 0;
/** Show a short message near the top of the screen. */
export function toast(message, kind = '', ms = 3200) {
  const root = document.getElementById('toast-root');
  if (!root) return;
  const id = ++toastSeq;
  const node = h('div', { class: `toast ${kind}`, 'data-id': id }, message);
  root.append(node);
  while (root.children.length > 3) root.firstElementChild.remove();
  setTimeout(() => node.remove(), ms);
}

/** Distance from point (x, y) to a DOMRect (0 when inside). */
export function distToRect(x, y, r) {
  const dx = Math.max(r.left - x, 0, x - r.right);
  const dy = Math.max(r.top - y, 0, y - r.bottom);
  return Math.hypot(dx, dy);
}

/** Nearest point inside a rect shrunk by `inset` (fraction of its size). */
export function nearestPointIn(x, y, r, inset = 0.2) {
  const ix = (r.width * inset) / 2;
  const iy = (r.height * inset) / 2;
  return {
    x: clamp(x, r.left + ix, r.right - ix),
    y: clamp(y, r.top + iy, r.bottom - iy),
  };
}

export function isVisible(el) {
  if (!el.isConnected) return false;
  if (el.closest('[inert], [hidden]')) return false;
  const style = getComputedStyle(el);
  return style.visibility !== 'hidden' && style.display !== 'none';
}
