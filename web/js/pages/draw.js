// Draw: a canvas you paint on by holding one eye closed and looking around.
// It uses ordinary pointer events, which is how the eye gestures reach any
// page: a held wink is a pressed pointer at the spot you look at.

import { h, toast } from '../dom.js';
import { icon } from '../icons.js';

const COLORS = [['#5eead4', 'Teal'], ['#a78bfa', 'Violet'], ['#fbbf24', 'Gold'], ['#fb7185', 'Pink'], ['#f8fafc', 'White']];
const SIZES = [['6', 'Thin'], ['14', 'Medium'], ['28', 'Thick']];

export default {
  title: 'Draw',
  render(el, params, app) {
    let color = COLORS[0][0];
    let size = 14;
    const canvas = h('canvas', { class: 'draw-canvas', 'data-no-longpress': '', 'aria-label': 'Drawing area' });
    const ctx = canvas.getContext('2d');
    const swatches = h('div', { class: 'options' });
    const widths = h('div', { class: 'options' });
    const clear = h('button', { class: 'btn', type: 'button', html: `${icon('trash')}<span>Clear</span>` });
    const hint = app.state.simulated
      ? 'Mouse demo: hold Q (or E) and move the mouse to draw.'
      : 'Close one eye and keep it closed: the brush follows your other eye until you open it.';
    el.append(
      h('div', { class: 'page-head' },
        h('div', {},
          h('div', { class: 'eyebrow' }, 'Draw'),
          h('h1', {}, 'Paint with your eyes'),
          h('p', { class: 'muted' }, hint))),
      h('div', { class: 'draw-tools' }, swatches, widths, clear),
      canvas);

    const paint = () => {
      for (const b of swatches.children) b.classList.toggle('selected', b.dataset.value === color);
      for (const b of widths.children) b.classList.toggle('selected', b.dataset.value === String(size));
    };
    for (const [value, name] of COLORS) {
      const b = h('button', { class: 'opt swatch', type: 'button', 'data-value': value, 'aria-label': name }, name);
      b.style.setProperty('--swatch', value);
      b.addEventListener('click', () => { color = value; paint(); });
      swatches.append(b);
    }
    for (const [value, name] of SIZES) {
      const b = h('button', { class: 'opt', type: 'button', 'data-value': value }, name);
      b.addEventListener('click', () => { size = Number(value); paint(); });
      widths.append(b);
    }
    paint();

    // Keep the bitmap in step with the element's size (and the screen's pixel ratio).
    const fit = () => {
      const r = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.round(r.width * dpr));
      const hh = Math.max(1, Math.round(r.height * dpr));
      if (canvas.width === w && canvas.height === hh) return;
      const old = canvas.width > 1 ? ctx.getImageData(0, 0, canvas.width, canvas.height) : null;
      canvas.width = w;
      canvas.height = hh;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (old) ctx.putImageData(old, 0, 0);
    };
    const ro = new ResizeObserver(fit);
    ro.observe(canvas);

    let last = null;
    const local = (e) => {
      const r = canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    const stroke = (a, b) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = size;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    };
    canvas.addEventListener('pointerdown', (e) => {
      fit();
      last = local(e);
      stroke(last, { x: last.x + 0.01, y: last.y });
      canvas.dataset.strokes = String(Number(canvas.dataset.strokes || 0) + 1);
    });
    // Moves and releases may happen anywhere once the brush is down.
    const onMove = (e) => {
      if (!last || !(e.buttons & 1)) return;
      const p = local(e);
      stroke(last, p);
      last = p;
    };
    const onUp = () => { last = null; };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    clear.addEventListener('click', () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      canvas.dataset.strokes = '0';
      toast('Cleared');
    });
    // Extra item in the gaze menu (long close on the canvas, for example).
    canvas.addEventListener('gazemenu', (e) => e.detail.items.push({
      id: 'clear', label: 'Clear drawing', icon: 'trash', run: () => clear.click(),
    }));

    return () => {
      ro.disconnect();
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  },
};
