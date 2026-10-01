// Calibration: every calibration, quick adjust and hand setup in the recording.
//
// For the selected one: its dots on the screen (sized by how many frames were
// kept as samples, the validation dots with the error of their mean
// prediction, dots left out of training crossed), the fit (cross-validated
// error, candidate networks, dropped dots, one-eye networks), the validation's
// per-dot errors, and, below, the accuracy measured over the whole recording.

import { api } from './api.js';
import { C, PHASE, PHASE_NAMES, S, clear, clock, fitCanvas, h, hideTip, isNum, num, showTip, ticks, xy } from './util.js';

export class CalibrationPanel {
  constructor(rv) {
    this.rv = rv;
    this.data = null;
    this.sel = 0;
    this.sessions = h('div', { class: 'cal-sessions' });
    this.canvas = h('canvas', { class: 'cal-canvas', 'aria-label': 'Calibration dots on the screen' });
    this.details = h('div', { class: 'cal-details' });
    this.accCanvas = h('canvas', { class: 'acc-canvas', 'aria-label': 'Accuracy over time' });
    this.accLegend = h('div', { class: 'legend' });
    this.el = h('section', { class: 'panel calib', id: 'sec-calibration' },
      h('header', { class: 'panel-head' }, h('h2', {}, 'Calibration')),
      this.sessions,
      h('div', { class: 'cal-grid' },
        h('div', { class: 'cal-map' }, this.canvas, h('div', { class: 'legend' },
          (rv.info.mode === 'hand' ? ['hspan', 'hpoint', 'hpinch'] : ['cal', 'head', 'val', 'adjust']).map((k) => h('span', { class: 'legend-item' },
            h('span', { class: 'legend-key dot', style: { '--c': PHASE[k] } }), PHASE_NAMES[k])),
          h('span', { class: 'legend-item' }, h('span', { class: 'legend-key line', style: { '--c': C.text } }), 'to the mean prediction'),
          h('span', { class: 'legend-item' }, h('span', { class: 'legend-key cross', style: { '--c': C.critical } }), 'left out of training'))),
        this.details),
      h('h3', { class: 'sub-head' }, 'Accuracy over the recording'),
      h('div', { class: 'acc-wrap' }, this.accCanvas), this.accLegend);
    this._bind();
  }

  async load() {
    try {
      this.data = await api.calibrations(this.rv.id);
    } catch (err) {
      this.sessions.textContent = `could not load calibrations: ${err.message}`;
      return;
    }
    this.sel = Math.max(0, this.data.sessions.length - 1);
    const t = this.rv.player.time;
    this.data.sessions.forEach((s, k) => {
      if (s.t0 <= t) this.sel = k;
    });
    this.render();
  }

  render() {
    if (!this.data) return;
    const p = this.rv.player;
    const list = this.data.sessions;
    if (!list.length) {
      this.sessions.replaceChildren(h('p', { class: 'card-note' }, 'No calibration in this recording.'));
      this.details.replaceChildren();
      this.draw();
      return;
    }
    this.sessions.replaceChildren(...list.map((s, k) => {
      const fit = s.fit || {};
      const val = s.validation || {};
      const summary = s.kind === 'hand'
        ? `${fit.points ?? '—'} dots · pointing ${num(fit.pointing_error_px, 1)} px`
        : s.mode === 'adjust'
          ? `${count(s.dots, 'adjust')} dots · ${num(fit.error_before_px, 1)} → ${num(fit.error_after_px, 1)} px`
          : `${count(s.dots, 'cal')} dots · cross-validated ${num(fit.cv_error_px, 1)} px`;
      const measured = isNum(val.mean_error_px) ? ` · measured ${num(val.mean_error_px, 1)} px` : '';
      const failed = (fit.ok === false || (s.started && s.started.ok === false)) ? ' · failed' : '';
      return h('button', {
        class: `chip session${k === this.sel ? ' on' : ''}`, type: 'button',
        onclick: () => {
          this.sel = k;
          this.render();
          p.pause();
          p.seek(s.t0);
        },
      }, h('strong', {}, title(s)), ` ${clock(s.t0 - p.start, false)}–${clock(s.t1 - p.start, false)} · ${summary}${measured}${failed}`);
    }));
    this._details(list[this.sel]);
    this.draw();
  }

  onFrame() {
    if (!this.data) return;
    const t = this.rv.player.time;
    const k = this.data.sessions.findIndex((s) => s.t0 <= t && t <= s.t1 + 0.5);
    if (k >= 0 && k !== this.sel) {
      this.sel = k;
      this.render();
    } else {
      this.draw();
    }
  }

  draw() {
    this._map();
    this._accuracy();
  }

  _map() {
    const wrap = this.canvas.parentElement;
    const width = Math.min(wrap.clientWidth, 640);
    if (!width || !this.data) return;
    const { w: W, h: H } = this.rv.info.screen;
    const height = width * (H / W);
    const ctx = fitCanvas(this.canvas, width, height);
    const s = width / W;
    ctx.fillStyle = C.plot;
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = C.grid;
    for (let k = 1; k < 10; k++) {
      ctx.beginPath();
      ctx.moveTo(Math.round((k * width) / 10) + 0.5, 0);
      ctx.lineTo(Math.round((k * width) / 10) + 0.5, height);
      ctx.moveTo(0, Math.round((k * height) / 10) + 0.5);
      ctx.lineTo(width, Math.round((k * height) / 10) + 0.5);
      ctx.stroke();
    }
    ctx.strokeStyle = C.axis;
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
    const sess = this.data.sessions[this.sel];
    this.hits = [];
    if (!sess) return;
    const t = this.rv.player.time;
    const heads = sess.dots.filter((d) => d.kind === 'head');
    for (const d of sess.dots) {
      if (d.x == null || d.kind === 'head') continue;
      const x = d.x * s;
      const y = d.y * s;
      const r = Math.max(4, Math.min(12, Math.sqrt(d.stored) * 1.6));
      const now = d.t0 <= t && t <= d.t1 + 0.04;
      if (d.kind === 'val' && Array.isArray(d.mean)) {
        const mx = d.mean[0] * s;
        const my = d.mean[1] * s;
        ctx.strokeStyle = C.text;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(mx, my);
        ctx.stroke();
        ctx.fillStyle = C.text;
        ctx.beginPath();
        ctx.arc(mx, my, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = PHASE[d.kind] || C.muted;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = now ? C.accent : C.plot;
      ctx.lineWidth = now ? 3 : 2;
      ctx.stroke();
      if (d.dropped) {
        ctx.strokeStyle = C.critical;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(x - r - 3, y - r - 3);
        ctx.lineTo(x + r + 3, y + r + 3);
        ctx.moveTo(x + r + 3, y - r - 3);
        ctx.lineTo(x - r - 3, y + r + 3);
        ctx.stroke();
      }
      ctx.font = '10px system-ui, sans-serif';
      ctx.fillStyle = C.muted;
      const text = d.kind === 'val' && isNum(d.error) ? `${d.pt}: ${num(d.error, 0)} px` : `${d.pt ?? ''} ${d.stored}/${d.frames}`;
      // Labels stay inside the map: on the left of dots near the right edge, below dots at the top.
      const tw = ctx.measureText(text).width;
      const lx = x + r + 3 + tw > width - 2 ? x - r - 3 - tw : x + r + 3;
      const ly = y - r + 2 < 10 ? y + r + 10 : y - r + 2;
      ctx.fillText(text, lx, ly);
      this.hits.push({ x, y, r: r + 4, d });
    }
    if (heads.length) {
      const { w: W2, h: H2 } = this.rv.info.screen;
      const x = (heads[0].x ?? W2 / 2) * s;
      const y = (heads[0].y ?? H2 / 2) * s;
      const n = heads.reduce((a, d) => a + d.stored, 0);
      ctx.strokeStyle = PHASE.head;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 20, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = C.muted;
      const text = `head movements: ${heads.length} poses, ${n} frames kept`;
      const tw = ctx.measureText(text).width;
      ctx.fillText(text, Math.min(x - tw / 2, width - tw - 4), y + 34);
      this.hits.push({ x, y, r: 22, d: { kind: 'head', pt: `${heads.length} poses`, stored: n, frames: heads.reduce((a, d) => a + d.frames, 0), t0: heads[0].t0, t1: heads[heads.length - 1].t1, x: heads[0].x, y: heads[0].y } });
    }
  }

  _accuracy() {
    const wrap = this.accCanvas.parentElement;
    const width = wrap.clientWidth;
    if (!width || !this.data) return;
    const acc = this.data.accuracy || [];
    const height = 150;
    const ctx = fitCanvas(this.accCanvas, width, height);
    ctx.clearRect(0, 0, width, height);
    const p = this.rv.player;
    const left = 46;
    const right = 12;
    const top = 10;
    const bottom = height - 24;
    ctx.fillStyle = C.plot;
    ctx.fillRect(left, top, width - left - right, bottom - top);
    const series = {
      validation: { color: S[0], label: 'measured on validation dots' },
      adjust: { color: S[1], label: 'quick adjust dots, after the correction' },
      cv: { color: S[2], label: 'cross-validated at training' },
      hand: { color: S[3], label: 'hand pointing (leave-one-out)' },
    };
    const kinds = [...new Set(acc.map((a) => a.kind))];
    this.accLegend.replaceChildren(...kinds.map((k) => h('span', { class: 'legend-item' },
      h('span', { class: 'legend-key dot', style: { '--c': series[k].color } }), series[k].label)));
    if (!acc.length) {
      ctx.fillStyle = C.faint;
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('No accuracy was measured in this recording.', left + 10, top + 24);
      return;
    }
    const hi = Math.max(10, ...acc.map((a) => a.error_px || 0)) * 1.2;
    const x = (t) => left + ((t - p.start) / Math.max(p.end - p.start, 1e-6)) * (width - left - right);
    const y = (v) => bottom - (v / hi) * (bottom - top);
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of ticks(0, hi, 4)) {
      ctx.strokeStyle = C.grid;
      ctx.beginPath();
      ctx.moveTo(left, Math.round(y(v)) + 0.5);
      ctx.lineTo(width - right, Math.round(y(v)) + 0.5);
      ctx.stroke();
      ctx.fillStyle = C.faint;
      ctx.fillText(`${v} px`, left - 6, y(v));
    }
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    for (const t of ticks(0, p.end - p.start, Math.max(2, Math.floor(width / 120)))) {
      const text = clock(t, false);
      ctx.fillStyle = C.faint;
      ctx.fillText(text, Math.min(x(p.start + t) + 2, width - ctx.measureText(text).width - 2), bottom + 6);
    }
    for (const k of kinds) {
      const pts = acc.filter((a) => a.kind === k && isNum(a.error_px));
      ctx.strokeStyle = series[k].color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      pts.forEach((a, i) => (i ? ctx.lineTo(x(a.t), y(a.error_px)) : ctx.moveTo(x(a.t), y(a.error_px))));
      ctx.stroke();
      for (const a of pts) {
        ctx.fillStyle = series[k].color;
        ctx.beginPath();
        ctx.arc(x(a.t), y(a.error_px), 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = C.plot;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = C.text;
        ctx.textBaseline = 'bottom';
        ctx.fillText(`${num(a.error_px, 1)} px`, x(a.t) + 6, y(a.error_px) - 3);
      }
    }
    ctx.fillStyle = C.accent;
    ctx.fillRect(Math.round(x(p.time)) - 1, top, 2, bottom - top);
    this.accX = x;
  }

  _details(s) {
    clear(this.details);
    if (!s) return;
    const p = this.rv.player;
    const fit = s.fit || {};
    const val = s.validation || {};
    const rows = [];
    const kv = (k, v) => rows.push(h('tr', {}, h('th', {}, k), h('td', {}, v)));
    if (s.kind === 'hand') {
      kv('result', fit.ok === false ? `failed: ${fit.error || ''}` : 'fitted');
      kv('pointing dots', String(fit.points ?? '—'));
      kv('pointing error', isNum(fit.pointing_error_px) ? `${num(fit.pointing_error_px, 1)} px (leave-one-out)` : '—');
      kv('palm width', isNum(fit.span) ? `${num(fit.span, 3)} of the image` : '—');
      kv('pinch', `closes below ${num(fit.pinch_on, 2)} · opens above ${num(fit.pinch_off, 2)}${fit.pinch_tuned ? ' (personal)' : ' (defaults)'}`);
    } else if (s.mode === 'adjust') {
      kv('result', fit.ok === false ? `failed: ${fit.error || ''}` : 'affine correction fitted on every network');
      kv('dots', String(fit.n_points ?? '—'));
      kv('error before → after', `${num(fit.error_before_px, 1)} → ${num(fit.error_after_px, 1)} px`);
    } else if (s.fit) {
      kv('result', fit.ok === false ? `failed: ${fit.error || ''}` : `trained in ${num(fit.train_seconds, 2)} s`);
      kv('frames → samples', `${fit.n_frames ?? '—'} labelled frames → ${fit.n_samples ?? '—'} averaged samples`);
      kv('dots used', `${fit.n_points ?? '—'}${(fit.dropped_points || []).length ? ` · left out: ${fit.dropped_points.join(', ')}` : ' · none left out'}`);
      kv('network', `${fit.config || '—'} (weight decay ${fit.l2 ?? '—'})`);
      kv('cross-validated error', `${num(fit.cv_error_px, 1)} px · linear model ${num(fit.linear_cv_error_px, 1)} px · on its own training data ${num(fit.train_error_px, 1)} px`);
      if (fit.candidates && Object.keys(fit.candidates).length) {
        kv('candidates', Object.entries(fit.candidates).map(([k, v]) => `${k}: ${num(v, 1)} px`).join(' · '));
      }
      if (fit.eye_cv_px) {
        kv('one-eye networks', Object.entries(fit.eye_cv_px).map(([k, v]) => `${k} ${num(v, 1)} px`).join(' · ') + ` → ${fit.eye || 'both'} leads`);
      }
    }
    if (s.validation) {
      kv('validation', val.ok === false ? `failed: ${val.error || ''}`
        : `mean error ${num(val.mean_error_px, 1)} px · worst ${num(val.max_error_px, 1)} px · precision ${num(val.precision_px, 1)} px`);
      if (val.refine) kv('improving round', `round ${val.refine.round}: ${num(val.refine.before_px, 1)} → ${num(val.refine.after_px, 1)} px, kept the ${val.refine.kept} network`);
      if (val.refit && val.refit.n_points) kv('refit with the validation dots', `${val.refit.config || ''} · cross-validated ${num(val.refit.cv_error_px, 1)} px`);
    }
    const dots = s.dots.slice().sort((a, b) => a.t0 - b.t0);
    this.details.append(
      h('table', { class: 'num-table kv' }, h('tbody', {}, rows)),
      h('table', { class: 'num-table dots' },
        h('thead', {}, h('tr', {}, ['dot', 'target', 'frames', 'kept', 'error', 'spread', 'used', 'at'].map((x) => h('th', {}, x)))),
        h('tbody', {}, dots.map((d) => h('tr', {
          class: `click${d.dropped ? ' dropped' : ''}`, title: 'Go to this dot',
          onclick: () => {
            p.pause();
            p.seek(d.t0);
          },
        },
        h('th', {}, h('span', { class: 'legend-key dot', style: { '--c': PHASE[d.kind] || C.muted } }), `${d.kind} ${d.pt ?? ''}`),
        h('td', {}, d.x == null ? '—' : xy([d.x, d.y])),
        h('td', {}, String(d.frames)),
        h('td', {}, `${d.stored}${d.frames ? ` (${Math.round((100 * d.stored) / d.frames)}%)` : ''}`),
        h('td', {}, isNum(d.error) ? `${num(d.error, 1)} px` : '—'),
        h('td', {}, isNum(d.spread) ? `${num(d.spread, 1)} px` : '—'),
        h('td', {}, isNum(d.used) ? String(d.used) : '—'),
        h('td', {}, clock(d.t0 - p.start)))))));
  }

  _bind() {
    this.canvas.addEventListener('pointermove', (e) => {
      const hit = (this.hits || []).find((q) => Math.hypot(q.x - e.offsetX, q.y - e.offsetY) <= q.r);
      if (!hit) {
        hideTip();
        this.canvas.style.cursor = '';
        return;
      }
      this.canvas.style.cursor = 'pointer';
      const d = hit.d;
      const rows = [{ title: `${PHASE_NAMES[d.kind] || d.kind} ${d.pt ?? ''}` },
        { value: xy([d.x, d.y]), label: 'target (screen px)' },
        { value: `${d.stored} of ${d.frames}`, label: 'frames kept as samples' }];
      if (isNum(d.error)) rows.push({ value: `${num(d.error, 1)} px`, label: 'error of the mean prediction' });
      if (isNum(d.spread)) rows.push({ value: `${num(d.spread, 1)} px`, label: 'spread of the predictions' });
      if (Array.isArray(d.raw_mean)) rows.push({ value: xy(d.raw_mean), label: 'mean raw gaze while kept' });
      if (d.dropped) rows.push({ value: 'left out', label: 'the eyes were not on it (suspect dot)' });
      showTip(e.clientX, e.clientY, rows);
    });
    this.canvas.addEventListener('pointerleave', hideTip);
    this.canvas.addEventListener('click', (e) => {
      const hit = (this.hits || []).find((q) => Math.hypot(q.x - e.offsetX, q.y - e.offsetY) <= q.r);
      if (hit && isNum(hit.d.t0)) {
        this.rv.player.pause();
        this.rv.player.seek(hit.d.t0);
      }
    });
    this.accCanvas.addEventListener('click', (e) => {
      if (!this.accX) return;
      const p = this.rv.player;
      const w = this.accCanvas.clientWidth;
      const t = p.start + ((e.offsetX - 46) / (w - 58)) * (p.end - p.start);
      p.pause();
      p.seek(t);
    });
  }
}

function title(s) {
  if (s.kind === 'hand') return s.mode === 'point' ? 'Hand re-point' : 'Hand setup';
  return { full: 'Calibration', adjust: 'Quick adjust', refine: 'Improving round' }[s.mode] || s.mode;
}

function count(dots, kind) {
  return dots.filter((d) => d.kind === kind).length;
}
