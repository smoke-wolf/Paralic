// Under the hood: what the models computed for the current frame.
//
// Eye mode: the blink and wink detectors' state, then the gaze network that
// led (or another one, by the tabs) step by step - its named inputs after
// standardising, every hidden layer of every ensemble member before and after
// tanh, how each member's output splits into the linear skip path, the hidden
// path and the bias, the members around their mean (their spread is the
// network's uncertainty), the quick adjust's correction, the one-eye networks
// and which inputs drive the gaze right now. All numbers come from the server,
// which runs the recorded model on the recorded features exactly as GazeNet.predict does.
//
// Hand mode: the pinch detector's state and thresholds, and the open-hand tests.

import {
  C, EYE, S, alpha, clamp, clear, clock, diverging, fill, fitCanvas, h, hideTip, isNum, num, showTip, signed, smart, svg, xy,
} from './util.js';

const NET_NAMES = { both: 'Both eyes', left: 'Left eye', right: 'Right eye' };

export class HoodPanel {
  constructor(rv) {
    this.rv = rv;
    this.mode = rv.info.mode;
    this.choice = 'led';
    this.detail = null;
    this.sub = h('div', { class: 'panel-sub' });
    this.tabs = h('div', { class: 'tabs', role: 'tablist' });
    this.body = h('div', { class: 'hood-body' });
    this.el = h('section', { class: 'panel hood', id: 'sec-hood' },
      h('header', { class: 'panel-head' }, h('h2', {}, 'Under the hood'), this.sub, this.mode === 'eyes' ? this.tabs : null),
      this.body);
    this._build();
    new ResizeObserver(() => this._redrawCanvases()).observe(this.el);
  }

  _build() {
    if (this.mode === 'hand') {
      this.machine = h('div', { class: 'card machine-card' });
      this.palm = h('div', { class: 'card' });
      this.measure = h('div', { class: 'card' });
      this.body.append(h('div', { class: 'hood-grid hand' }, this.machine, this.palm, this.measure));
      return;
    }
    this.blinkCard = h('div', { class: 'card machine-card' });
    this.winkCard = h('div', { class: 'card machine-card' });
    this.pipeline = h('div', { class: 'card pipeline-card' });
    this.inputsCanvas = h('canvas', { class: 'inputs-canvas' });
    this.inputsNote = h('div', { class: 'card-note' });
    this.layersCanvas = h('canvas', { class: 'layers-canvas' });
    this.layersNote = h('div', { class: 'card-note' });
    this.stretch = true;
    const stretch = h('input', { type: 'checkbox', checked: true });
    stretch.addEventListener('change', () => {
      this.stretch = stretch.checked;
      this._redrawCanvases();
    });
    this.stretchToggle = h('label', { class: 'toggle' }, stretch, 'Stretch colours to this frame');
    this.decomp = h('div', { class: 'decomp' });
    this.scatterCanvas = h('canvas', { class: 'scatter-canvas' });
    this.scatterNote = h('div', { class: 'card-note' });
    this.correction = h('div', { class: 'correction' });
    this.eyes = h('div', { class: 'eyes-table' });
    this.sens = h('div', { class: 'sens-table' });
    this.empty = h('div', { class: 'hood-empty', hidden: true });
    this.netArea = h('div', { class: 'hood-grid nets' },
      h('div', { class: 'card inputs-card' }, h('h3', {}, 'Inputs, standardised'), this.inputsNote, this.inputsCanvas),
      h('div', { class: 'card layers-card' }, h('div', { class: 'card-head' }, h('h3', {}, 'Hidden layers of each ensemble member'),
        this.stretchToggle), this.layersNote, this.layersCanvas, this.decomp),
      h('div', { class: 'card scatter-card' }, h('h3', {}, 'Members, mean and correction'), this.scatterNote,
        this.scatterCanvas, this.correction),
      h('div', { class: 'card eyes-card' }, h('h3', {}, 'The three networks'), this.eyes),
      h('div', { class: 'card sens-card' }, h('h3', {}, 'What drives the gaze right now'), this.sens));
    this.body.append(
      h('div', { class: 'hood-grid detectors' }, this.blinkCard, this.winkCard),
      this.pipeline, this.empty, this.netArea);
    this._bindCanvasTips();
  }

  onDetail(d) {
    this.detail = d;
    this.render();
  }

  render() {
    const d = this.detail;
    if (!d) return;
    if (this.mode === 'hand') {
      this._hand(d);
      return;
    }
    this._blink(d);
    this._wink(d);
    const ins = d.internals;
    const model = d.model;
    const p = this.rv.player;
    if (model) {
      this.sub.textContent = `model ${model.name} (${model.why}${model.config ? ` · ${model.config}` : ''} · ${model.members} members` +
        `${isNum(model.cv_error_px) ? ` · cross-validated ${num(model.cv_error_px)} px` : ''}) in force ` +
        `${model.t == null ? 'from the start' : `since ${clock(model.t - p.start)}`} (${placed(model.t_source)})`;
    } else {
      this.sub.textContent = 'no gaze model in force yet (before the first calibration)';
    }
    if (!ins) {
      this.pipeline.hidden = true;
      this.netArea.hidden = true;
      this.empty.hidden = false;
      this.tabs.replaceChildren();
      const msg = d.frame && d.frame.msg;
      this.empty.textContent = !model ? 'No gaze network has been trained yet at this point of the recording, so nothing ran on this frame.'
        : msg && !msg.face ? 'No face in this frame: the gaze networks did not run.'
          : 'This frame has no feature vector: the gaze networks did not run.';
      return;
    }
    this.pipeline.hidden = false;
    this.netArea.hidden = false;
    this.empty.hidden = true;
    const eye = this.choice === 'led' ? (ins.led || ins.preferred || 'both') : this.choice;
    this.eye = ins.networks[eye] ? eye : 'both';
    this._tabs(ins);
    this._pipeline(d, ins);
    this._redrawCanvases();
    this._decomposition(ins.networks[this.eye]);
    this._correction(ins.networks[this.eye]);
    this._eyesTable(ins);
    this._sensitivity(ins.networks[this.eye]);
  }

  _redrawCanvases() {
    const d = this.detail;
    if (!d || this.mode !== 'eyes' || !d.internals) return;
    const net = d.internals.networks[this.eye];
    if (!net) return;
    this._inputs(net);
    this._layers(net);
    this._scatter(d.internals, net);
  }

  // -- tabs -----------------------------------------------------------------------------------
  _tabs(ins) {
    const tab = (key, text) => {
      const b = h('button', {
        class: `tab${this.choice === key ? ' on' : ''}`, type: 'button', role: 'tab',
        onclick: () => {
          this.choice = key;
          this.render();
        },
      }, text);
      return b;
    };
    const led = ins.led || ins.preferred;
    this.tabs.replaceChildren(
      tab('led', `The one that led (${NET_NAMES[led] || '—'})`),
      ...ins.available.map((e) => tab(e, NET_NAMES[e] + (e === ins.preferred ? ' · preferred' : ''))));
  }

  // -- detectors ------------------------------------------------------------------------------
  _blink(d) {
    const b = d.blink || {};
    const state = b.deep ? 'deep' : b.closed ? 'closed' : b.closing ? 'closing' : 'open';
    const W = 560;
    const H = 150;
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'machine', role: 'img', 'aria-label': `Blink detector: ${state}` });
    const states = [
      ['open', 'OPEN', 'eyes open'],
      ['closing', 'CLOSING', 'cursor frozen'],
      ['closed', 'CLOSED', 'timing the closure'],
      ['deep', 'DEEP', 'fully shut'],
    ];
    const bw = 100;
    const gap = 26;
    states.forEach(([key, name, sub], k) => {
      const x = 8 + k * (bw + gap);
      const on = key === state;
      s.append(svg('rect', { x, y: 18, width: bw, height: 46, rx: 9, class: `st${on ? ' on' : ''}` }));
      s.append(svg('text', { x: x + bw / 2, y: 38, class: `st-name${on ? ' on' : ''}` }, name));
      s.append(svg('text', { x: x + bw / 2, y: 54, class: `st-sub${on ? ' on' : ''}` }, sub));
      if (k < states.length - 1) {
        const ax = x + bw + 3;
        s.append(svg('path', { d: `M${ax} 41 L${ax + gap - 8} 41`, class: 'arrow', 'marker-end': 'url(#ah)' }));
        s.append(svg('text', { x: ax + (gap - 6) / 2, y: 12, class: 'edge' },
          ['≥ open', '≥ closed', '≥ deep'][k]));
      }
    });
    s.append(svg('path', { d: `M${8 + 3 * (bw + gap) + bw / 2} 66 L${8 + 3 * (bw + gap) + bw / 2} 92 L${8 + bw / 2} 92 L${8 + bw / 2} 68`, class: 'arrow back', 'marker-end': 'url(#ah)' }));
    s.append(svg('text', { x: 8 + 1.5 * (bw + gap) + bw / 2, y: 106, class: 'edge wide' },
      'signal < open level: eyes reopened — a blink if it lasted 25–700 ms (two within the gap: a double blink)'));
    s.prepend(svg('defs', {}, svg('marker', { id: 'ah', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
      svg('path', { d: 'M0 0 L10 5 L0 10 z', class: 'arrowhead' }))));
    // The signal against its thresholds (the baseline is worked out from them in older recordings).
    const approx = b.baseline_recorded ? '' : ' ≈';
    const gauge = this._gauge([
      { v: b.signal, label: 'signal', color: S[2], bar: true },
      { v: b.baseline, label: `baseline${approx}`, color: C.faint },
      { v: b.open_thr, label: 'open', color: C.muted, dash: true },
      { v: b.close_thr, label: 'closed', color: C.text, dash: true },
      { v: b.deep_thr, label: 'deep', color: S[7], dash: true },
    ], 0, 1);
    const pending = this._pending(d);
    fill(this.blinkCard,
      h('h3', {}, 'Blink detector'),
      h('div', { class: 'machine-row' }, s, gauge),
      h('div', { class: 'card-note' },
        `watches ${blinkSignalName(b.watches)} · signal ${num(b.signal, 3)} · baseline${approx} ${num(b.baseline, 3)} · ` +
        `open below ${num(b.open_thr, 3)} · closed at ${num(b.close_thr, 3)}${isNum(b.deep_thr) ? ` · deep from ${num(b.deep_thr, 2)}` : ''}`),
      pending ? h('div', { class: 'card-note strong' }, pending) : null);
  }

  /** A first blink waiting for its partner (the detector's own flag when recorded), or a click just made. */
  _pending(d) {
    const recent = d.recent || [];
    const t = d.frame && d.frame.t;
    const flag = d.blink ? d.blink.pending : null;
    let last = null;
    for (const e of recent) {
      if (e.kind !== 'event' || e.t > t) continue;
      if (e.type === 'blink') last = e;
      if (e.type === 'double_blink' || e.type === 'blink_expired') last = { done: e };
    }
    if (last && last.done) {
      const age = t - last.done.t;
      if (age < 0.6) return `${last.done.type === 'double_blink' ? 'double blink: click' : 'the first blink expired'} ${num(age * 1000, 0)} ms ago`;
    }
    if (flag === 0 || (flag == null && (!last || last.done))) return null;
    const gap = (this.rv.info.meta.effective || {}).double_gap_ms;
    const since = last && !last.done ? `first blink ${num((t - last.t) * 1000, 0)} ms ago` : 'a first blink';
    return `${since} — waiting for a second${isNum(gap) ? ` (within ${num(gap, 0)} ms)` : ''}`;
  }

  _wink(d) {
    const w = d.wink || {};
    const state = w.state || 'idle';
    const eye = w.eye;
    const W = 470;
    const H = 150;
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'machine', role: 'img', 'aria-label': `Wink detector: ${state}` });
    const states = [
      ['idle', 'IDLE', 'both eyes open'],
      ['candidate', 'CANDIDATE', eye && state === 'candidate' ? `${eye} eye closing` : 'one eye closing'],
      ['active', 'PRESSED', eye && state === 'active' ? `${eye} eye held` : 'held: press / drag'],
    ];
    const bw = 118;
    const gap = 34;
    states.forEach(([key, name, sub], k) => {
      const x = 8 + k * (bw + gap);
      const on = key === state;
      s.append(svg('rect', { x, y: 18, width: bw, height: 46, rx: 9, class: `st${on ? ' on' : ''}` }));
      s.append(svg('text', { x: x + bw / 2, y: 38, class: `st-name${on ? ' on' : ''}` }, name));
      s.append(svg('text', { x: x + bw / 2, y: 54, class: `st-sub${on ? ' on' : ''}` }, sub));
      if (k < states.length - 1) {
        const ax = x + bw + 3;
        s.append(svg('path', { d: `M${ax} 41 L${ax + gap - 8} 41`, class: 'arrow', 'marker-end': 'url(#ah2)' }));
        s.append(svg('text', { x: ax + (gap - 6) / 2, y: 12, class: 'edge' },
          ['2 frames', isNum(w.hold_ms) ? `≥ ${num(w.hold_ms, 0)} ms` : '≥ hold time'][k]));
      }
    });
    s.append(svg('path', { d: `M${8 + 2 * (bw + gap) + bw / 2} 66 L${8 + 2 * (bw + gap) + bw / 2} 92 L${8 + bw / 2} 92 L${8 + bw / 2} 68`, class: 'arrow back', 'marker-end': 'url(#ah2)' }));
    s.append(svg('text', { x: 8 + (bw + gap) + bw / 2, y: 106, class: 'edge wide' },
      'eye opens: a short wink, or the end of a press (drop) — both eyes closing: not a wink'));
    s.prepend(svg('defs', {}, svg('marker', { id: 'ah2', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
      svg('path', { d: 'M0 0 L10 5 L0 10 z', class: 'arrowhead' }))));
    // Each eye's closure against its own closed level (when the recording has the detector's levels).
    const level = (arr, k) => (Array.isArray(arr) ? arr[k] : null);
    const gauge = this._gauge([
      { v: w.cl, label: 'left', color: EYE.left, bar: true },
      { v: w.cr, label: 'right', color: EYE.right, bar: true, second: true },
      { v: level(w.close_thr, 0), label: 'left closed', color: C.text, dash: true, side: 'left' },
      { v: level(w.close_thr, 1), label: 'right closed', color: C.text, dash: true, side: 'right' },
    ], 0, 1);
    const eyeNote = (k, name) => (Array.isArray(w.close_thr)
      ? `${name} ${num(k ? w.cr : w.cl, 3)} (baseline ${num(level(w.base, k), 2)}, closed at ${num(level(w.close_thr, k), 2)}, ` +
        `open below ${num(level(w.open_thr, k), 2)}${isNum(level(w.asym, k)) ? `, must rise ${num(w.asym[k], 2)} more than the other` : ''})`
      : `${name} ${num(k ? w.cr : w.cl, 3)}`);
    fill(this.winkCard,
      h('h3', {}, 'Wink detector'),
      h('div', { class: 'machine-row' }, s, gauge),
      h('div', { class: 'card-note' }, `each eye's closure: ${eyeNote(0, 'left')} · ${eyeNote(1, 'right')}`),
      eye ? h('div', { class: 'card-note strong' },
        `${eye} eye ${state === 'active' ? 'held: a press' : 'closing'} — the ${eye === 'left' ? 'right' : 'left'} eye's network moves the cursor`) : null);
  }

  /** A vertical gauge: bars for values, ticks for levels (across one of two bars with ``side``). */
  _gauge(items, lo, hi) {
    const W = 150;
    const H = 132;
    const top = 8;
    const bottom = H - 10;
    const y = (v) => bottom - ((clamp(v, lo, hi) - lo) / (hi - lo)) * (bottom - top);
    const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'gauge', role: 'img' });
    g.append(svg('rect', { x: 10, y: top, width: 30, height: bottom - top, class: 'gauge-track' }));
    for (const it of items) {
      if (!isNum(it.v)) continue;
      if (it.bar) {
        const x = it.second ? 26 : 10;
        const w = items.filter((i) => i.bar).length > 1 ? 14 : 30;
        g.append(svg('rect', { x, y: y(it.v), width: w, height: bottom - y(it.v), fill: it.color, rx: 2 }));
      } else {
        const [x1, x2] = it.side === 'left' ? [6, 25] : it.side === 'right' ? [25, 44] : [4, 46];
        g.append(svg('line', { x1, x2, y1: y(it.v), y2: y(it.v), stroke: it.color, 'stroke-width': 1.5, 'stroke-dasharray': it.dash ? '4 3' : null }));
      }
    }
    // Labels on the right, spread out so they do not overlap.
    const labels = items.filter((it) => isNum(it.v)).map((it) => ({ ...it, y: y(it.v) })).sort((a, b) => a.y - b.y);
    for (let k = 1; k < labels.length; k++) labels[k].y = Math.max(labels[k].y, labels[k - 1].y + 12);
    for (const it of labels) {
      g.append(svg('line', { x1: 47, x2: 54, y1: y(it.v), y2: it.y, stroke: C.faint }));
      g.append(svg('text', { x: 57, y: it.y + 4, class: 'gauge-label' }, `${it.label} ${num(it.v, 2)}`));
    }
    g.append(svg('text', { x: 6, y: H - 1, class: 'gauge-label faint' }, String(lo)));
    g.append(svg('text', { x: 6, y: top + 3, class: 'gauge-label faint' }, String(hi)));
    return g;
  }

  // -- pipeline -------------------------------------------------------------------------------
  _pipeline(d, ins) {
    const net = ins.networks[this.eye];
    const msg = (d.frame && d.frame.msg) || {};
    const clipped = net.clipped.filter(Boolean).length;
    const m = net.members.length;
    const hid = net.hidden.length ? `${net.inputs.length}→${net.hidden[0]}→${net.hidden[1]}→2` : '';
    const corr = net.correction;
    const shift = corr.identity ? null : Math.hypot(net.final[0] - net.ensemble_px[0], net.final[1] - net.ensemble_px[1]);
    const check = ins.check;
    const steps = [
      ['Inputs', `${net.inputs.length} features`, this.eye === 'both' ? 'both eyes + head' : `${this.eye} eye + head`],
      ['Standardise', 'z = (x − mean) / std', clipped ? `${clipped} clipped at ±6` : 'none clipped'],
      ['Ensemble', `${m} member${m === 1 ? '' : 's'} ${hid}`, net.members.some((x) => x.linear) ? 'ridge (linear) model' : 'tanh layers + linear skip'],
      ['Mean', xy(net.ensemble_px, 1), `spread ${num(net.spread_px, 2)} px`],
      ['Correction', corr.identity ? 'none' : `moves ${num(shift, 1)} px`, corr.identity ? 'no quick adjust yet' : 'from a quick adjust'],
      ['Network output', xy(net.final, 1), NET_NAMES[this.eye]],
    ];
    if (this.eye === ins.led && check) {
      // The recorded gaze is rounded to 0.1 px and the recorded features to 6 decimals.
      const ok = check.distance < 0.15;
      const shift = { wink: 'wink', glare: 'glare', switch: 'switch' }[check.shifted];
      const sub = shift ? `${shift} offset ${xy(check.delta, 1)}` : ok ? 'matches (rounded)' : `differs by ${num(check.distance, 2)} px`;
      steps.push(['Recorded raw', xy(check.recorded, 1), sub, ok || shift ? 'good' : 'bad']);
    }
    steps.push(['Cursor', xy(msg.gaze, 1), msg.frozen ? 'frozen through the blink' : 'after One Euro smoothing']);
    clear(this.pipeline).append(h('h3', {}, 'From features to the cursor'),
      h('div', { class: 'pipeline' }, steps.map(([title, main, sub, status], k) => [
        k ? h('span', { class: 'pipe-arrow', 'aria-hidden': 'true' }, '→') : null,
        h('div', { class: `pipe-step${status ? ` ${status}` : ''}` },
          h('div', { class: 'pipe-title' }, title), h('div', { class: 'pipe-main' }, main), h('div', { class: 'pipe-sub' }, sub)),
      ])));
  }

  // -- inputs bar chart -----------------------------------------------------------------------
  _inputs(net) {
    const width = this.inputsCanvas.parentElement.clientWidth - 24;
    if (width <= 0) return;
    const rowH = 17;
    const n = net.inputs.length;
    const height = n * rowH + 26;
    const ctx = fitCanvas(this.inputsCanvas, width, height);
    ctx.clearRect(0, 0, width, height);
    const nameW = 112;
    const rawW = 70;
    const x0 = nameW + rawW + 10;
    const x1 = width - 56;
    const mid = (x0 + x1) / 2;
    const scale = (x1 - x0) / 2 / 6;
    // Grid: 0, ±1, ±3 and the clip at ±6.
    for (const v of [-6, -3, -1, 0, 1, 3, 6]) {
      const x = Math.round(mid + v * scale) + 0.5;
      ctx.strokeStyle = v === 0 ? C.axis : Math.abs(v) === 6 ? alpha(C.critical, 0.5) : C.grid;
      ctx.beginPath();
      ctx.moveTo(x, 4);
      ctx.lineTo(x, n * rowH + 4);
      ctx.stroke();
      ctx.fillStyle = C.faint;
      ctx.font = '10px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(v === 6 || v === -6 ? `${v > 0 ? '+' : '−'}6 clip` : signed(v, 0), x, n * rowH + 18);
    }
    ctx.textBaseline = 'middle';
    net.inputs.forEach((col, k) => {
      const y = 4 + k * rowH;
      const z = net.z[k];
      ctx.font = '11px ui-monospace, Menlo, Consolas, monospace';
      ctx.textAlign = 'right';
      ctx.fillStyle = C.text;
      ctx.fillText(net.names[k], nameW, y + rowH / 2);
      ctx.fillStyle = C.faint;
      ctx.fillText(fmtRaw(net.raw[k]), nameW + rawW, y + rowH / 2);
      const w = z * scale;
      ctx.fillStyle = S[0];
      roundBar(ctx, w >= 0 ? mid : mid + w, y + 3, Math.abs(w), rowH - 6, w >= 0);
      if (net.clipped[k]) {
        ctx.fillStyle = C.critical;
        ctx.fillRect(z > 0 ? x1 - 3 : x0, y + 2, 3, rowH - 4);
      }
      ctx.textAlign = 'left';
      ctx.fillStyle = C.muted;
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillText(`${signed(net.z_raw[k], 2)}${net.clipped[k] ? ' clipped' : ''}`, x1 + 6, y + rowH / 2);
    });
    this.inputsGeom = { rowH, n, top: 4 };
    this.inputsNote.textContent = 'Each feature as the network sees it: its distance from the calibration mean in standard ' +
      'deviations (raw value on the left). Beyond ±6 it is clipped.';
  }

  // -- hidden layer heatmaps ------------------------------------------------------------------
  _layers(net) {
    const width = this.layersCanvas.parentElement.clientWidth - 24;
    if (width <= 0) return;
    const labelW = 118;
    const units = Math.max(...net.members.map((m) => m.act1.length), 1);
    const cell = clamp(Math.floor((width - labelW - 8) / units), 6, 20);
    const rowH = cell + 3;
    const blockH = 4 * rowH + 24;
    const height = net.members.length * blockH + 34;
    const ctx = fitCanvas(this.layersCanvas, width, height);
    ctx.clearRect(0, 0, width, height);
    const cells = [];
    // Colour scales: fixed (tanh saturates beyond ±2), or stretched to this frame's largest value
    // of each row kind (the same for every member, so members compare).
    const span = (key) => Math.max(1e-6, ...net.members.flatMap((m) => m[key].map(Math.abs)));
    const scales = this.stretch
      ? { pre1: span('pre1'), act1: span('act1'), pre2: span('pre2'), act2: span('act2') }
      : { pre1: 2, act1: 1, pre2: 2, act2: 1 };
    const wHidden = net.members.some((m) => Math.max(...m.hidden_px.map(Math.abs)) < 0.01);
    net.members.forEach((m, k) => {
      const y0 = k * blockH;
      ctx.font = '600 12px system-ui, sans-serif';
      ctx.fillStyle = C.text;
      ctx.textBaseline = 'middle';
      ctx.fillText(`member ${k + 1}${m.linear ? ' — ridge model, hidden layers unused' : ''}`, 0, y0 + 9);
      const rows = [
        ['layer 1, before tanh', m.pre1, 'pre1', 1],
        ['layer 1, after tanh', m.act1, 'act1', 1],
        ['layer 2, before tanh', m.pre2, 'pre2', 2],
        ['layer 2, after tanh', m.act2, 'act2', 2],
      ];
      rows.forEach(([name, vals, key, layer], r) => {
        const y = y0 + 20 + r * rowH;
        const div = scales[key];
        ctx.font = '11px system-ui, sans-serif';
        ctx.fillStyle = C.muted;
        ctx.fillText(name, 0, y + cell / 2);
        vals.forEach((v, u) => {
          const x = labelW + u * cell;
          ctx.fillStyle = diverging(v / div);
          ctx.globalAlpha = m.linear ? 0.35 : 1;
          ctx.fillRect(x, y, cell - 2, cell);
          ctx.globalAlpha = 1;
          if (key.startsWith('act') && Math.abs(v) >= 0.95) {
            ctx.fillStyle = C.text;
            ctx.beginPath();
            ctx.arc(x + (cell - 2) / 2, y + cell / 2, Math.max(1.5, cell / 7), 0, Math.PI * 2);
            ctx.fill();
          }
          cells.push({ x, y, w: cell - 2, h: cell, member: k, layer, unit: u, pre: layer === 1 ? m.pre1[u] : m.pre2[u],
            act: layer === 1 ? m.act1[u] : m.act2[u], push: layer === 2 && m.hidden_by_unit_px ? m.hidden_by_unit_px[u] : null,
            preScale: scales[`pre${layer}`], actScale: scales[`act${layer}`] });
        });
      });
    });
    // Colour scale.
    const y = net.members.length * blockH + 6;
    const gw = Math.min(260, width - labelW - 60);
    for (let i = 0; i < gw; i++) {
      ctx.fillStyle = diverging((i / (gw - 1)) * 2 - 1);
      ctx.fillRect(labelW + i, y, 1, 10);
    }
    ctx.fillStyle = C.faint;
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('−', labelW, y + 20);
    ctx.textAlign = 'right';
    ctx.fillText('+', labelW + gw, y + 20);
    ctx.textAlign = 'center';
    ctx.fillText('0', labelW + gw / 2, y + 20);
    ctx.textAlign = 'left';
    ctx.fillStyle = C.muted;
    const sc = (v) => `±${smart(v)}`;
    ctx.fillText(this.stretch
      ? `full colour = layer 1 ${sc(scales.pre1)} before, ${sc(scales.act1)} after tanh · layer 2 ${sc(scales.pre2)} / ${sc(scales.act2)}`
      : 'full colour = ±2 before tanh (it saturates there), ±1 after', labelW + gw + 10, y + 6);
    ctx.fillText('white dot: a saturated unit (|after tanh| ≥ 0.95)', labelW + gw + 10, y + 20);
    this.layerCells = cells;
    this.layersNote.textContent = `${net.members.length} networks trained from different starting weights, averaged. ` +
      'Each square is one hidden unit; hover for its numbers and how far it pushes the output.' +
      (wHidden ? ' Here the hidden path adds almost nothing: the linear skip path does nearly all the work (see the table).' : '');
  }

  _decomposition(net) {
    const rows = net.members.map((m, k) => [`member ${k + 1}`, m.skip_px, m.hidden_px, m.bias_px, m.out_px]);
    const fmt = (v) => (Array.isArray(v) ? `${smart(v[0])}, ${smart(v[1])}` : '—');
    clear(this.decomp).append(h('table', { class: 'num-table' },
      h('thead', {}, h('tr', {}, h('th', {}, 'output, px (x, y)'), h('th', {}, 'linear skip'), h('th', {}, '+ hidden path'),
        h('th', {}, '+ bias'), h('th', {}, '= member'))),
      h('tbody', {}, rows.map((r) => h('tr', {}, r.map((v, k) => h(k ? 'td' : 'th', {}, k ? fmt(v) : v)))),
        h('tr', { class: 'total' }, h('th', {}, 'mean of the members'), h('td', { colspan: 3 }, ''), h('td', {}, fmt(net.ensemble_px))))));
  }

  // -- members scatter --------------------------------------------------------------------------
  _scatter(ins, net) {
    const width = Math.min(this.scatterCanvas.parentElement.clientWidth - 24, 420);
    if (width <= 0) return;
    const height = Math.min(width, 320);
    const ctx = fitCanvas(this.scatterCanvas, width, height);
    ctx.clearRect(0, 0, width, height);
    const c = net.ensemble_px;
    const pts = net.members.map((m, k) => ({ p: m.out_px, color: S[k] || C.muted, label: `member ${k + 1}`, kind: 'member' }));
    pts.push({ p: net.final, color: C.text, label: 'after correction', kind: 'final' });
    for (const [eye, other] of Object.entries(ins.networks)) {
      if (eye !== this.eye) pts.push({ p: other.final, color: C.text, label: `${NET_NAMES[eye]} network`, kind: 'other', tag: eye[0].toUpperCase() });
    }
    // Scaled to the members and the corrected point: the spread is the point of this view. Other
    // networks further away are pinned to the edge, with their distance.
    let r = Math.max(net.spread_px * 2.5, 0.05);
    for (const q of pts) {
      if (q.kind !== 'other') r = Math.max(r, Math.abs(q.p[0] - c[0]) * 1.2, Math.abs(q.p[1] - c[1]) * 1.2);
    }
    r = niceCeil(r);
    const pad = 30;
    const sc = (Math.min(width, height) / 2 - pad) / r;
    const cx = width / 2;
    const cy = height / 2;
    const edge = Math.min(width, height) / 2 - 12;
    const P = (q) => {
      let dx = (q[0] - c[0]) * sc;
      let dy = (q[1] - c[1]) * sc;
      const far = Math.max(Math.abs(dx), Math.abs(dy));
      if (far > edge) {
        dx *= edge / far;
        dy *= edge / far;
      }
      return [cx + dx, cy + dy, far > edge];
    };
    ctx.fillStyle = C.plot;
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = C.grid;
    for (const v of [-r, -r / 2, r / 2, r]) {
      ctx.beginPath();
      ctx.moveTo(cx + v * sc, 0);
      ctx.lineTo(cx + v * sc, height);
      ctx.moveTo(0, cy + v * sc);
      ctx.lineTo(width, cy + v * sc);
      ctx.stroke();
    }
    ctx.strokeStyle = C.axis;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, height);
    ctx.moveTo(0, cy);
    ctx.lineTo(width, cy);
    ctx.stroke();
    ctx.fillStyle = C.faint;
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillText(`+${smart(r)} px`, cx + r * sc - 34, cy - 4);
    ctx.fillText(`+${smart(r)} px`, cx + 4, cy + r * sc - 4);
    // The spread (root-mean-square distance of the members from their mean).
    if (net.spread_px > 0) {
      ctx.strokeStyle = 'rgba(238, 242, 255, 0.35)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(cx, cy, net.spread_px * sc, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // The correction's move.
    const [fx, fy] = P(net.final);
    if (!net.correction.identity) {
      ctx.strokeStyle = C.muted;
      ctx.lineWidth = 1.5;
      arrow(ctx, cx, cy, fx, fy);
    }
    ctx.strokeStyle = C.text;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.stroke();
    this.scatterPts = [{ x: cx, y: cy, label: 'mean of the members', p: c }];
    for (const q of pts) {
      const [x, y, pinned] = P(q.p);
      if (pinned) {
        // Off the chart: an arrowhead on the edge pointing at it, with how far it is.
        const a = Math.atan2(y - cy, x - cx);
        ctx.fillStyle = C.text;
        ctx.beginPath();
        ctx.moveTo(x + 7 * Math.cos(a), y + 7 * Math.sin(a));
        ctx.lineTo(x + 6 * Math.cos(a + 2.4), y + 6 * Math.sin(a + 2.4));
        ctx.lineTo(x + 6 * Math.cos(a - 2.4), y + 6 * Math.sin(a - 2.4));
        ctx.fill();
        const d = Math.hypot(q.p[0] - c[0], q.p[1] - c[1]);
        const text = `${q.tag} ${smart(d)} px`;
        const tw = ctx.measureText(text).width;
        // Beside the arrowhead, on the chart's side of it.
        let tx = x - tw / 2;
        let ty = Math.sin(a) < 0 ? y + 18 : y - 10;
        if (Math.abs(Math.cos(a)) >= Math.abs(Math.sin(a))) {
          tx = Math.cos(a) < 0 ? x + 10 : x - 10 - tw;
          ty = y + 4;
        }
        ctx.fillText(text, clamp(tx, 4, width - tw - 4), clamp(ty, 12, height - 4));
        this.scatterPts.push({ x, y, label: `${q.label} (off the chart)`, p: q.p });
        continue;
      }
      if (q.kind === 'member') {
        ctx.fillStyle = q.color;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = C.plot;
        ctx.lineWidth = 2;
        ctx.stroke();
      } else if (q.kind === 'final') {
        ctx.fillStyle = C.text;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.strokeStyle = C.text;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x - 4.5, y - 4.5, 9, 9);
        ctx.fillStyle = C.text;
        ctx.fillText(q.tag, x + 7, y + 4);
      }
      this.scatterPts.push({ x, y, label: q.label, p: q.p, color: q.kind === 'member' ? q.color : null });
    }
    this.scatterNote.textContent = `Centred on the mean of the members (x right, y down). They spread ${smart(net.spread_px)} px ` +
      '(RMS): how unsure the network is. White dot: after the correction. Squares or edge arrows: the other networks for this frame.';
  }

  _correction(net) {
    const c = net.correction;
    const A = c.A;
    const b = c.b;
    if (c.identity) {
      clear(this.correction).append(h('p', { class: 'card-note' },
        'No correction: no quick adjust since this network was trained (the identity map).'));
      return;
    }
    const shift = [net.final[0] - net.ensemble_px[0], net.final[1] - net.ensemble_px[1]];
    clear(this.correction).append(
      h('p', { class: 'card-note' }, 'The quick adjust fitted an affine map, applied after the network: final = A · mean + b'),
      h('div', { class: 'matrix' },
        h('span', { class: 'mlabel' }, 'A ='), h('span', { class: 'mat' }, `${num(A[0][0], 4)}  ${num(A[0][1], 4)}`, h('br'), `${num(A[1][0], 4)}  ${num(A[1][1], 4)}`),
        h('span', { class: 'mlabel' }, 'b ='), h('span', { class: 'mat' }, `${num(b[0], 1)}`, h('br'), `${num(b[1], 1)}`)),
      h('p', { class: 'card-note' }, `This frame: ${xy(net.ensemble_px, 1)} → ${xy(net.final, 1)} (moved ${signed(shift[0], 1)}, ${signed(shift[1], 1)} px).`));
  }

  _eyesTable(ins) {
    const both = ins.networks.both;
    const rows = ins.available.map((eye) => {
      const net = ins.networks[eye];
      const d = both && eye !== 'both' ? Math.hypot(net.final[0] - both.final[0], net.final[1] - both.final[1]) : null;
      const cv = ins.eye_cv_px ? ins.eye_cv_px[eye] : net.cv_error_px;
      const role = [eye === ins.led ? 'led this frame' : null, eye === ins.preferred ? 'preferred' : null].filter(Boolean).join(', ');
      return h('tr', { class: eye === this.eye ? 'sel' : '' },
        h('th', {}, h('span', { class: 'legend-key line', style: { '--c': eye === 'both' ? C.text : EYE[eye] } }), NET_NAMES[eye]),
        h('td', {}, String(net.inputs.length)),
        h('td', {}, xy(net.final, 1)),
        h('td', {}, d == null ? '—' : `${num(d, 1)} px`),
        h('td', {}, `${num(net.spread_px, 2)} px`),
        h('td', {}, isNum(cv) ? `${num(cv, 1)} px` : '—'),
        h('td', {}, role || '—'));
    });
    clear(this.eyes).append(
      h('p', { class: 'card-note' }, 'Besides the network that reads both eyes, each eye has its own (that eye\'s features and the head pose). ' +
        'During a wink the open eye\'s network moves the cursor, shifted to line up with the usual one.'),
      h('table', { class: 'num-table' },
        h('thead', {}, h('tr', {}, ['network', 'inputs', 'output (px)', 'from both-eyes', 'members spread', 'cross-validated', 'role'].map((t) => h('th', {}, t)))),
        h('tbody', {}, rows)));
  }

  _sensitivity(net) {
    const rows = net.sensitivity || [];
    const max = Math.max(1e-9, ...rows.map((r) => Math.max(Math.abs(r.dz[0]), Math.abs(r.dz[1]))));
    const bar = (v, color) => {
      const w = (Math.abs(v) / max) * 50;
      return h('span', { class: 'dbar' }, h('span', {
        class: 'dfill', style: { width: `${w}%`, left: v >= 0 ? '50%' : `${50 - w}%`, background: color },
      }));
    };
    const ablation = (r) => {
      const k = net.inputs.indexOf(r.input);
      const a = k >= 0 && net.ablation_px ? net.ablation_px[k] : null;
      return a ? `${num(Math.hypot(a[0], a[1]), 1)} px` : '—';
    };
    const linear = (r) => {
      const k = net.inputs.indexOf(r.input);
      const s = k >= 0 && net.skip_by_input_px ? net.skip_by_input_px[k] : null;
      return s ? `${signed(s[0], 0)}, ${signed(s[1], 0)}` : '—';
    };
    clear(this.sens).append(
      h('p', { class: 'card-note' }, 'Exact derivatives of the output: how far the gaze moves when an input changes by one ' +
        'standard deviation (x right, y down). "Share" is that times the input\'s current distance from the calibration mean; ' +
        '"through the skip" is its part on the linear path alone; "without it" is how far the point would move with the input at its mean.'),
      h('table', { class: 'num-table sens' },
        h('thead', {}, h('tr', {}, ['input', 'z', 'x per σ', '', 'y per σ', '', 'share (x, y)', 'through the skip', 'without it'].map((t) => h('th', {}, t)))),
        h('tbody', {}, rows.map((r) => h('tr', {},
          h('th', { class: 'mono' }, r.name),
          h('td', {}, signed(r.z, 2)),
          h('td', {}, signed(r.dz[0], 1)), h('td', { class: 'barcell' }, bar(r.dz[0], S[0])),
          h('td', {}, signed(r.dz[1], 1)), h('td', { class: 'barcell' }, bar(r.dz[1], S[1])),
          h('td', {}, `${signed(r.share[0], 0)}, ${signed(r.share[1], 0)}`),
          h('td', {}, linear(r)),
          h('td', {}, ablation(r)))))),
      h('div', { class: 'legend' },
        h('span', { class: 'legend-item' }, h('span', { class: 'legend-key box', style: { '--c': S[0] } }), 'x (px per σ)'),
        h('span', { class: 'legend-item' }, h('span', { class: 'legend-key box', style: { '--c': S[1] } }), 'y (px per σ)')));
  }

  // -- hover on the canvases --------------------------------------------------------------------
  _bindCanvasTips() {
    this.inputsCanvas.addEventListener('pointermove', (e) => {
      const g = this.inputsGeom;
      const net = this.detail && this.detail.internals && this.detail.internals.networks[this.eye];
      if (!g || !net) return;
      const k = Math.floor((e.offsetY - g.top) / g.rowH);
      if (k < 0 || k >= g.n) {
        hideTip();
        return;
      }
      showTip(e.clientX, e.clientY, [
        { title: `${net.names[k]} (feature ${net.inputs[k]})` },
        { value: fmtRaw(net.raw[k]), label: 'recorded value' },
        { value: `${fmtRaw(net.mean[k])} ± ${fmtRaw(net.std[k])}`, label: 'calibration mean ± std' },
        { value: signed(net.z_raw[k], 3), label: net.clipped[k] ? 'standard deviations (clipped to ±6)' : 'standard deviations' },
      ]);
    });
    this.inputsCanvas.addEventListener('pointerleave', hideTip);
    this.layersCanvas.addEventListener('pointermove', (e) => {
      const cell = (this.layerCells || []).find((c) => e.offsetX >= c.x && e.offsetX < c.x + c.w + 2 && e.offsetY >= c.y && e.offsetY < c.y + c.h);
      if (!cell) {
        hideTip();
        return;
      }
      const rows = [
        { title: `member ${cell.member + 1} · layer ${cell.layer} · unit ${cell.unit + 1}` },
        { color: diverging(cell.pre / cell.preScale), value: smart(cell.pre), label: 'before tanh' },
        { color: diverging(cell.act / cell.actScale), value: smart(cell.act), label: Math.abs(cell.act) >= 0.95 ? 'after tanh (saturated)' : 'after tanh' },
      ];
      if (cell.push) rows.push({ value: `${smart(cell.push[0])}, ${smart(cell.push[1])} px`, label: 'its push on the output (x, y)' });
      showTip(e.clientX, e.clientY, rows);
    });
    this.layersCanvas.addEventListener('pointerleave', hideTip);
    this.scatterCanvas.addEventListener('pointermove', (e) => {
      const near = (this.scatterPts || []).map((q) => ({ q, d: Math.hypot(q.x - e.offsetX, q.y - e.offsetY) }))
        .filter((x) => x.d <= 12).sort((a, b) => a.d - b.d);
      if (!near.length) {
        hideTip();
        return;
      }
      showTip(e.clientX, e.clientY, near.slice(0, 3).map(({ q }) => ({ color: q.color, value: xy(q.p, 2), label: q.label })));
    });
    this.scatterCanvas.addEventListener('pointerleave', hideTip);
  }

  // -- hand mode ---------------------------------------------------------------------------------
  _hand(d) {
    const msg = (d.frame && d.frame.msg) || {};
    const hand = d.hand;
    const thr = d.pinch || {};
    const state = !msg.face ? 'none' : msg.scrolling ? 'scrolling' : msg.pinching ? 'pinching' : 'open';
    this.sub.textContent = `pinch closes below ${num(thr.on, 2)} and opens above ${num(thr.off, 2)} palm widths` +
      ((this.rv.info.meta.hand || {}).pinch_tuned ? ' (personal)' : '');
    const W = 560;
    const s = svg('svg', { viewBox: `0 0 ${W} 150`, class: 'machine', role: 'img', 'aria-label': `Pinch detector: ${state}` });
    const states = [
      ['open', 'OPEN', 'fingers apart'],
      ['pinching', 'PINCHING', 'cursor held where it aimed'],
      ['scrolling', 'SCROLLING', 'pinch moved up / down'],
    ];
    const bw = 140;
    const gap = 40;
    states.forEach(([key, name, sub], k) => {
      const x = 8 + k * (bw + gap);
      const on = key === state;
      s.append(svg('rect', { x, y: 18, width: bw, height: 46, rx: 9, class: `st${on ? ' on' : ''}` }));
      s.append(svg('text', { x: x + bw / 2, y: 38, class: `st-name${on ? ' on' : ''}` }, name));
      s.append(svg('text', { x: x + bw / 2, y: 54, class: `st-sub${on ? ' on' : ''}` }, sub));
      if (k < states.length - 1) {
        const ax = x + bw + 3;
        s.append(svg('path', { d: `M${ax} 41 L${ax + gap - 8} 41`, class: 'arrow', 'marker-end': 'url(#ah3)' }));
        s.append(svg('text', { x: ax + (gap - 6) / 2, y: 12, class: 'edge' }, ['< closes', 'moved'][k]));
      }
    });
    s.append(svg('path', { d: `M${8 + 1 * (bw + gap) + bw / 2} 66 L${8 + 1 * (bw + gap) + bw / 2} 92 L${8 + bw / 2} 92 L${8 + bw / 2} 68`, class: 'arrow back', 'marker-end': 'url(#ah3)' }));
    s.append(svg('text', { x: 8 + (bw + gap) / 2 + bw / 2, y: 106, class: 'edge wide' },
      '> opens: a click if it was short and still, else it just ends'));
    s.prepend(svg('defs', {}, svg('marker', { id: 'ah3', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
      svg('path', { d: 'M0 0 L10 5 L0 10 z', class: 'arrowhead' }))));
    const hi = Math.max(1.4, (hand && hand.pinch) || 0, thr.off || 0);
    const gauge = this._gauge([
      { v: hand ? hand.pinch : null, label: 'distance', color: S[0], bar: true },
      { v: thr.on, label: 'closes', color: C.text, dash: true },
      { v: thr.off, label: 'opens', color: C.muted, dash: true },
    ], 0, Math.round(hi * 10) / 10);
    clear(this.machine).append(h('h3', {}, 'Pinch detector'), h('div', { class: 'machine-row' }, s, gauge),
      h('div', { class: 'card-note' }, state === 'none' ? 'no hand in this frame (a pinch survives a short dropout)' :
        `thumb–index distance ${num(hand && hand.pinch, 3)} palm widths`));
    if (!hand) {
      clear(this.palm).append(h('h3', {}, 'Open hand ("stop")'), h('p', { class: 'card-note' }, 'No hand landmarks in this frame.'));
      clear(this.measure).append(h('h3', {}, 'Measurements'), h('p', { class: 'card-note' }, '—'));
      return;
    }
    const t = hand.palm_tests;
    const test = (name, value, need, ok) => h('tr', { class: ok ? 'ok' : 'no' }, h('th', {}, name), h('td', {}, value), h('td', {}, need),
      h('td', { class: ok ? 'good' : 'bad' }, ok ? 'yes' : 'no'));
    clear(this.palm).append(h('h3', {}, 'Open hand ("stop") tests'),
      h('p', { class: 'card-note' }, 'All must hold, for 0.7 s and with a still hand, to pause or resume.'),
      h('table', { class: 'num-table' },
        h('thead', {}, h('tr', {}, ['test', 'value', 'needs', ''].map((x) => h('th', {}, x)))),
        h('tbody', {},
          Object.entries(hand.reach).map(([f, r]) => test(`${f} finger extended`, `${num(r, 2)}× tip/joint reach`, `> ${hand.extend_ratio}`, hand.fingers[f])),
          test('fingers spread', `${num(t.spread, 2)} palm widths`, `≥ ${t.spread_min}`, t.spread >= t.spread_min),
          test('thumb out', `${num(t.thumb_out, 2)} palm widths`, `≥ ${t.thumb_out_min}`, t.thumb_out >= t.thumb_out_min),
          test('thumb away from index', `${num(t.thumb_free, 2)} palm widths`, `≥ ${t.thumb_free_min}`, t.thumb_free >= t.thumb_free_min),
          h('tr', { class: 'total' }, h('th', {}, 'open "stop" hand'), h('td', { colspan: 2 }, ''),
            h('td', { class: hand.stop_palm ? 'good' : 'bad' }, hand.stop_palm ? 'yes' : 'no')))));
    clear(this.measure).append(h('h3', {}, 'Measurements'),
      h('table', { class: 'num-table' }, h('tbody', {},
        h('tr', {}, h('th', {}, 'palm width'), h('td', {}, `${num(hand.palm_width, 4)} of the image`)),
        h('tr', {}, h('th', {}, 'palm centre'), h('td', {}, xy(hand.palm_centre, 3))),
        h('tr', {}, h('th', {}, 'fingertip (mirrored)'), h('td', {}, msg.hand ? xy(msg.hand.tip, 3) : '—')),
        h('tr', {}, h('th', {}, 'cursor (mapped)'), h('td', {}, xy(msg.raw, 1))),
        h('tr', {}, h('th', {}, 'cursor (smoothed)'), h('td', {}, xy(msg.gaze, 1) + (msg.frozen ? ' · held' : ''))))));
  }
}

function placed(source) {
  return {
    event: 'from its model event', file: 'time from the model file', reply: 'placed by the reply that reported it',
    trained: 'placed by its training time', mtime: 'placed by the file time', start: 'in use at the start', order: 'time unknown',
  }[source] || source;
}

function blinkSignalName(s) {
  return { both: 'the more open eye', mean: 'the average of both eyes', left: 'the left eye', right: 'the right eye' }[s] || 'the more open eye';
}

function fmtRaw(v) {
  if (!isNum(v)) return '—';
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(3);
}

function roundBar(ctx, x, y, w, hgt, right) {
  if (w < 1) {
    ctx.fillRect(x, y, Math.max(w, 1), hgt);
    return;
  }
  const r = Math.min(4, w, hgt / 2);
  ctx.beginPath();
  if (right) {
    ctx.moveTo(x, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + hgt - r);
    ctx.arcTo(x + w, y + hgt, x + w - r, y + hgt, r);
    ctx.lineTo(x, y + hgt);
  } else {
    ctx.moveTo(x + w, y);
    ctx.lineTo(x + r, y);
    ctx.arcTo(x, y, x, y + r, r);
    ctx.lineTo(x, y + hgt - r);
    ctx.arcTo(x, y + hgt, x + r, y + hgt, r);
    ctx.lineTo(x + w, y + hgt);
  }
  ctx.closePath();
  ctx.fill();
}

/** The next 1, 2 or 5 times a power of ten at or above v. */
function niceCeil(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  const f = v / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

function arrow(ctx, x0, y0, x1, y1) {
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x1 - 7 * Math.cos(a - 0.4), y1 - 7 * Math.sin(a - 0.4));
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - 7 * Math.cos(a + 0.4), y1 - 7 * Math.sin(a + 0.4));
  ctx.stroke();
}
