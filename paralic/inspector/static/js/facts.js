// "This frame": the current frame's numbers at a glance.

import { C, EYE, h, isNum, num, xy } from './util.js';

export class FactsPanel {
  constructor(rv) {
    this.rv = rv;
    this.mode = rv.info.mode;
    this.body = h('div', { class: 'facts' });
    this.el = h('section', { class: 'panel facts-panel', id: 'sec-facts' },
      h('header', { class: 'panel-head' }, h('h2', {}, 'This frame')), this.body);
    this.detail = null;
  }

  onDetail(d) {
    this.detail = d;
    this.render();
  }

  onFrame() {
    this.render();
  }

  render() {
    const p = this.rv.player;
    const d = this.detail && this.detail.n === p.n ? this.detail : null;
    const f = (d && d.frame) || this.rv.frames.get(p.n);
    const rows = [];
    const row = (k, v, extra) => rows.push(h('div', { class: 'fact' }, h('span', { class: 'k' }, k), h('span', { class: 'v' }, v), extra || null));
    row('time', `${(p.t[p.n] - p.start).toFixed(3)} s  (session clock ${p.t[p.n].toFixed(3)})`);
    row('frame', `${p.n + 1} of ${p.length} · id ${p.ids[p.n]}`);
    if (!f) {
      row('', 'loading…');
      this.body.replaceChildren(...rows);
      return;
    }
    const m = f.msg || {};
    if (this.mode === 'eyes') {
      row('face', m.face ? `yes${isNum(m.faces) && m.faces > 1 ? ` · ${m.faces} faces in view` : ''}`
        : m.waiting ? `${m.faces > 1 ? `${m.faces} faces` : 'a face'} in view, not this person's: waiting` : 'no face');
      if ('glasses' in m) {
        row('glasses', `${m.glasses ? 'on' : 'off'}${m.glare ? ` · glare on the ${m.glare} lens` : ''}` +
          (Array.isArray(m.glare_score) ? ` · glare score L ${num(m.glare_score[0], 2)} R ${num(m.glare_score[1], 2)}` : ''));
      }
      row('raw gaze', xy(m.raw, 1) + (m.net ? ` · ${m.net === 'both' ? 'both eyes' : `${m.net} eye`} network` : ''));
      row('cursor', xy(m.gaze, 1) + (m.frozen ? ' · frozen' : ''));
      rows.push(closureBars(m, f.blink));
      const b = f.blink;
      row('blink thresholds', b ? `closed ≥ ${num(b.close_thr, 3)} · open < ${num(b.open_thr, 3)} · signal ${num(b.signal, 3)}` +
        (isNum(b.baseline) ? ` · baseline ${num(b.baseline, 3)}` : '') + (b.pending ? ' · a first blink waits for a second' : '')
        : Array.isArray(m.thr) ? `closed ≥ ${num(m.thr[0], 3)} · open < ${num(m.thr[1], 3)}` : '—');
      const w = f.wink || {};
      row('wink', w.pressed ? `${w.pressed} eye held (pressed)` : w.winking ? `${w.winking} eye closing (candidate)` : 'none');
      if (f.gesture) row('wink test', f.gesture === 'rest' ? 'asks to keep both eyes open' : `asks to close the ${f.gesture} eye`);
      row('head', Array.isArray(m.head) ? `yaw ${num(m.head[0])}° · pitch ${num(m.head[1])}° · roll ${num(m.head[2])}°` : '—');
      row('position', `${isNum(m.dist) ? `${num(m.dist)} cm from the camera` : '—'}${Array.isArray(m.pos) ? ` · x ${num(m.pos[0])} y ${num(m.pos[1])} cm` : ''}`);
    } else {
      const hand = m.hand || {};
      row('hand', (m.face ? 'in view' : 'not in view') + (isNum(m.hands) && m.hands > 1 ? ` · ${m.hands} hands: the one in control counts` : ''));
      row('fingertip', Array.isArray(hand.tip) ? `${num(hand.tip[0], 3)}, ${num(hand.tip[1], 3)} (image, mirrored)` : '—');
      row('cursor', xy(m.gaze, 1) + (m.frozen ? ' · held' : ''));
      const thr = d && d.pinch;
      row('pinch', `${num(hand.pinch, 3)} palm widths${thr ? ` · closes < ${num(thr.on, 2)} · opens > ${num(thr.off, 2)}` : ''}`);
      row('state', [m.pinching ? 'pinching' : 'open', m.scrolling ? 'scrolling' : null, hand.open ? 'open "stop" hand' : null]
        .filter(Boolean).join(' · '));
      row('palm width', isNum(hand.span) ? `${num(hand.span, 4)} of the image width` : '—');
    }
    const label = f.label;
    if (label) {
      row('label', `${label.kind} ${label.pt ?? label.i ?? ''}` + (isNum(label.x) ? ` at ${num(label.x, 0)}, ${num(label.y, 0)}` : '') +
        (isNum(label.fx) ? ` at ${num(label.fx, 2)}, ${num(label.fy, 2)} of the screen` : '') +
        (f.stored ? ' · kept as a sample' : ' · not kept') + (isNum(m.settled) ? ` · settled ${m.settled}` : ''));
    }
    row('timing', `${num(m.ms, 1)} ms to process · ${num(m.fps, 1)} fps`);
    const events = f.events || [];
    if (events.length) row('events', events.join(', '));
    if (m.error) row('error', m.error);
    this.body.replaceChildren(...rows);
  }
}

function closureBars(m, blink) {
  const bar = (v, color) => h('span', { class: 'cbar' }, h('span', { class: 'cfill', style: { width: `${Math.max(0, Math.min(1, v || 0)) * 100}%`, background: color } }));
  const close = blink && isNum(blink.close_thr) ? blink.close_thr : Array.isArray(m.thr) ? m.thr[0] : null;
  const mark = isNum(close) ? h('span', { class: 'cmark', style: { left: `${close * 100}%` } }) : null;
  return h('div', { class: 'fact' }, h('span', { class: 'k' }, 'eye closure'),
    h('span', { class: 'v closures' },
      h('span', { class: 'ceye' }, 'L', h('span', { class: 'cwrap' }, bar(m.cl, EYE.left), mark && mark.cloneNode()), num(m.cl, 2)),
      h('span', { class: 'ceye' }, 'R', h('span', { class: 'cwrap' }, bar(m.cr, EYE.right), mark), num(m.cr, 2)),
      m.closed ? h('span', { class: 'badge', style: { borderColor: C.muted } }, 'closed') : null));
}
