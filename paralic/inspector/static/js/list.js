// The recordings list: who, how (eyes or hand), when, how long, how big, and
// what was measured (glasses seen, the last accuracy measured).

import { api } from './api.js';
import { C, bytes, count, duration, h, isNum, num } from './util.js';

export async function renderList(root) {
  const body = h('div', { class: 'list-body' }, h('p', { class: 'card-note' }, 'Loading…'));
  const refresh = h('button', { class: 'chip', type: 'button', onclick: () => renderList(root) }, 'Refresh');
  root.replaceChildren(h('div', { class: 'list-page' },
    h('header', { class: 'list-head' },
      h('div', { class: 'brand' }, h('span', { class: 'brand-mark', 'aria-hidden': 'true' }), 'Paralic Inspector'),
      refresh),
    body));
  let data;
  try {
    data = await api.recordings();
  } catch (err) {
    body.replaceChildren(h('p', { class: 'error' }, `Could not list the recordings: ${err.message}`));
    return;
  }
  const recs = data.recordings;
  if (!recs.length) {
    body.replaceChildren(h('div', { class: 'empty' },
      h('h2', {}, 'No recordings yet'),
      h('p', {}, 'Recordings are read from ', h('code', {}, data.root), '.'),
      h('p', {}, 'Record a session with the ● Rec button in Paralic, or start it with ', h('code', {}, 'python -m paralic --record'),
        '. To try the Inspector without one, make demo recordings with ', h('code', {}, 'python tools/make_demo_recording.py'), '.')));
    return;
  }
  const rows = recs.map((r) => {
    const acc = r.accuracy;
    const status = r.error ? 'unreadable' : recordingStatus(r);
    const tr = h('tr', { class: 'rec-row', tabindex: '0', title: r.path },
      h('td', {}, h('div', { class: 'rec-date' }, fmtDate(r.started)), h('div', { class: 'rec-id mono' }, r.id)),
      h('td', {}, r.person || '—'),
      h('td', {}, h('span', { class: `badge mode-${r.mode}` }, r.mode === 'hand' ? 'hand' : 'eyes')),
      h('td', { class: 'num' }, duration(r.duration_s)),
      h('td', { class: 'num' }, count(r.frames)),
      h('td', { class: 'num' }, r.frames ? `${count(r.video_frames)}` : '—'),
      h('td', { class: 'num' }, bytes(r.bytes)),
      h('td', {}, r.glasses === true ? 'yes' : r.glasses === false ? 'no' : h('span', { class: 'faint', title: 'these frames do not say' }, '—')),
      h('td', { class: 'num' }, acc && isNum(acc.mean_error_px) ? h('span', { class: 'acc' },
        h('span', { class: 'acc-dot', style: { background: rating(acc.mean_error_px, r.screen).color } }),
        `${num(acc.mean_error_px, 1)} px`, h('span', { class: 'faint' }, ` ${rating(acc.mean_error_px, r.screen).label}`)) : '—'),
      h('td', { class: 'num' }, count(r.models)),
      h('td', { class: 'num' }, count(r.events)),
      h('td', {}, h('span', { class: r.complete ? 'faint' : 'warn-text' }, status)));
    const open = () => { location.hash = `#/rec/${encodeURIComponent(r.id)}`; };
    tr.addEventListener('click', open);
    tr.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') open();
    });
    return tr;
  });
  body.replaceChildren(
    h('p', { class: 'card-note' }, `${recs.length} recording${recs.length === 1 ? '' : 's'} in `, h('code', {}, data.root),
      '. They contain camera images of faces: share them only knowingly.'),
    h('table', { class: 'rec-table' },
      h('thead', {}, h('tr', {}, [['started'], ['person'], ['mode'], ['duration', 1], ['frames', 1], ['video', 1], ['size', 1],
        ['glasses seen'], ['accuracy', 1], ['models', 1], ['events', 1], ['status']]
        .map(([t, numeric]) => h('th', { class: numeric ? 'num' : null }, t)))),
      h('tbody', {}, rows)));
}

/** How a recording ended (or that it is still going on). */
export function recordingStatus(r) {
  if (r.complete) return r.truncated ? 'complete · last line cut off' : 'complete';
  if (r.recorder_error) return `stopped by itself: ${r.recorder_error}`;
  if (r.live) return 'still recording';
  return 'not closed: Paralic stopped while recording';
}

function fmtDate(s) {
  if (!s) return '—';
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  return d.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/** The page's words for an accuracy (web/js/calibration.js rateAccuracy: relative to the screen diagonal). */
export function rating(px, screen) {
  const diag = Math.hypot((screen && screen.w) || 1920, (screen && screen.h) || 1080);
  const rel = px / diag;
  if (rel < 0.035) return { label: 'excellent', color: C.good };
  if (rel < 0.055) return { label: 'good', color: C.good };
  if (rel < 0.085) return { label: 'fair', color: C.warning };
  return { label: 'poor', color: C.critical };
}
