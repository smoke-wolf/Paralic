// Config: meta.json (the configuration at the start and the summary at the
// end) and every change of settings and gestures during the recording.

import { api } from './api.js';
import { clock, h } from './util.js';

// Top-level parts of meta.json shown open.
const OPEN = new Set(['clock', 'screen', 'settings', 'video', 'summary', 'person', 'hand']);

export class ConfigPanel {
  constructor(rv) {
    this.rv = rv;
    this.tree = h('div', { class: 'json-tree' });
    this.changes = h('div', { class: 'changes' });
    this.el = h('section', { class: 'panel config', id: 'sec-config' },
      h('header', { class: 'panel-head' }, h('h2', {}, 'Config')),
      h('div', { class: 'config-grid' },
        h('div', {}, h('h3', {}, 'meta.json'), this.tree),
        h('div', {}, h('h3', {}, 'Changes during the recording'), this.changes)));
  }

  async load() {
    let data;
    try {
      data = await api.config(this.rv.id);
    } catch (err) {
      this.tree.textContent = `could not load the configuration: ${err.message}`;
      return;
    }
    this.tree.replaceChildren(...Object.entries(data.meta || {}).map(([k, v]) => node(k, v, OPEN.has(k))));
    if (!Object.keys(data.meta || {}).length) this.tree.replaceChildren(h('p', { class: 'card-note' }, 'meta.json is missing or unreadable.'));
    const p = this.rv.player;
    const rows = [];
    for (const c of data.changes) {
      const go = () => {
        if (typeof c.t === 'number') {
          p.pause();
          p.seek(c.t);
        }
      };
      if (c.type === 'effective') {
        // The reply to a settings command: shown with that command.
        const effective = h('details', { class: 'effective' }, h('summary', {}, 'settings in force after this change'),
          node('effective', c.effective, true));
        const prev = rows[rows.length - 1];
        if (prev && prev.dataset.type === 'settings') prev.append(effective);
        else {
          rows.push(h('div', { class: 'change' }, h('button', { class: 'linkish mono', type: 'button', onclick: go },
            clock(c.t - p.start)), effective));
        }
        continue;
      }
      rows.push(h('div', { class: 'change', dataset: { type: c.type } },
        h('button', { class: 'linkish mono', type: 'button', onclick: go }, clock(c.t - p.start)),
        h('span', { class: 'badge kind-command' }, c.type),
        c.changes.length ? h('span', { class: 'diffs' }, c.changes.map((d) => h('span', { class: 'diff' },
          h('span', { class: 'mono' }, d.key), ': ', h('span', { class: 'from' }, fmt(d.from)), ' → ', h('span', { class: 'to' }, fmt(d.to)))))
          : h('span', { class: 'card-note' }, 'no change')));
    }
    this.changes.replaceChildren(...(rows.length ? rows : [h('p', { class: 'card-note' }, 'No settings or gestures were changed.')]));
  }
}

function fmt(v) {
  if (v === undefined) return '(default)';
  return typeof v === 'object' ? JSON.stringify(v) : String(v);
}

/** One key of a JSON value, as text (never HTML): objects and arrays fold. */
function node(key, value, open = false) {
  if (value !== null && typeof value === 'object') {
    const entries = Array.isArray(value) ? value.map((v, k) => [k, v]) : Object.entries(value);
    const size = Array.isArray(value) ? `[${value.length}]` : `{${entries.length}}`;
    const det = h('details', { class: 'jn' }, h('summary', {}, h('span', { class: 'jk' }, String(key)), h('span', { class: 'js' }, ` ${size}`)));
    if (open) det.open = true;
    // Children are built when first opened (meta.json can hold a large personal profile).
    let built = false;
    const build = () => {
      if (built) return;
      built = true;
      det.append(...entries.slice(0, 500).map(([k, v]) => node(k, v, false)));
      if (entries.length > 500) det.append(h('div', { class: 'card-note' }, `… ${entries.length - 500} more`));
    };
    if (open) build();
    det.addEventListener('toggle', build);
    return det;
  }
  const cls = value === null ? 'jnull' : typeof value === 'number' ? 'jnum' : typeof value === 'boolean' ? 'jbool' : 'jstr';
  return h('div', { class: 'jl' }, h('span', { class: 'jk' }, String(key)), ': ',
    h('span', { class: cls }, typeof value === 'string' ? JSON.stringify(value) : String(value)));
}
