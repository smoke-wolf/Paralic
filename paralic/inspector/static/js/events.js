// Events log: every command from the page, reply, pushed result and gesture
// event. Filter by kind and type or search the text; click a row to go there.

import { api } from './api.js';
import { clock, h, summarize } from './util.js';

const KINDS = ['command', 'reply', 'push', 'event'];
const MAX_ROWS = 3000;

export class EventsPanel {
  constructor(rv) {
    this.rv = rv;
    this.events = [];
    this.shown = [];
    this.kinds = new Set(KINDS);
    this.type = '';
    this.query = '';
    this.follow = true;
    this.current = -1;
    this.kindBtns = KINDS.map((k) => h('button', {
      class: `chip kind-${k} on`, type: 'button', 'aria-pressed': 'true',
      onclick: (e) => {
        if (this.kinds.has(k)) this.kinds.delete(k);
        else this.kinds.add(k);
        e.currentTarget.classList.toggle('on', this.kinds.has(k));
        e.currentTarget.setAttribute('aria-pressed', String(this.kinds.has(k)));
        this.render();
      },
    }, k));
    this.typeSel = h('select', { 'aria-label': 'Event type' });
    this.typeSel.addEventListener('change', () => {
      this.type = this.typeSel.value;
      this.render();
    });
    this.search = h('input', { type: 'search', placeholder: 'search…', 'aria-label': 'Search events' });
    this.search.addEventListener('input', () => {
      this.query = this.search.value.trim().toLowerCase();
      this.render();
    });
    const follow = h('input', { type: 'checkbox', checked: true });
    follow.addEventListener('change', () => {
      this.follow = follow.checked;
    });
    this.count = h('span', { class: 'ev-count' });
    this.tbody = h('tbody');
    this.scroller = h('div', { class: 'ev-scroll' }, h('table', { class: 'ev-table' },
      h('thead', {}, h('tr', {}, h('th', {}, 'time'), h('th', {}, 'kind'), h('th', {}, 'type'), h('th', {}, 'details'))),
      this.tbody));
    this.el = h('section', { class: 'panel events', id: 'sec-events' },
      h('header', { class: 'panel-head' }, h('h2', {}, 'Events'),
        h('div', { class: 'toggles' }, this.kindBtns, this.typeSel, this.search,
          h('label', { class: 'toggle' }, follow, 'Follow'), this.count)),
      this.scroller);
  }

  async load() {
    try {
      this.events = (await api.events(this.rv.id)).events;
    } catch (err) {
      this.count.textContent = `could not load events: ${err.message}`;
      return [];
    }
    this.times = Float64Array.from(this.events.map((e) => (typeof e.t === 'number' ? e.t : -Infinity)));
    const types = new Map();
    for (const e of this.events) types.set(e.type, (types.get(e.type) || 0) + 1);
    this.typeSel.replaceChildren(h('option', { value: '' }, `all types (${types.size})`),
      ...[...types.entries()].sort((a, b) => String(a[0]).localeCompare(String(b[0])))
        .map(([t, n]) => h('option', { value: t }, `${t} (${n})`)));
    const byKind = Object.fromEntries(KINDS.map((k) => [k, 0]));
    for (const e of this.events) byKind[e.kind] = (byKind[e.kind] || 0) + 1;
    this.kindBtns.forEach((b, k) => { b.textContent = `${KINDS[k]} ${byKind[KINDS[k]]}`; });
    this.render();
    return this.events;
  }

  render() {
    const p = this.rv.player;
    const q = this.query;
    this.shown = this.events.filter((e) => (this.kinds.has(e.kind) || !KINDS.includes(e.kind))
      && (!this.type || e.type === this.type)
      && (!q || `${e.type} ${summarize(e.data, 2000)}`.toLowerCase().includes(q)));
    const rows = this.shown.slice(0, MAX_ROWS).map((e) => {
      const tr = h('tr', { class: `ev kind-${e.kind}`, dataset: { n: String(e.n) } },
        h('td', { class: 'mono' }, typeof e.t === 'number' ? clock(e.t - p.start) : '—'),
        h('td', {}, h('span', { class: `badge kind-${e.kind}` }, e.kind)),
        h('td', { class: 'mono' }, e.type || ''),
        h('td', { class: 'ev-data' }, summarize(e.data)));
      tr.addEventListener('click', () => {
        if (typeof e.t === 'number') {
          p.pause();
          p.seek(e.t);
        }
        this._toggleDetail(tr, e);
      });
      return tr;
    });
    this.tbody.replaceChildren(...rows);
    this.count.textContent = this.shown.length > MAX_ROWS
      ? `showing ${MAX_ROWS} of ${this.shown.length} — narrow the filter` : `${this.shown.length} of ${this.events.length}`;
    this.current = -1;
    this.onTime();
  }

  _toggleDetail(tr, e) {
    const next = tr.nextElementSibling;
    if (next && next.classList.contains('ev-detail')) {
      next.remove();
      return;
    }
    tr.after(h('tr', { class: 'ev-detail' }, h('td', { colspan: 4 }, h('pre', {}, JSON.stringify(e.data, null, 1)))));
  }

  onTime() {
    if (!this.times) return;
    const t = this.rv.player.time;
    // The last shown event at or before the playhead (events are in time order).
    const shown = this.shown.length > MAX_ROWS ? this.shown.slice(0, MAX_ROWS) : this.shown;
    let lo = 0;
    let hi = shown.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((shown[mid].t ?? -Infinity) <= t) lo = mid + 1;
      else hi = mid;
    }
    const k = lo - 1;
    if (k === this.current) return;
    const rows = this.tbody.querySelectorAll('tr.ev');
    if (this.current >= 0 && rows[this.current]) rows[this.current].classList.remove('now');
    this.current = k;
    if (k >= 0 && rows[k]) {
      rows[k].classList.add('now');
      if (this.follow) {
        const box = this.scroller;
        const row = rows[k];
        const top = row.offsetTop - box.clientHeight / 2;
        box.scrollTop = Math.max(0, top);
      }
    }
  }
}
