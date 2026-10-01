import { h } from '../dom.js';
import { icon } from '../icons.js';
import { sayHtml } from '../mode.js';
import { PLANETS } from '../data/planets.js';

function planetEl(p, big = false) {
  const el = h('div', { class: `planet ${p.id}${big ? ' big' : ''}`, 'aria-hidden': 'true' });
  if (p.ring) el.append(h('div', { class: 'ring' }));
  return el;
}

function renderList(el) {
  el.classList.add('space');
  el.append(
    h('div', { class: 'page-head' },
      h('div', {},
        h('div', { class: 'eyebrow' }, 'Explore'),
        h('h1', {}, 'The Solar System'),
        h('p', { class: 'muted', html: sayHtml('Look at a planet and blink twice to visit it.', 'Point at a planet and pinch to visit it.') }))),
    h('div', { class: 'grid cols-4' },
      PLANETS.map((p) => h('a', { class: 'card planet-card', href: `#/explore/${p.id}` },
        planetEl(p),
        h('h3', {}, p.name),
        h('p', {}, p.tagline)))),
  );
}

function renderDetail(el, id) {
  const index = PLANETS.findIndex((p) => p.id === id);
  if (index < 0) {
    el.append(h('h1', {}, 'Planet not found'), h('a', { class: 'btn', href: '#/explore' }, 'All planets'));
    return;
  }
  const p = PLANETS[index];
  const prev = PLANETS[(index + PLANETS.length - 1) % PLANETS.length];
  const next = PLANETS[(index + 1) % PLANETS.length];
  el.classList.add('space');
  el.append(
    h('div', { class: 'planet-detail' },
      h('div', { class: 'planet-stage' }, planetEl(p, true)),
      h('div', {},
        h('div', { class: 'eyebrow' }, `Planet ${index + 1} of ${PLANETS.length}`),
        h('h1', {}, p.name),
        h('p', { class: 'muted', style: { fontSize: '1.2rem' } }, p.tagline),
        p.text.map((t) => h('p', {}, t)),
        h('div', { class: 'facts' }, p.facts.map(([v, label]) => h('div', { class: 'fact' }, h('b', {}, v), h('span', {}, label)))),
        h('div', { class: 'note-box', style: { marginBottom: '26px' } }, h('b', {}, 'Did you know? '), p.fun))),
    h('div', { class: 'btn-row', style: { marginTop: '10px' } },
      h('a', { class: 'btn', href: `#/explore/${prev.id}`, html: `${icon('arrowLeft')}<span>${prev.name}</span>` }),
      h('a', { class: 'btn', href: '#/explore', html: `${icon('grid')}<span>All planets</span>` }),
      h('a', { class: 'btn primary', href: `#/explore/${next.id}`, html: `<span>${next.name}</span>${icon('arrowRight')}` })),
  );
}

export default {
  title: 'Explore',
  render(el, [id]) {
    if (id) renderDetail(el, id);
    else renderList(el);
  },
};
