import { h } from '../dom.js';
import { icon } from '../icons.js';
import { ARTICLES } from '../data/articles.js';

function renderList(el) {
  el.append(
    h('div', { class: 'page-head' },
      h('div', {},
        h('div', { class: 'eyebrow' }, 'Read'),
        h('h1', {}, 'Articles'),
        h('p', { class: 'muted' }, 'Pick an article with a double blink. To scroll, look at the arrows on the right edge of the screen — look further towards an arrow to scroll faster.'))),
    h('div', { class: 'grid cols-2' },
      ARTICLES.map((a) => h('a', { class: 'card article-card', href: `#/read/${a.id}` },
        h('div', { class: 'art-icon', html: icon(a.icon) }),
        h('div', {},
          h('h3', {}, a.title),
          h('p', {}, a.summary),
          h('span', { class: 'pill', style: { marginTop: '10px' } }, `${a.minutes} min read`))))),
  );
}

function block([kind, content]) {
  if (kind === 'h2') return h('h2', {}, content);
  if (kind === 'quote') return h('blockquote', {}, content);
  if (kind === 'list') return h('ul', {}, content.map((item) => h('li', {}, item)));
  return h('p', {}, content);
}

function renderArticle(el, id) {
  const index = ARTICLES.findIndex((a) => a.id === id);
  if (index < 0) {
    el.append(h('h1', {}, 'Article not found'), h('a', { class: 'btn', href: '#/read' }, 'All articles'));
    return;
  }
  const a = ARTICLES[index];
  const next = ARTICLES[(index + 1) % ARTICLES.length];
  el.append(
    h('article', { class: 'article' },
      h('div', { class: 'eyebrow' }, 'Read'),
      h('h1', {}, a.title),
      h('div', { class: 'meta' }, `${a.minutes} min read · Look at the arrows on the right to scroll`),
      a.body.map(block)),
    h('div', { class: 'article-end' },
      h('div', { class: 'btn-row' },
        h('a', { class: 'btn', href: '#/read', html: `${icon('grid')}<span>All articles</span>` }),
        h('a', { class: 'btn primary', href: `#/read/${next.id}`, html: `<span>Next: ${next.title}</span>${icon('arrowRight')}` }))),
  );
}

export default {
  title: 'Read',
  render(el, [id]) {
    if (id) renderArticle(el, id);
    else renderList(el);
  },
};
