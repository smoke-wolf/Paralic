import { esc, h } from '../dom.js';
import { icon } from '../icons.js';
import { sayHtml } from '../mode.js';
import { ARTICLES } from '../data/articles.js';

// Articles with a hand-mode version (`hands`, see data/articles.js) are
// rendered twice and CSS shows the one for the current mode (see mode.js).

/** The article as read in hand mode. */
const forHands = (a) => ({ ...a, ...(a.hands || {}) });

/** Escaped text, or both wordings when the hand-mode one differs. */
const both = (eyes, hands) => (hands === eyes ? esc(eyes) : sayHtml(eyes, hands));

function renderList(el) {
  el.append(
    h('div', { class: 'page-head' },
      h('div', {},
        h('div', { class: 'eyebrow' }, 'Read'),
        h('h1', {}, 'Articles'),
        h('p', { class: 'muted', html: sayHtml(
          'Pick an article with a double blink. To scroll, look at the arrows on the right edge of the screen — look further towards an arrow to scroll faster.',
          'Pick an article with a pinch. To scroll, pinch and move your hand down or up — or point at the arrows on the right edge of the screen.') }))),
    h('div', { class: 'grid cols-2' },
      ARTICLES.map((a) => {
        const hand = forHands(a);
        return h('a', { class: 'card article-card', href: `#/read/${a.id}` },
          h('div', { class: 'art-icon', html: icon(a.icon) }),
          h('div', {},
            h('h3', { html: both(a.title, hand.title) }),
            h('p', { html: both(a.summary, hand.summary) }),
            h('span', { class: 'pill', style: { marginTop: '10px' }, html: both(`${a.minutes} min read`, `${hand.minutes} min read`) })));
      })),
  );
}

function blockEl(kind, content, cls) {
  const attrs = cls ? { class: cls } : {};
  if (kind === 'h2') return h('h2', attrs, content);
  if (kind === 'quote') return h('blockquote', attrs, content);
  if (kind === 'list') return h('ul', attrs, content.map((item) => h('li', {}, item)));
  return h('p', attrs, content);
}

/** A block; one with a third element has a hand-mode version of its text. */
function block([kind, content, hands]) {
  if (hands === undefined) return blockEl(kind, content);
  return [blockEl(kind, content, 'eyes-only'), blockEl(kind, hands, 'hands-only')];
}

function articleEl(a, cls = '') {
  return h('article', { class: `article ${cls}` },
    h('div', { class: 'eyebrow' }, 'Read'),
    h('h1', {}, a.title),
    h('div', { class: 'meta', html: `${a.minutes} min read · ${sayHtml('Look at the arrows on the right to scroll',
      'Pinch and move your hand down or up to scroll')}` }),
    a.body.map(block));
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
    ...(a.hands ? [articleEl(a, 'eyes-only'), articleEl(forHands(a), 'hands-only')] : [articleEl(a)]),
    h('div', { class: 'article-end' },
      h('div', { class: 'btn-row' },
        h('a', { class: 'btn', href: '#/read', html: `${icon('grid')}<span>All articles</span>` }),
        h('a', { class: 'btn primary', href: `#/read/${next.id}`, html: `<span>Next: ${both(next.title, forHands(next).title)}</span>${icon('arrowRight')}` }))),
  );
}

export default {
  title: 'Read',
  render(el, [id]) {
    if (id) renderArticle(el, id);
    else renderList(el);
  },
};
