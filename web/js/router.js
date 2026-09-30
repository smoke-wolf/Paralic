// Minimal hash router: #/name/param1/param2

import { $$, h } from './dom.js';

export function parseHash(hash = location.hash) {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
  return { name: parts[0] || 'home', params: parts.slice(1) };
}

export class Router {
  constructor(el, routes, app) {
    this.el = el;
    this.routes = routes;
    this.app = app;
    this.cleanup = null;
    this.stack = [];
    this.goingBack = false;
  }

  start() {
    window.addEventListener('hashchange', () => this.render());
    this.render();
  }

  render() {
    const { name, params } = parseHash();
    const key = this.routes[name] ? name : 'home';
    const route = this.routes[key];
    if (this.cleanup) {
      try {
        this.cleanup();
      } catch (err) {
        console.error(err);
      }
      this.cleanup = null;
    }
    const current = location.hash || '#/home';
    if (this.goingBack) this.goingBack = false;
    else if (this.stack[this.stack.length - 1] !== current) this.stack.push(current);
    if (this.stack.length > 50) this.stack.shift();

    this.el.innerHTML = '';
    this.el.scrollTop = 0;
    const inner = h('div', { class: 'page-inner' });
    this.el.append(inner);
    try {
      this.cleanup = route.render(inner, params, this.app) || null;
    } catch (err) {
      console.error(err);
      inner.append(h('h1', {}, 'Something went wrong'), h('p', { class: 'muted' }, String(err)));
    }
    for (const a of $$('[data-route]')) a.classList.toggle('active', a.dataset.route === key);
    document.title = `${route.title || 'Paralic'} — Paralic`;
  }

  back() {
    if (this.stack.length >= 2) {
      this.stack.pop();
      this.goingBack = true;
      location.hash = this.stack[this.stack.length - 1];
      return;
    }
    const { name, params } = parseHash();
    this.stack = [];
    location.hash = params.length ? `#/${name}` : '#/home';
  }
}
