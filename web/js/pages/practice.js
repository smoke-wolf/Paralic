// Practice: pop targets with a double blink (a pinch in hand mode). Targets
// shrink every level.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { say, sayHtml } from '../mode.js';
import { sounds } from '../sound.js';
import { PursuitTrainer } from '../pursuit.js';

const SIZES = [240, 190, 150, 120, 96];
const TARGETS_PER_ROUND = 8;
let level = 1;

export default {
  title: 'Practice',
  render(el, params, app) {
    const stats = h('div', { class: 'stats' });
    const arena = h('div', { class: 'arena' });
    el.append(
      h('div', { class: 'hud' },
        h('div', {}, h('div', { class: 'eyebrow' }, 'Practice'), h('h2', { style: { margin: 0 } }, 'Target practice')),
        stats),
      arena);

    let round = null;
    let timer = null;

    const renderStats = () => {
      const r = round || { hits: 0, misses: 0, times: [] };
      const avg = r.times.length ? (r.times.reduce((a, b) => a + b, 0) / r.times.length / 1000).toFixed(1) : '–';
      stats.innerHTML = '';
      stats.append(
        h('span', { class: 'pill' }, `Level ${level} · ${SIZES[level - 1]} px`),
        h('span', { class: 'pill ok' }, `Hits ${r.hits}`),
        h('span', { class: 'pill warn' }, `Misses ${r.misses}`),
        h('span', { class: 'pill' }, `Avg ${avg} s`));
    };

    const center = (...children) => {
      arena.innerHTML = '';
      arena.append(h('div', { class: 'arena-center' }, h('div', {}, ...children)));
    };

    const button = (label, ic, onClick, primary = false) => {
      const b = h('button', { class: `btn ${primary ? 'primary' : ''}`, type: 'button', html: `${icon(ic)}<span>${label}</span>` });
      b.addEventListener('click', onClick);
      return b;
    };

    const intro = () => {
      round = null;
      renderStats();
      center(
        h('h1', {}, 'Pop the targets'),
        h('p', { class: 'muted', style: { maxWidth: '44ch', margin: '0 auto 28px' }, html: sayHtml(
          `Look at each target and blink twice to pop it. ${TARGETS_PER_ROUND} targets per level; they get smaller as you improve.`,
          `Point at each target and pinch to pop it. ${TARGETS_PER_ROUND} targets per level; they get smaller as you improve.`) }),
        h('div', { class: 'btn-row' },
          button(`Start level ${level}`, 'play', start, true),
          button('Follow a trail', 'target', () => pursuit.run('lissajous')),
          button('Snake chase', 'play', () => pursuit.runSnake())),
        h('p', { class: 'muted', style: { maxWidth: '48ch', margin: '18px auto 0', fontSize: '0.9em' }, html: sayHtml(
          'Smooth-pursuit modes: follow a moving dot with your eyes. They sharpen your tracking and, once you are calibrated, quietly fine-tune the model from the moments you stay locked on.',
          'Smooth-pursuit modes: follow a moving dot with your fingertip — good practice for smooth, steady pointing.') }));
    };

    const pursuit = new PursuitTrainer(app);

    const start = () => {
      round = { left: TARGETS_PER_ROUND, hits: 0, misses: 0, times: [], target: null, shownAt: 0 };
      arena.innerHTML = '';
      renderStats();
      timer = setTimeout(spawn, 500);
    };

    const spawn = () => {
      if (!round) return;
      const size = SIZES[level - 1];
      const w = arena.clientWidth;
      const hgt = arena.clientHeight;
      const pad = size / 2 + 12;
      let x;
      let y;
      const last = round.last;
      for (let tries = 0; tries < 20; tries++) {
        x = pad + Math.random() * Math.max(1, w - 2 * pad);
        y = pad + Math.random() * Math.max(1, hgt - 2 * pad);
        if (!last || Math.hypot(x - last.x, y - last.y) > Math.min(w, hgt) * 0.35) break;
      }
      round.last = { x, y };
      const t = h('button', {
        class: 'target', type: 'button', 'aria-label': 'Target', 'data-learn': 'practice',
        style: { width: `${size}px`, height: `${size}px`, left: `${x - size / 2}px`, top: `${y - size / 2}px` },
      });
      t.addEventListener('click', () => hit(t));
      arena.append(t);
      round.target = t;
      round.shownAt = performance.now();
    };

    const hit = (t) => {
      if (!round || round.target !== t) return;
      round.hits += 1;
      round.times.push(performance.now() - round.shownAt);
      round.left -= 1;
      round.target = null;
      sounds.pop();
      t.classList.add('hit');
      t.disabled = true;
      setTimeout(() => t.remove(), 320);
      renderStats();
      if (round.left > 0) timer = setTimeout(spawn, 350);
      else timer = setTimeout(finish, 450);
    };

    const finish = () => {
      const r = round;
      round = null;
      const avg = r.times.length ? (r.times.reduce((a, b) => a + b, 0) / r.times.length / 1000).toFixed(1) : '–';
      const accuracy = Math.round((100 * r.hits) / Math.max(1, r.hits + r.misses));
      sounds.success();
      const canLevelUp = level < SIZES.length;
      center(
        h('div', { class: 'eyebrow' }, `Level ${level} complete`),
        h('h1', {}, accuracy >= 80 ? 'Great aim!' : 'Nice work!'),
        h('p', { class: 'muted', style: { fontSize: '1.2rem', marginBottom: '28px' } },
          `${r.hits} targets in ${avg} s on average · ${accuracy}% of ${say('double blinks', 'pinches')} on target`),
        h('div', { class: 'btn-row' },
          canLevelUp ? button('Next level', 'arrowRight', () => { level += 1; intro(); }, true) : null,
          button('Play again', 'refresh', start),
          level > 1 ? button('Easier', 'arrowLeft', () => { level -= 1; intro(); }) : null));
      renderStats();
    };

    // A double blink that doesn't land on the target counts as a miss.
    const off = app.on('activate', ({ element, point }) => {
      if (!round || !round.target) return;
      if (element === round.target) return;
      const r = arena.getBoundingClientRect();
      const inside = point && point.x >= r.left && point.x <= r.right && point.y >= r.top && point.y <= r.bottom;
      if (inside || (element && arena.contains(element))) {
        round.misses += 1;
        renderStats();
      }
    });

    intro();
    return () => {
      off();
      clearTimeout(timer);
    };
  },
};
