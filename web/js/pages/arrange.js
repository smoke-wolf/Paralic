// Arrange: put the planets in order from the Sun - a drag-and-drop game that
// is played with the eyes alone (hold one eye closed to drag, or use
// "Pick up to move" from the menu and blink twice to drop).

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { sounds } from '../sound.js';
import { PLANETS } from '../data/planets.js';

function shuffled(list) {
  const a = [...list];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  // Never start already solved.
  return a.every((p, i) => p.id === list[i].id) ? shuffled(list) : a;
}

function card(p) {
  const el = h('div', {
    class: 'card arrange-card', 'data-draggable': '', 'data-gaze': '', 'data-planet': p.id, 'data-label': p.name,
    'data-no-dwell': '',
  });
  const ball = h('div', { class: `planet ${p.id}`, 'aria-hidden': 'true' });
  if (p.ring) ball.append(h('div', { class: 'ring' }));
  el.append(ball, h('b', {}, p.name));
  return el;
}

export default {
  title: 'Arrange',
  render(el, params, app) {
    el.classList.add('space');
    const status = h('p', { class: 'muted arrange-status' });
    const slots = h('div', { class: 'arrange-slots' });
    const tray = h('div', { class: 'arrange-tray', 'data-dropzone': '', 'data-zone': 'tray', 'data-label': 'Tray' });
    const shuffleBtn = h('button', { class: 'btn', type: 'button', html: `${icon('refresh')}<span>Shuffle</span>` });
    const hint = app.state.simulated
      ? 'Mouse demo: point at a planet, hold Q (or E), move to a slot, release.'
      : 'Look at a planet, close one eye and keep it closed, look at a slot, then open your eye.';
    el.append(
      h('div', { class: 'page-head' },
        h('div', {},
          h('div', { class: 'eyebrow' }, 'Arrange'),
          h('h1', {}, 'Put the planets in order'),
          h('p', { class: 'muted' }, `Drag each planet into its place, from the Sun outwards. ${hint} `,
            'No wink? Close both eyes for a second, or hold your gaze still for the menu and choose “Pick up to move”.')),
        h('div', { class: 'btn-row' }, shuffleBtn)),
      h('div', { class: 'arrange-sun', html: `<span>☀</span> Sun` }),
      slots, status, tray);

    const slotEls = PLANETS.map((_, i) => {
      const s = h('div', { class: 'arrange-slot', 'data-dropzone': '', 'data-zone': 'slot', 'data-slot': String(i),
        'data-label': `Place ${i + 1}` },
      h('span', { class: 'slot-n' }, String(i + 1)));
      slots.append(s);
      return s;
    });

    const deal = () => {
      tray.innerHTML = '';
      slotEls.forEach((s) => {
        s.querySelector('.arrange-card')?.remove();
        s.classList.remove('right', 'wrong');
      });
      shuffled(PLANETS).forEach((p) => tray.append(card(p)));
      check();
    };

    const check = () => {
      const placed = slotEls.map((s) => s.querySelector('.arrange-card')?.dataset.planet || null);
      const filled = placed.filter(Boolean).length;
      slotEls.forEach((s, i) => {
        s.classList.toggle('right', filled === PLANETS.length && placed[i] === PLANETS[i].id);
        s.classList.toggle('wrong', filled === PLANETS.length && placed[i] !== PLANETS[i].id);
      });
      if (filled < PLANETS.length) {
        status.textContent = `${filled} of ${PLANETS.length} planets placed.`;
        return;
      }
      const right = placed.filter((id, i) => id === PLANETS[i].id).length;
      if (right === PLANETS.length) {
        status.textContent = 'Perfect! Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.';
        status.classList.add('ok');
        sounds.success();
      } else {
        status.textContent = `${right} of ${PLANETS.length} are in the right place — swap the red ones.`;
        status.classList.remove('ok');
      }
    };

    // A drop moves the card into a slot (swapping with whatever was there) or back to the tray.
    const onDrop = (e) => {
      const { item, zone } = e.detail;
      if (!item.classList.contains('arrange-card') || !el.contains(item)) return;
      const from = item.parentElement;
      if (zone.dataset.zone === 'tray') {
        tray.append(item);
      } else {
        const other = zone.querySelector('.arrange-card');
        if (other && other !== item) from.append(other);
        zone.append(item);
      }
      check();
    };
    el.addEventListener('gazedrop', onDrop);
    shuffleBtn.addEventListener('click', deal);
    deal();
  },
};
