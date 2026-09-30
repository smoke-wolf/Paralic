import { h } from '../dom.js';
import { icon } from '../icons.js';

const TILES = [
  { href: '#/explore', icon: 'planet', title: 'Explore', text: 'Tour the eight planets of the Solar System.' },
  { href: '#/read', icon: 'book', title: 'Read', text: 'Articles you can scroll with your eyes.', cls: 'violet' },
  { href: '#/talk', icon: 'chat', title: 'Talk', text: 'Speak phrases out loud or type with your eyes.' },
  { href: '#/practice', icon: 'target', title: 'Practice', text: 'Pop targets to sharpen your aim.', cls: 'warm' },
  { href: '#/settings', icon: 'sliders', title: 'Settings', text: 'Blink speed, smoothing, sounds and calibration.', cls: 'violet' },
  { href: '#/help', icon: 'help', title: 'Help', text: 'Gestures, tips and troubleshooting.' },
];

const GESTURES = [
  ['eye', 'Look', 'Move the cursor — the nearest button lights up.'],
  ['blink', 'Blink twice', 'Click whatever is highlighted.'],
  ['scroll', 'Look at the arrows', 'On the right edge, to scroll long pages.'],
  ['pause', 'Pause', 'Rest your eyes; blink twice to resume.'],
];

function calibrationSummary(app) {
  if (app.state.simulated) return ['Demo mode', 'The mouse is your gaze. Press B twice quickly to double-blink.'];
  if (!app.state.calibrated) return ['Not calibrated yet', 'Calibrate so the cursor follows your eyes.'];
  const label = app.accuracyLabel();
  const names = { excellent: 'excellent', good: 'good', fair: 'fair', poor: 'poor' };
  return [
    label ? `Calibration accuracy: ${names[label]}` : 'Calibrated',
    'If the cursor drifts away from where you look, run a quick adjustment.',
  ];
}

export default {
  title: 'Home',
  render(el, params, app) {
    const exploreBtn = h('a', { class: 'btn primary', href: '#/explore', html: `${icon('planet')}<span>Explore the planets</span>` });
    const howBtn = h('a', { class: 'btn', href: '#/read/how-it-works', html: `${icon('brain')}<span>How it works</span>` });

    el.append(
      h('section', { class: 'hero' },
        h('div', { class: 'hero-main' },
          h('div', { class: 'eyebrow' }, 'Welcome to Paralic'),
          h('h1', { html: 'Browse with your <span>eyes</span>.' }),
          h('p', {}, 'Everything on this website works without your hands. Look at something to highlight it, then blink twice to open it.'),
          h('div', { class: 'btn-row' }, exploreBtn, howBtn)),
        h('div', { class: 'gesture-list' },
          GESTURES.map(([ic, title, text]) => h('div', { class: 'gesture', html: `${icon(ic)}<div><b>${title}</b><span>${text}</span></div>` })))),
      h('section', { class: 'grid cols-3' },
        TILES.map((t) => h('a', { class: `tile ${t.cls || ''}`, href: t.href },
          h('div', { class: 'tile-icon', html: icon(t.icon) }),
          h('div', {}, h('h3', {}, t.title), h('p', {}, t.text))))),
    );

    const calCard = h('section', { class: 'card calib-card' });
    const renderCalib = () => {
      const [title, text] = calibrationSummary(app);
      calCard.innerHTML = '';
      const info = h('div', {}, h('h3', {}, title), h('p', { class: 'muted', style: { margin: 0 } }, text));
      calCard.append(info);
      if (!app.state.simulated) {
        const quick = h('button', { class: 'btn', type: 'button', html: `${icon('crosshair')}<span>Quick adjust</span>` });
        const full = h('button', { class: 'btn', type: 'button', html: `${icon('refresh')}<span>Full calibration</span>` });
        quick.addEventListener('click', () => app.calibrate('adjust'));
        full.addEventListener('click', () => app.calibrate('full'));
        calCard.append(h('div', { class: 'btn-row' }, app.state.calibrated ? quick : null, full));
      }
    };
    renderCalib();
    el.append(calCard);
    return app.on('calibrated', renderCalib);
  },
};
