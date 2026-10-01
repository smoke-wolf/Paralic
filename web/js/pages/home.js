import { h } from '../dom.js';
import { icon } from '../icons.js';
import { handSummary, sayHtml } from '../mode.js';

// `hands`: a tile's text in hand mode; `eyesOnly`: not shown in hand mode (see mode.js).
const TILES = [
  { href: '#/explore', icon: 'planet', title: 'Explore', text: 'Tour the eight planets of the Solar System.' },
  { href: '#/read', icon: 'book', title: 'Read', text: 'Articles you can scroll with your eyes.', hands: 'Articles you can scroll with a pinch.', cls: 'violet' },
  { href: '#/talk', icon: 'chat', title: 'Talk', text: 'Speak phrases out loud or type with your eyes.', hands: 'Speak phrases out loud or type by pointing.' },
  { href: '#/arrange', icon: 'move', title: 'Arrange', text: 'Drag the planets into order — hold one eye closed to drag.', hands: 'Put the planets in order — pinch one to pick it up, pinch again to put it down.', cls: 'warm' },
  { href: '#/draw', icon: 'brush', title: 'Draw', text: 'Paint with your gaze while one eye stays closed.', cls: 'violet', eyesOnly: true },
  { href: '#/practice', icon: 'target', title: 'Practice', text: 'Pop targets to sharpen your aim.' },
  { href: '#/lab', icon: 'flask', title: 'Personalize', text: 'What Paralic learned about your eyes, and A/B tests to tune it.', hands: 'Your hand setup: how well pointing works, and a quick re-point.', cls: 'warm' },
  { href: '#/settings', icon: 'sliders', title: 'Settings', text: 'Gestures, dwell click, blink speed, smoothing and sounds.', hands: 'Eyes or hands, your hand setup, dwell click and sounds.', cls: 'violet' },
  { href: '#/help', icon: 'help', title: 'Help', text: 'Gestures, tips and troubleshooting.' },
];

const GESTURES = [
  ['eye', 'Look', 'Move the cursor — the nearest button lights up.'],
  ['blink', 'Blink twice', 'Click whatever is highlighted.'],
  ['wink', 'Hold one eye closed', 'Press and hold: look away to drag, keep still for the menu.'],
  ['scroll', 'Look at the arrows', 'On the right edge, to scroll long pages.'],
];

const HAND_GESTURES = [
  ['point', 'Point', 'Move the cursor with your index finger — the nearest button lights up.'],
  ['pinch', 'Pinch', 'Touch thumb and index finger, then let go, to click what is highlighted.'],
  ['scroll', 'Pinch and move', 'Move your hand down or up while pinching to scroll.'],
  ['hand', 'Open hand', 'Hold it up, fingers spread, to pause or resume.'],
];

function gestureList(list, cls) {
  return h('div', { class: `gesture-list ${cls}` },
    list.map(([ic, title, text]) => h('div', { class: 'gesture', html: `${icon(ic)}<div><b>${title}</b><span>${text}</span></div>` })));
}

function calibrationSummary(app) {
  if (app.state.simulated) {
    return ['Demo mode', 'The mouse is your gaze. Press B twice quickly to double-blink; hold Q or E to keep your left or right eye closed.'];
  }
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
    const button = (label, ic, onClick) => {
      const b = h('button', { class: 'btn', type: 'button', html: `${icon(ic)}<span>${label}</span>` });
      b.addEventListener('click', onClick);
      return b;
    };

    el.append(
      h('section', { class: 'hero' },
        h('div', { class: 'hero-main' },
          h('div', { class: 'eyebrow' }, 'Welcome to Paralic'),
          h('h1', { html: `Browse with your <span>${sayHtml('eyes', 'hands')}</span>.` }),
          h('p', { html: sayHtml('Everything on this website works without your hands. Look at something to highlight it, then blink twice to open it.',
            'Everything on this website works with one hand held up to the camera — no mouse needed. Point at something to highlight it, then pinch to open it.') }),
          h('div', { class: 'btn-row' }, exploreBtn, howBtn)),
        gestureList(GESTURES, 'eyes-only'),
        gestureList(HAND_GESTURES, 'hands-only')),
      h('section', { class: 'grid cols-3' },
        TILES.map((t) => h('a', { class: `tile ${t.cls || ''} ${t.eyesOnly ? 'eyes-only' : ''}`, href: t.href },
          h('div', { class: 'tile-icon', html: icon(t.icon) }),
          h('div', {}, h('h3', {}, t.title), t.hands ? h('p', { html: sayHtml(t.text, t.hands) }) : h('p', {}, t.text))))),
    );

    const calCard = h('section', { class: 'card calib-card eyes-only' });
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
    // Hand mode: the person's hand setup instead of the eye calibration.
    const handCard = h('section', { class: 'card calib-card hands-only' });
    const renderHand = () => {
      const s = app.state.hand;
      handCard.innerHTML = '';
      handCard.append(
        h('div', {},
          h('h3', {}, s ? handSummary(s) : 'Hand not set up yet'),
          h('p', { class: 'muted', style: { margin: 0 } }, s
            ? 'If the cursor doesn’t land where you point, re-point. If pinches are missed, redo the whole setup.'
            : 'The hand setup lets your fingertip reach the whole screen and fits the pinch to your hand.')),
        h('div', { class: 'btn-row' },
          s ? button('Re-point', 'crosshair', () => app.handSetup?.('point')) : null,
          button(s ? 'Redo hand setup' : 'Start hand setup', 'refresh', () => app.handSetup?.('full'))));
    };
    renderCalib();
    renderHand();
    el.append(calCard, handCard);
    const offs = [app.on('calibrated', renderCalib), app.on('hand', renderHand), app.on('people', renderHand)];
    return () => offs.forEach((off) => off());
  },
};
