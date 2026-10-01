import { h } from '../dom.js';
import { icon } from '../icons.js';

const CARDS = [
  {
    icon: 'eye', title: 'Moving the cursor',
    items: [
      'Look where you want to go — the cursor follows your gaze.',
      'The nearest button lights up; you don’t need to hit it exactly.',
      'The cursor holds still while you blink.',
    ],
  },
  {
    icon: 'blink', title: 'Clicking',
    items: [
      'Blink twice quickly, like a relaxed “yes, yes”.',
      'After the first blink a small “1” appears on the cursor and the button pulses.',
      'The click goes to what you looked at before the first blink.',
    ],
  },
  {
    icon: 'wink', title: 'Holding, dragging and the menu',
    items: [
      'Close one eye and keep it closed: that is like holding the mouse button down where you were looking.',
      'Keep it closed and look somewhere else to drag (try Arrange), or to draw (try Draw). Open the eye to drop.',
      'Keep looking at the same spot for a second instead: a menu opens (a right click) — Click, Pick up to move, Read aloud.',
      'Open the eye again quickly without looking away: a click.',
      'The cursor keeps following your open eye the whole time.',
    ],
  },
  {
    icon: 'blink', title: 'If winking or blinking is hard',
    items: [
      'Settings → Eye gestures → Test my winks checks each eye; an eye that can’t wink on its own is ignored.',
      'Close both eyes for about a second: can open the menu, pick up / drop, or click.',
      'Dwell click: rest your eyes on a button until the ring fills — no blinking at all.',
      '“Pick up to move” in the menu carries a thing without holding; blink twice (or dwell) to drop it.',
      'If one eye squints or droops, the tracker can follow the steadier eye (Tracking eye: Auto).',
    ],
  },
  {
    icon: 'scroll', title: 'Scrolling',
    items: [
      'Look at the Scroll up or Scroll down zone on the right edge.',
      'Look further towards the arrow to scroll faster; look away to stop.',
      'Blink twice on a zone to jump a whole page.',
    ],
  },
  {
    icon: 'pause', title: 'Pausing',
    items: [
      'Look at Pause (top right) and blink twice to rest your eyes.',
      'While paused nothing gets clicked. Blink twice to resume.',
    ],
  },
  {
    icon: 'crosshair', title: 'When the cursor drifts',
    items: [
      'Run a Quick adjust (9 dots) from Home or Settings.',
      'If it is far off, run a Full calibration.',
      'Calibrate sitting the way you will use the computer.',
    ],
  },
  {
    icon: 'keyboard', title: 'Keyboard shortcuts for helpers',
    html: '<ul><li><kbd>C</kbd> full calibration</li><li><kbd>A</kbd> quick adjust</li><li><kbd>P</kbd> pause / resume</li><li><kbd>Esc</kbd> cancel a calibration</li><li><kbd>B</kbd> blink (mouse demo mode: press twice to click, hold 1 s for a long close)</li><li><kbd>Q</kbd> / <kbd>E</kbd> hold to keep the left / right eye closed (demo mode)</li><li><kbd>Esc</kbd> drop a carried item / close the menu</li><li><kbd>F11</kbd> full screen</li></ul>',
  },
  {
    icon: 'sun', title: 'Troubleshooting',
    items: [
      'Cursor jumpy: choose more smoothing, improve lighting, sit still while calibrating.',
      'Blinks missed: raise blink sensitivity or choose the Relaxed double-blink speed.',
      'Unwanted clicks: lower blink sensitivity or choose the Fast speed.',
      'No face found: face the camera, light your face from the front, remove strong backlight.',
      'Camera blocked: allow it via the camera icon in the address bar and reload.',
    ],
  },
  {
    icon: 'camera', title: 'Privacy',
    items: [
      'Video is analysed on this computer by the Python server and never stored.',
      'Your calibration (numbers only, no images) is saved in the data folder so you don’t have to recalibrate each time.',
    ],
  },
];

export default {
  title: 'Help',
  render(el) {
    el.append(
      h('div', { class: 'page-head' },
        h('div', {},
          h('div', { class: 'eyebrow' }, 'Help'),
          h('h1', {}, 'Using Paralic with your eyes'),
          h('p', { class: 'muted' }, 'Look, blink twice, and look at the arrows on the right to scroll. Here are the details.'))),
      h('div', { class: 'help-grid' },
        CARDS.map((c) => h('section', { class: 'card' },
          h('h3', { html: `${icon(c.icon)}<span>${c.title}</span>` }),
          c.html ? h('div', { class: 'muted', html: c.html }) : h('ul', {}, c.items.map((i) => h('li', {}, i)))))),
      h('div', { class: 'btn-row', style: { marginTop: 'var(--gap)' } },
        h('a', { class: 'btn primary', href: '#/practice', html: `${icon('target')}<span>Practice clicking</span>` }),
        h('a', { class: 'btn', href: '#/read/tips', html: `${icon('sparkle')}<span>Accuracy tips</span>` })),
    );
  },
};
