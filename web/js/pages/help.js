import { esc, h } from '../dom.js';
import { icon } from '../icons.js';
import { sayHtml } from '../mode.js';

// Help for both ways of controlling Paralic (see mode.js): `mode: 'eyes'` or
// `'hands'` shows a card only in that mode, and an item given as
// [eyes, hands] has a wording for each.
const CARDS = [
  {
    mode: 'eyes', icon: 'eye', title: 'Moving the cursor',
    items: [
      'Look where you want to go — the cursor follows your gaze.',
      'The nearest button lights up; you don’t need to hit it exactly.',
      'The cursor holds still while you blink.',
    ],
  },
  {
    mode: 'eyes', icon: 'blink', title: 'Clicking',
    items: [
      'Blink twice quickly, like a relaxed “yes, yes”.',
      'After the first blink a small “1” appears on the cursor and the button pulses.',
      'The click goes to what you looked at before the first blink.',
    ],
  },
  {
    mode: 'eyes', icon: 'wink', title: 'Holding, dragging and the menu',
    items: [
      'Close one eye and keep it closed: that is like holding the mouse button down where you were looking.',
      'Keep it closed and look somewhere else to drag (try Arrange), or to draw (try Draw). Open the eye to drop.',
      'Keep looking at the same spot for a second instead: a menu opens (a right click) — Click, Pick up to move, Read aloud.',
      'Open the eye again quickly without looking away: a click.',
      'The cursor keeps following your open eye the whole time.',
    ],
  },
  {
    mode: 'eyes', icon: 'blink', title: 'If winking or blinking is hard',
    items: [
      'Settings → Eye gestures → Test my winks checks each eye; an eye that can’t wink on its own is ignored.',
      'Close both eyes for about a second: can open the menu, pick up / drop, or click.',
      'Dwell click: rest your eyes on a button until the ring fills — no blinking at all.',
      '“Pick up to move” in the menu carries a thing without holding; blink twice (or dwell) to drop it.',
      'If one eye squints or droops, the tracker can follow the steadier eye (Tracking eye: Auto).',
    ],
  },
  {
    mode: 'eyes', icon: 'scroll', title: 'Scrolling',
    items: [
      'Look at the Scroll up or Scroll down zone on the right edge.',
      'Look further towards the arrow to scroll faster; look away to stop.',
      'Blink twice on a zone to jump a whole page.',
    ],
  },
  {
    mode: 'eyes', icon: 'pause', title: 'Pausing',
    items: [
      'Look at Pause (top right) and blink twice to rest your eyes.',
      'While paused nothing gets clicked. Blink twice to resume.',
    ],
  },
  {
    mode: 'eyes', icon: 'crosshair', title: 'When the cursor drifts',
    items: [
      'Run a Quick adjust (9 dots) from Home or Settings.',
      'If it is far off, run a Full calibration.',
      'Calibrate sitting the way you will use the computer.',
    ],
  },
  {
    mode: 'eyes', icon: 'keyboard', title: 'Keyboard shortcuts for helpers',
    html: '<ul><li><kbd>C</kbd> full calibration</li><li><kbd>A</kbd> quick adjust</li><li><kbd>P</kbd> pause / resume</li><li><kbd>Esc</kbd> cancel a calibration</li><li><kbd>B</kbd> blink (mouse demo mode: press twice to click, hold 1 s for a long close)</li><li><kbd>Q</kbd> / <kbd>E</kbd> hold to keep the left / right eye closed (demo mode)</li><li><kbd>Esc</kbd> drop a carried item / close the menu</li><li><kbd>F11</kbd> full screen</li></ul>',
  },
  {
    mode: 'eyes', icon: 'sun', title: 'Troubleshooting',
    items: [
      'Cursor jumpy: choose more smoothing, improve lighting, sit still while calibrating.',
      'Blinks missed: raise blink sensitivity or choose the Relaxed double-blink speed.',
      'Unwanted clicks: lower blink sensitivity or choose the Fast speed.',
      'No face found: face the camera, light your face from the front, remove strong backlight.',
      'Camera blocked: allow it via the camera icon in the address bar and reload.',
    ],
  },
  // Hand mode, after the "Using your hands" section below.
  {
    mode: 'hands', icon: 'move', title: 'Moving things',
    items: [
      'In Arrange, pinch a planet to pick it up: it follows your finger. Pinch on a place to put it down there; pinch anywhere else to put it back.',
      'Drawing needs one eye held closed, so Draw works with eye control only.',
    ],
  },
  {
    mode: 'hands', icon: 'crosshair', title: 'When the cursor is off',
    items: [
      'Re-point from Home, Settings or the Lab: just the pointing dots, about 30 seconds.',
      'If pinches are missed or happen by themselves, redo the full hand setup — it measures your hand and tunes the pinch to it.',
      'Do the setup with your hand where you will hold it.',
    ],
  },
  {
    mode: 'hands', icon: 'keyboard', title: 'Keyboard shortcuts for helpers',
    html: '<ul><li><kbd>C</kbd> full hand setup</li><li><kbd>A</kbd> quick re-point</li><li><kbd>P</kbd> pause / resume</li><li><kbd>Esc</kbd> cancel the hand setup</li><li><kbd>Esc</kbd> put a carried item back</li><li><kbd>F11</kbd> full screen</li></ul>',
  },
  {
    mode: 'hands', icon: 'help', title: 'Troubleshooting',
    items: [
      'Cursor jumpy: add light, keep your whole hand in view and rest your elbow.',
      'Pinches missed: pinch clearly, tip to tip, or redo the full hand setup.',
      'Unwanted clicks: keep your thumb away from your index finger while you point.',
      'It pauses by itself: curl your other fingers while you point — only a spread hand pauses.',
      'No hand found: raise your hand into view, palm towards the camera, and light it from the front.',
      'Camera blocked: allow it via the camera icon in the address bar and reload.',
    ],
  },
  {
    icon: 'camera', title: 'Privacy',
    items: [
      'Video is analysed on this computer by the Python server and never stored.',
      ['Your calibration (numbers only, no images) is saved in the data folder so you don’t have to recalibrate each time.',
        'Your hand setup (numbers only, no images) is saved in the data folder so you don’t have to redo it each time.'],
    ],
  },
];

// "Using your hands": the four hand gestures and how to be seen well.
const HAND_CARDS = [
  {
    icon: 'point', title: 'Pointing',
    items: [
      'Hold one hand up in front of the camera and point with your index finger: the cursor follows your fingertip.',
      'The nearest button lights up; you don’t need to hit it exactly.',
      'Small movements are enough — the hand setup stretches the reach of your fingertip to the whole screen.',
    ],
  },
  {
    icon: 'pinch', title: 'Clicking: pinch',
    items: [
      'Touch the tips of your thumb and index finger together, then let go.',
      'The cursor holds still while your fingers close, and the click goes to what you pointed at just before.',
      'Keep it quick: a long pinch, or one that moves your hand a lot, doesn’t click.',
    ],
  },
  {
    icon: 'scroll', title: 'Scrolling: pinch and move',
    items: [
      'Pinch and keep pinching, then move your hand down to scroll down or up to scroll up. Let go to stop.',
      'Or point at the Scroll up or Scroll down zone on the right edge; pinch on a zone to jump a whole page.',
    ],
  },
  {
    icon: 'hand', title: 'Pausing: open hand',
    items: [
      'Hold up an open hand, all five fingers spread, and keep it still for a moment: Paralic pauses.',
      'Do the same again to resume. While paused nothing gets clicked, not even by a pinch.',
      'Only a spread hand pauses, so pointing never does. You can also point at Pause (top right) and pinch.',
    ],
  },
  {
    icon: 'sun', title: 'Tips for good tracking', cls: 'wide',
    items: [
      'Light your hand well from the front, and avoid bright light behind you.',
      'Keep your hand about 40–60 cm from the camera, palm facing the camera, with the whole hand in view.',
      'Rest your elbow on the desk or an armrest: your arm tires less and the cursor stays steadier.',
      'Use one hand at a time — Paralic follows a single hand.',
    ],
  },
];

const MODE_CLASS = { eyes: 'eyes-only', hands: 'hands-only' };

function item(text) {
  return Array.isArray(text) ? h('li', { html: sayHtml(...text) }) : h('li', {}, text);
}

function card(c) {
  return h('section', { class: ['card', MODE_CLASS[c.mode], c.cls].filter(Boolean).join(' ') },
    h('h3', { html: `${icon(c.icon)}<span>${esc(c.title)}</span>` }),
    c.html ? h('div', { class: 'muted', html: c.html }) : h('ul', {}, c.items.map(item)));
}

export default {
  title: 'Help',
  render(el) {
    el.append(
      h('div', { class: 'page-head' },
        h('div', {},
          h('div', { class: 'eyebrow' }, 'Help'),
          h('h1', { html: sayHtml('Using Paralic with your eyes', 'Using Paralic with your hands') }),
          h('p', { class: 'muted', html: sayHtml('Look, blink twice, and look at the arrows on the right to scroll. Here are the details.',
            'Point with your index finger, pinch to click, and pinch and move your hand to scroll. Here are the details.') }))),
      h('h2', { class: 'section-title hands-only', html: `${icon('hand')}<span>Using your hands</span>` }),
      h('div', { class: 'help-grid hands-only' }, HAND_CARDS.map(card)),
      h('h2', { class: 'section-title hands-only', html: `${icon('help')}<span>More help</span>` }),
      h('div', { class: 'help-grid' }, CARDS.map(card)),
      h('div', { class: 'btn-row', style: { marginTop: 'var(--gap)' } },
        h('a', { class: 'btn primary', href: '#/practice', html: `${icon('target')}<span>Practice clicking</span>` }),
        h('a', { class: 'btn', href: '#/read/tips', html: `${icon('sparkle')}<span>Accuracy tips</span>` })),
    );
  },
};
