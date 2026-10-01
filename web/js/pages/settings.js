import { h, toast } from '../dom.js';
import { icon } from '../icons.js';
import { getSettings, updateSettings, DEFAULTS } from '../settings.js';

// Settings kept in this browser.
const GROUPS = [
  {
    key: 'smoothing', title: 'Cursor smoothing',
    desc: 'More smoothing gives a steadier cursor that follows a little more slowly. Auto uses the level learned from your calibration and experiments.',
    options: [['auto', 'Auto'], ['low', 'Low'], ['medium', 'Medium'], ['high', 'High']],
  },
  {
    key: 'snap', title: 'Snap to buttons',
    desc: 'Pull the cursor towards the nearest button so you don’t need perfect aim. Auto sizes it from your accuracy.',
    options: [['auto', 'Auto'], ['off', 'Off'], ['normal', 'Normal'], ['strong', 'Strong']],
  },
  {
    key: 'doubleBlink', title: 'Double-blink speed',
    desc: 'How soon the second blink must follow the first. Personal uses your blink test. Choose Relaxed if clicks are missed.',
    options: [['personal', 'Personal'], ['fast', 'Fast'], ['normal', 'Normal'], ['relaxed', 'Relaxed']],
  },
  {
    key: 'blinkSensitivity', title: 'Blink sensitivity',
    desc: 'Raise it if your blinks are not noticed; lower it if you get clicks you didn’t mean. Personal uses your blink test.',
    options: [['personal', 'Personal'], ['low', 'Low'], ['normal', 'Normal'], ['high', 'High']],
  },
  {
    key: 'scrollSpeed', title: 'Scroll speed',
    desc: 'How fast pages move while you look at the scroll arrows on the right.',
    options: [['slow', 'Slow'], ['normal', 'Normal'], ['fast', 'Fast']],
  },
  {
    key: 'driftCorrection', title: 'Learn from clicks',
    desc: 'Each click gently shifts the cursor to line up with the buttons you actually pick.',
    options: [[true, 'On'], [false, 'Off']],
  },
  {
    key: 'learning', title: 'Keep learning my eyes',
    desc: 'Practice targets and clicks become training data, and your gaze network is fine-tuned in the background.',
    options: [[true, 'On'], [false, 'Off']],
  },
  {
    key: 'cursorSize', title: 'Cursor size',
    desc: 'Size of the gaze cursor.',
    options: [['small', 'Small'], ['normal', 'Normal'], ['large', 'Large']],
  },
  {
    key: 'showCamera', title: 'Camera preview',
    desc: 'Small video in the corner with your eyes outlined, the frame rate and a blink meter.',
    options: [[true, 'Show'], [false, 'Hide']],
  },
  {
    key: 'sounds', title: 'Sounds',
    desc: 'Short sounds for the first blink, clicks, presses and calibration steps.',
    options: [[true, 'On'], [false, 'Off']],
  },
  {
    key: 'speechRate', title: 'Speaking speed',
    desc: 'Speed of the voice on the Talk page and for “Read aloud”.',
    options: [['slow', 'Slow'], ['normal', 'Normal'], ['fast', 'Fast']],
  },
];

// Per-person gesture settings, stored with the person's profile on this computer.
const HOLD = [['drag', 'Press & drag'], ['menu', 'Right-click menu'], ['off', 'Off']];
const GESTURE_GROUPS = [
  {
    key: 'left_hold', title: 'Hold your left eye closed', eye: 'left',
    desc: 'Like holding the mouse button: look elsewhere to drag, keep still for a second for the menu, open quickly to click.',
    options: HOLD,
  },
  {
    key: 'right_hold', title: 'Hold your right eye closed', eye: 'right',
    desc: 'The same for your right eye — or make it a right click.',
    options: HOLD,
  },
  {
    key: 'quick', title: 'Short wink',
    desc: 'A quick wink of either eye (shorter than the hold time).',
    options: [['off', 'Nothing'], ['click', 'Click'], ['menu', 'Menu']],
    get: (g) => (g.left_quick === g.right_quick ? g.left_quick : 'off'),
    set: (v) => ({ left_quick: v, right_quick: v }),
  },
  {
    key: 'long_close', title: 'Close both eyes for a second',
    desc: 'For when winking is hard. Menu, pick up and drop things, or click.',
    options: [['off', 'Nothing'], ['menu', 'Menu'], ['grab', 'Pick up / drop'], ['click', 'Click']],
  },
  {
    key: 'dwell', title: 'Dwell click',
    desc: 'Rest your eyes on a button to click it, no blinking needed. A ring fills on the cursor first.',
    options: [['off', 'Off'], ['600', '0.6 s'], ['800', '0.8 s'], ['1000', '1 s'], ['1500', '1.5 s'], ['2000', '2 s']],
    get: (g) => (g.dwell ? String(g.dwell_ms) : 'off'),
    set: (v) => (v === 'off' ? { dwell: false } : { dwell: true, dwell_ms: Number(v) }),
  },
  {
    key: 'hold_ms', title: 'Wink hold time',
    desc: 'How long an eye must stay closed before it counts as holding.',
    options: [['250', '0.25 s'], ['350', '0.35 s'], ['600', '0.6 s'], ['1000', '1 s']],
    get: (g) => String(g.hold_ms),
    set: (v) => ({ hold_ms: Number(v) }),
  },
  {
    key: 'tracking_eye', title: 'Tracking eye',
    desc: 'Auto lets cross-validation pick both eyes or the steadier one (e.g. with a squint or a drooping lid).',
    options: [['auto', 'Auto'], ['both', 'Both'], ['left', 'Left only'], ['right', 'Right only']],
  },
];

export default {
  title: 'Settings',
  render(el, params, app) {
    const actionBtn = (label, ic, onClick, cls = '') => {
      const b = h('button', { class: `btn ${cls}`, type: 'button', html: `${icon(ic)}<span>${label}</span>` });
      b.addEventListener('click', onClick);
      return b;
    };

    el.append(h('div', { class: 'page-head' },
      h('div', {},
        h('div', { class: 'eyebrow' }, 'Settings'),
        h('h1', {}, 'Make it yours'),
        h('p', { class: 'muted' }, 'Look at an option and blink twice to choose it. Changes apply immediately.'))));

    if (!app.state.simulated) {
      el.append(h('div', { class: 'settings-actions' },
        actionBtn('Quick adjust', 'crosshair', () => app.calibrate('adjust'), 'primary'),
        actionBtn('Full calibration', 'refresh', () => app.calibrate('full')),
        actionBtn('Forget calibration', 'trash', async () => {
          try {
            const res = await app.tracker.request({ type: 'profile_delete' }, 'profile', 5000);
            if (res.ok === false) throw new Error(res.error || 'unknown error');
            toast('Saved calibration deleted. It stays active until you close the page.', 'ok', 5000);
          } catch (err) {
            toast(`Could not delete: ${err.message}`, 'bad');
          }
        }, 'danger')));
    }

    // -- eye gestures (per person) -------------------------------------------------
    const person = app.state.person ? app.state.person.name : null;
    el.append(h('h2', { class: 'section-title', html: `${icon('wink')}<span>Eye gestures${person ? ` for ${person}` : ''}</span>` }));
    const gestureRows = [];
    for (const g of GESTURE_GROUPS) {
      const opts = h('div', { class: 'options', role: 'radiogroup', 'aria-label': g.title });
      const note = h('p', { class: 'setting-note', hidden: true });
      for (const [value, label] of g.options) {
        const b = h('button', { class: 'opt', type: 'button', 'data-value': value }, label);
        b.addEventListener('click', async () => {
          const patch = g.set ? g.set(value) : { [g.key]: value };
          try {
            const reply = await app.tracker.request({ type: 'gestures_set', gestures: patch }, 'personal', 8000);
            if (reply.ok === false) throw new Error(reply.error);
          } catch (err) {
            toast(`Could not save: ${err.message}`, 'bad');
          }
        });
        opts.append(b);
      }
      const row = h('section', { class: 'setting', 'data-gesture': g.key },
        h('div', {}, h('h3', {}, g.title), h('p', {}, g.desc), note), opts);
      gestureRows.push({ g, opts, note });
      el.append(row);
    }
    const paintGestures = () => {
      const view = app.state.personal || {};
      const gs = view.gestures || {};
      for (const { g, opts, note } of gestureRows) {
        const current = g.get ? g.get(gs) : String(gs[g.key]);
        for (const b of opts.children) b.classList.toggle('selected', b.dataset.value === current);
        const tested = view.wink_profile && view.wink_profile[g.eye];
        if (g.eye && tested && !tested.ok && gs[g.key] !== 'off' && view.winks && !view.winks[g.eye]) {
          note.hidden = false;
          note.innerHTML = '';
          const force = h('button', { class: 'btn small', type: 'button' }, 'Use it anyway');
          force.addEventListener('click', () => app.tracker.request(
            { type: 'gestures_set', gestures: { [`${g.eye}_forced`]: true } }, 'personal', 8000).catch(() => {}));
          note.append(`Your wink test found that this eye doesn’t close on its own reliably (${tested.reason}), so it is ignored. `, force);
        } else {
          note.hidden = true;
        }
      }
    };
    paintGestures();
    el.append(h('div', { class: 'btn-row', style: { marginBottom: 'var(--gap)' } },
      app.state.simulated ? null : actionBtn('Test my winks', 'wink', () => app.calibrate('wink'), 'primary'),
      app.state.simulated ? null : actionBtn('Test my blinks', 'blink', () => app.calibrate('blink')),
      h('a', { class: 'btn', href: '#/arrange', html: `${icon('move')}<span>Try dragging</span>` })));

    // -- this browser -----------------------------------------------------------------
    el.append(h('h2', { class: 'section-title', html: `${icon('sliders')}<span>Cursor, sounds and display</span>` }));
    for (const g of GROUPS) {
      const opts = h('div', { class: 'options', role: 'radiogroup', 'aria-label': g.title });
      const paint = () => {
        const current = getSettings()[g.key];
        for (const b of opts.children) b.classList.toggle('selected', b.dataset.value === String(current));
      };
      for (const [value, label] of g.options) {
        const b = h('button', { class: 'opt', type: 'button', 'data-value': String(value) }, label);
        b.addEventListener('click', () => {
          updateSettings({ [g.key]: value });
          paint();
        });
        opts.append(b);
      }
      paint();
      el.append(h('section', { class: 'setting' }, h('div', {}, h('h3', {}, g.title), h('p', {}, g.desc)), opts));
    }

    el.append(h('div', { class: 'btn-row', style: { marginTop: '10px' } },
      actionBtn('Full screen', 'grid', () => {
        const req = document.documentElement.requestFullscreen?.();
        if (!req) toast('Press F11 for full screen', 'warn');
        else req.catch(() => toast('Your browser needs a real click or key for this — press F11 for full screen', 'warn', 5000));
      }),
      actionBtn('Reset settings', 'refresh', () => {
        updateSettings({ ...DEFAULTS });
        app.router.render();
        toast('Settings reset', 'ok');
      })));

    return app.on('personal', paintGestures);
  },
};
