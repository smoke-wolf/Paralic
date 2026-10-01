import { esc, h, toast } from '../dom.js';
import { icon } from '../icons.js';
import { getSettings, updateSettings, DEFAULTS } from '../settings.js';
import { handMode, handSummary, say, sayHtml } from '../mode.js';

// In the groups below `hands` is the description in hand mode, and `eyesOnly`
// hides a setting in hand mode because only the eye tracking uses it (see
// mode.js: the page may render before the mode is known, so CSS decides).

// Settings kept in this browser.
const GROUPS = [
  {
    key: 'smoothing', title: 'Cursor smoothing', eyesOnly: true,
    desc: 'More smoothing gives a steadier cursor that follows a little more slowly. Auto uses the level learned from your calibration and experiments.',
    options: [['auto', 'Auto'], ['low', 'Low'], ['medium', 'Medium'], ['high', 'High']],
  },
  {
    key: 'snap', title: 'Snap to buttons',
    desc: 'Pull the cursor towards the nearest button so you don’t need perfect aim. Auto sizes it from your accuracy.',
    hands: 'Pull the cursor towards the nearest button so you don’t need perfect aim. With a hand, Auto is Normal.',
    options: [['auto', 'Auto'], ['off', 'Off'], ['normal', 'Normal'], ['strong', 'Strong']],
  },
  {
    key: 'doubleBlink', title: 'Double-blink speed', eyesOnly: true,
    desc: 'How soon the second blink must follow the first. Personal uses your blink test. Choose Relaxed if clicks are missed.',
    options: [['personal', 'Personal'], ['fast', 'Fast'], ['normal', 'Normal'], ['relaxed', 'Relaxed']],
  },
  {
    key: 'blinkSensitivity', title: 'Blink sensitivity', eyesOnly: true,
    desc: 'Raise it if your blinks are not noticed; lower it if you get clicks you didn’t mean. Personal uses your blink test.',
    options: [['personal', 'Personal'], ['low', 'Low'], ['normal', 'Normal'], ['high', 'High']],
  },
  {
    key: 'scrollSpeed', title: 'Scroll speed',
    desc: 'How fast pages move while you look at the scroll arrows on the right.',
    hands: 'How fast pages move while you point at the scroll arrows on the right. Pinch-and-move scrolling follows your hand.',
    options: [['slow', 'Slow'], ['normal', 'Normal'], ['fast', 'Fast']],
  },
  {
    key: 'driftCorrection', title: 'Learn from clicks',
    desc: 'Each click gently shifts the cursor to line up with the buttons you actually pick.',
    options: [[true, 'On'], [false, 'Off']],
  },
  {
    key: 'learning', title: 'Keep learning my eyes', eyesOnly: true,
    desc: 'Practice targets and clicks become training data, and your gaze network is fine-tuned in the background.',
    options: [[true, 'On'], [false, 'Off']],
  },
  {
    key: 'cursorSize', title: 'Cursor size',
    desc: 'Size of the gaze cursor.',
    hands: 'Size of the cursor.',
    options: [['small', 'Small'], ['normal', 'Normal'], ['large', 'Large']],
  },
  {
    key: 'showCamera', title: 'Camera preview',
    desc: 'Small video in the corner with your eyes outlined, the frame rate and a blink meter.',
    hands: 'Small video in the corner with the frame rate, to check that your hand is in view.',
    options: [[true, 'Show'], [false, 'Hide']],
  },
  {
    key: 'sounds', title: 'Sounds',
    desc: 'Short sounds for the first blink, clicks, presses and calibration steps.',
    hands: 'Short sounds when a pinch starts, for clicks and for the hand setup steps.',
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
    key: 'left_hold', title: 'Hold your left eye closed', eye: 'left', eyesOnly: true,
    desc: 'Like holding the mouse button: look elsewhere to drag, keep still for a second for the menu, open quickly to click.',
    options: HOLD,
  },
  {
    key: 'right_hold', title: 'Hold your right eye closed', eye: 'right', eyesOnly: true,
    desc: 'The same for your right eye — or make it a right click.',
    options: HOLD,
  },
  {
    key: 'quick', title: 'Short wink', eyesOnly: true,
    desc: 'A quick wink of either eye (shorter than the hold time).',
    options: [['off', 'Nothing'], ['click', 'Click'], ['menu', 'Menu']],
    get: (g) => (g.left_quick === g.right_quick ? g.left_quick : 'off'),
    set: (v) => ({ left_quick: v, right_quick: v }),
  },
  {
    key: 'long_close', title: 'Close both eyes for a second', eyesOnly: true,
    desc: 'For when winking is hard. Menu, pick up and drop things, or click.',
    options: [['off', 'Nothing'], ['menu', 'Menu'], ['grab', 'Pick up / drop'], ['click', 'Click']],
  },
  {
    // The cursor clicks what it rests on - with the eyes or with a finger.
    key: 'dwell', title: 'Dwell click',
    desc: 'Rest your eyes on a button to click it, no blinking needed. A ring fills on the cursor first.',
    hands: 'Keep pointing at a button to click it, no pinch needed. A ring fills on the cursor first.',
    options: [['off', 'Off'], ['600', '0.6 s'], ['800', '0.8 s'], ['1000', '1 s'], ['1500', '1.5 s'], ['2000', '2 s']],
    get: (g) => (g.dwell ? String(g.dwell_ms) : 'off'),
    set: (v) => (v === 'off' ? { dwell: false } : { dwell: true, dwell_ms: Number(v) }),
  },
  {
    key: 'hold_ms', title: 'Wink hold time', eyesOnly: true,
    desc: 'How long an eye must stay closed before it counts as holding.',
    options: [['250', '0.25 s'], ['350', '0.35 s'], ['600', '0.6 s'], ['1000', '1 s']],
    get: (g) => String(g.hold_ms),
    set: (v) => ({ hold_ms: Number(v) }),
  },
  {
    key: 'tracking_eye', title: 'Tracking eye', eyesOnly: true,
    desc: 'Auto lets cross-validation pick both eyes or the steadier one (e.g. with a squint or a drooping lid).',
    options: [['auto', 'Auto'], ['both', 'Both'], ['left', 'Left only'], ['right', 'Right only']],
  },
];

// Per-person cursor movement (web/js/motion.js), stored like the gestures. A
// fingertip cursor moves its own way (gaze.js), so these are for the eyes.
const NUDGE_SPEEDS = [['35', 'Slow'], ['60', 'Normal'], ['100', 'Fast']];
const MOTION_GROUPS = [
  {
    key: 'motion', title: 'Cursor movement', eyesOnly: true,
    desc: 'How the cursor travels to where you look. Glide moves gracefully and is easiest to follow; Snappy gets there fastest.',
    options: [['glide', 'Glide'], ['balanced', 'Balanced'], ['snappy', 'Snappy']],
  },
  {
    key: 'hold_still', title: 'Hold still while you look', eyesOnly: true,
    desc: 'While your eyes rest on one place, the cursor sits on the average of where you look: steadier, and closer to the spot.',
    options: [['true', 'On'], ['false', 'Off']],
    get: (g) => String(g.hold_still !== false),
    set: (v) => ({ hold_still: v === 'true' }),
  },
  {
    key: 'head_nudge', title: 'Head nudge', eyesOnly: true,
    desc: 'Tilt your head a little up, down, left or right to move the cursor the last bit of the way — the further you tilt, the faster it goes. Hold your head still to stop. Looking somewhere else starts afresh. (Demo mode: hold the arrow keys.)',
    options: [['off', 'Off'], ...NUDGE_SPEEDS],
    get: (g) => {
      if (!g.head_nudge) return 'off';
      const speed = Number(g.nudge_speed) || 60;
      return NUDGE_SPEEDS.reduce((best, [v]) => (Math.abs(v - speed) < Math.abs(best - speed) ? v : best), '60');
    },
    set: (v) => (v === 'off' ? { head_nudge: false } : { head_nudge: true, nudge_speed: Number(v) }),
  },
];

// Face print (per person): stored on this computer only.
const FACE_GROUPS = [
  {
    key: 'faceprint', title: 'Recognise my face', eyesOnly: true,
    desc: 'Paralic keeps a face print — a few different views of your face, as numbers and small grey pictures, on this computer only — so it knows who is using it and opens your profile by itself. Off deletes it.',
    options: [['true', 'On'], ['false', 'Off']],
    get: (g, view) => String(!view.faceprint || view.faceprint.enabled !== false),
    command: (v) => ({ type: 'faceprint_set', enabled: v === 'true' }),
  },
];

/** A setting's description: with a hand-mode wording when it has one. */
function describe(g) {
  return g.hands ? h('p', { html: sayHtml(g.desc, g.hands) }) : h('p', {}, g.desc);
}

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
        h('p', { class: 'muted', html: sayHtml('Look at an option and blink twice to choose it. Changes apply immediately.',
          'Point at an option and pinch to choose it. Changes apply immediately.') }))));

    // -- control with: eyes or a hand (remembered in this browser) --------------------
    el.append(h('h2', { class: 'section-title', html: `${icon('eye', 'say-eyes')}${icon('hand', 'say-hands')}<span>Control with</span>` }));
    const modeChoice = (mode, ic, label, sub) => {
      const b = h('button', {
        class: 'choice mode-choice', type: 'button', 'data-mode': mode, 'data-no-learn': '',
        html: `${icon(ic)}<span>${label}</span><small>${sub}</small><small class="mode-current">In use</small>`,
      });
      b.addEventListener('click', () => {
        if (!app.state.simulated && mode === (handMode() ? 'hand' : 'eyes')) {
          toast(say('You are using your eyes already', 'You are using your hand already'), 'ok');
          return;
        }
        app.setMode?.(mode);  // remembers the choice and reloads into it
      });
      return b;
    };
    el.append(
      h('div', { class: 'mode-choices', role: 'group', 'aria-label': 'Control with' },
        modeChoice('eyes', 'eye', 'Eyes', 'Look to move the cursor, blink twice to click'),
        modeChoice('hand', 'hand', 'Hands', 'Point with your index finger, pinch to click')),
      h('p', { class: 'muted mode-note' }, 'Paralic restarts in the way you choose. Your people and settings stay.'));

    // Hand mode: the person's hand setup (pointing map and pinch).
    const handCard = h('section', { class: 'setting hands-only', 'data-setting': 'hand-setup' });
    const paintHand = () => {
      const s = app.state.hand;
      handCard.innerHTML = '';
      handCard.append(
        h('div', {}, h('h3', {}, 'Your hand setup'), h('p', {}, s ? handSummary(s)
          : 'Not set up yet. It takes about a minute: hold up your hand, point at a few dots, then pinch a few times.')),
        h('div', { class: 'btn-row' },
          s ? actionBtn('Re-point', 'crosshair', () => app.handSetup?.('point'), 'primary') : null,
          actionBtn(s ? 'Redo hand setup' : 'Start hand setup', 'refresh', () => app.handSetup?.('full'), s ? '' : 'primary')));
    };
    paintHand();
    el.append(handCard);

    if (!app.state.simulated) {
      el.append(h('div', { class: 'settings-actions eyes-only' },
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

    // -- cursor movement and eye gestures (per person) --------------------------------
    const person = app.state.person ? app.state.person.name : null;
    const gestureRows = [];
    const section = (title, ic, groups, { hands = null, handsIcon = null, cls = '' } = {}) => {
      const label = hands ? sayHtml(title, hands) : esc(title);
      const pic = handsIcon ? icon(ic, 'say-eyes') + icon(handsIcon, 'say-hands') : icon(ic);
      el.append(h('h2', { class: `section-title ${cls}`, html: `${pic}<span>${label}${person ? ` for ${esc(person)}` : ''}</span>` }));
      for (const g of groups) addGesture(g);
    };
    const addGesture = (g) => {
      const opts = h('div', { class: 'options', role: 'radiogroup', 'aria-label': g.title });
      const note = h('p', { class: 'setting-note', hidden: true });
      for (const [value, label] of g.options) {
        const b = h('button', { class: 'opt', type: 'button', 'data-value': value }, label);
        b.addEventListener('click', async () => {
          const patch = g.set ? g.set(value) : { [g.key]: value };
          try {
            const cmd = g.command ? g.command(value) : { type: 'gestures_set', gestures: patch };
            const reply = await app.tracker.request(cmd, 'personal', 8000);
            if (reply.ok === false) throw new Error(reply.error);
          } catch (err) {
            toast(`Could not save: ${err.message}`, 'bad');
          }
        });
        opts.append(b);
      }
      const row = h('section', { class: `setting ${g.eyesOnly ? 'eyes-only' : ''}`, 'data-gesture': g.key },
        h('div', {}, h('h3', {}, g.title), describe(g), note), opts);
      gestureRows.push({ g, opts, note });
      el.append(row);
    };
    section('Cursor movement', 'move', MOTION_GROUPS, { cls: 'eyes-only' });
    if (!app.state.simulated) {
      el.append(h('div', { class: 'btn-row eyes-only', style: { marginBottom: 'var(--gap)' } },
        actionBtn('Check head directions', 'head', () => app.calibrate('head'))));
    }
    // With a hand only the dwell click is left here.
    section('Eye gestures', 'wink', GESTURE_GROUPS, { hands: 'Gestures', handsIcon: 'hand' });
    if (!app.state.simulated) section('Recognise me', 'head', FACE_GROUPS, { cls: 'eyes-only' });
    const paintGestures = () => {
      const view = app.state.personal || {};
      const gs = view.gestures || {};
      for (const { g, opts, note } of gestureRows) {
        const current = g.get ? g.get(gs, view) : String(gs[g.key]);
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
      app.state.simulated ? null : actionBtn('Test my winks', 'wink', () => app.calibrate('wink'), 'primary eyes-only'),
      app.state.simulated ? null : actionBtn('Test my blinks', 'blink', () => app.calibrate('blink'), 'eyes-only'),
      // Dragging works with a hand too: a pinch picks a thing up (gestures.js).
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
      el.append(h('section', { class: `setting ${g.eyesOnly ? 'eyes-only' : ''}`, 'data-setting': g.key },
        h('div', {}, h('h3', {}, g.title), describe(g)), opts));
    }

    el.append(h('div', { class: 'btn-row', style: { marginTop: '10px' } },
      app.state.simulated ? null : actionBtn(
        (app.state.systemControl && app.state.systemControl.enabled) ? 'Stop controlling my computer' : 'Control my whole computer (beta)',
        'mouse', () => app.toggleSystemControl(),
        (app.state.systemControl && app.state.systemControl.enabled) ? 'primary' : ''),
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

    const offs = [app.on('personal', paintGestures), app.on('hand', paintHand), app.on('people', paintHand)];
    return () => offs.forEach((off) => off());
  },
};
