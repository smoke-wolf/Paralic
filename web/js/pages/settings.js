import { h, toast } from '../dom.js';
import { icon } from '../icons.js';
import { getSettings, updateSettings, DEFAULTS } from '../settings.js';

const GROUPS = [
  {
    key: 'smoothing', title: 'Cursor smoothing',
    desc: 'More smoothing gives a steadier cursor that follows a little more slowly.',
    options: [['low', 'Low'], ['medium', 'Medium'], ['high', 'High']],
  },
  {
    key: 'snap', title: 'Snap to buttons',
    desc: 'Pull the cursor towards the nearest button so you don’t need perfect aim.',
    options: [['off', 'Off'], ['normal', 'Normal'], ['strong', 'Strong']],
  },
  {
    key: 'doubleBlink', title: 'Double-blink speed',
    desc: 'How soon the second blink must follow the first. Choose Relaxed if clicks are missed.',
    options: [['fast', 'Fast'], ['normal', 'Normal'], ['relaxed', 'Relaxed']],
  },
  {
    key: 'blinkSensitivity', title: 'Blink sensitivity',
    desc: 'Raise it if your blinks are not noticed; lower it if you get clicks you didn’t mean.',
    options: [['low', 'Low'], ['normal', 'Normal'], ['high', 'High']],
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
    desc: 'Short sounds for the first blink, clicks and calibration steps.',
    options: [[true, 'On'], [false, 'Off']],
  },
  {
    key: 'speechRate', title: 'Speaking speed',
    desc: 'Speed of the voice on the Talk page.',
    options: [['slow', 'Slow'], ['normal', 'Normal'], ['fast', 'Fast']],
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
  },
};
