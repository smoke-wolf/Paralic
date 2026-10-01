// Glasses: what the page says and offers when the camera sees glasses.
//
// The server (paralic/glasses.py, session.py) notices glasses and reflections
// on their lenses. With a reflection on one lens the cursor follows the other
// eye by itself, and each person can keep a calibration made with glasses and
// one made without. This module, in eye mode:
//   - offers to switch calibration - or to do a quick adjust - when glasses
//     go on or come off during a session (`glasses_changed`),
//   - says, now and then, when a reflection keeps hiding an eye,
//   - words the welcome-back choice when the saved calibration was made the
//     other way (`glassesAdvice`).

import { h, toast } from './dom.js';
import { icon } from './icons.js';

const GLARE_NOTE_MS = 3000;                  // a reflection this long gets a note...
const GLARE_NOTE_GAP_MS = 10 * 60 * 1000;    // ...at most this often
const BANNER_MS = 20000;

/** A sentence for the welcome-back choice when the calibration was made the other way, else null. */
export function glassesAdvice(p) {
  if (!p || !p.glasses_mismatch) return null;
  return p.glasses_now
    ? 'You are wearing glasses now, but this calibration was made without them — a quick adjust makes it accurate again.'
    : 'This calibration was made with your glasses on — a quick adjust makes it accurate without them.';
}

export function attachGlassesUi(app) {
  const t = app.tracker;
  let glareSince = null;
  let lastNote = -Infinity;
  t.on('frame', (m) => {
    if (!m.face || !m.glare || app.state.handMode || !app.state.calibrated || app.calibrator.running) {
      glareSince = null;
      return;
    }
    const now = performance.now();
    glareSince = glareSince ?? now;
    if (now - glareSince < GLARE_NOTE_MS || now - lastNote < GLARE_NOTE_GAP_MS) return;
    lastNote = now;
    const other = { left: 'right', right: 'left' }[m.glare];
    toast(other
      ? `A reflection on your glasses hides your ${m.glare} eye, so the cursor follows your ${other} eye. Tilting the screen a little or moving the lamp helps.`
      : 'Reflections on your glasses make tracking harder — tilt the screen a little or move the lamp.', 'warn', 9000);
  });
  t.on('glasses_changed', (m) => offerSwitch(app, m));
}

/** Glasses went on or came off: use the matching calibration, or adjust. */
function offerSwitch(app, m) {
  if (!m || m.in_use || app.state.handMode || !app.state.calibrated || app.calibrator.running
      || app.overlayRoot.children.length) return;          // in_use: the calibration in use fits already
  document.querySelector('.glasses-banner')?.remove();
  const on = !!m.glasses;
  const button = (label, ic, primary) => h('button', { class: `btn${primary ? ' primary' : ''}`, type: 'button',
    html: ic ? `${icon(ic)}<span>${label}</span>` : `<span>${label}</span>` });
  const use = m.slot_available
    ? button(on ? 'Use my glasses calibration' : 'Use my calibration without glasses', 'refresh', true) : null;
  const adjust = button('Quick adjust', 'crosshair', !m.slot_available);
  const no = button('Not now', null, false);
  const banner = h('div', { class: 'face-banner glasses-banner', role: 'alert' },
    h('span', {}, on ? 'Glasses on?' : 'Glasses off?'), use, adjust, no);
  document.body.append(banner);
  const close = () => {
    clearTimeout(timer);
    banner.remove();
  };
  const timer = setTimeout(close, BANNER_MS);
  no.addEventListener('click', close);
  adjust.addEventListener('click', () => {
    close();
    app.calibrate('adjust');
  });
  if (use) {
    use.addEventListener('click', async () => {
      close();
      try {
        const p = await app.tracker.request({ type: 'profile_load', glasses: on }, 'profile', 15000);
        if (!p.loaded) throw new Error(p.error || 'not loaded');
        if (p.personal) app.setPersonal(p.personal);
        app.state.accuracy = p.accuracy_px || null;
        app.gaze.resetBias();
        toast(on ? 'Using your calibration with glasses' : 'Using your calibration without glasses', 'ok');
      } catch (err) {
        toast(`Could not switch calibration: ${err.message || err}`, 'bad');
      }
    });
  }
}
