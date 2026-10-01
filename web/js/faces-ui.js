// Several faces (or hands) in view: show who is in control.
//
// The server follows one person (paralic/face_select.py) and ignores everyone
// else. Frame messages then carry `faces`, `you` and `others` (face boxes in
// image fractions) - or `hands` in hand mode - and `waiting` while the person
// being followed is out of view and someone else is not. This module outlines
// the faces in the corner camera view ("You" / "Ignored"), says once in a
// while that only the person in control counts, and shows a note while
// Paralic waits for them.

import { h, toast } from './dom.js';
import { say } from './mode.js';

const NOTE_GAP_MS = 5 * 60 * 1000;
const WAIT_NOTE_MS = 1500;

export function attachFacesUi(app) {
  const view = document.querySelector('#camera-panel .camera-view');
  const canvas = h('canvas', { class: 'faces-overlay', 'aria-hidden': 'true' });
  if (view) view.append(canvas);
  let noteAt = -Infinity;
  let waitingSince = null;
  let banner = null;
  app.tracker.on('frame', (m) => {
    draw(canvas, app.tracker.video, m);
    const now = performance.now();
    if (((m.faces || 0) > 1 || (m.hands || 0) > 1) && app.state.calibrated && now - noteAt > NOTE_GAP_MS) {
      noteAt = now;
      toast(say('Someone else is in view — only you control Paralic. Your face is outlined in the camera view.',
        'Another hand is in view — only the hand in control moves the cursor.'), 'ok', 7000);
    }
    if (m.waiting && app.state.calibrated) {
      waitingSince = waitingSince ?? now;
      if (!banner && now - waitingSince > WAIT_NOTE_MS) {
        banner = h('div', { class: 'face-banner waiting-banner', role: 'status' },
          'Waiting for you — someone else is in front of the camera. Paralic carries on when you are back.');
        document.body.append(banner);
      }
    } else {
      waitingSince = null;
      if (banner) {
        banner.remove();
        banner = null;
      }
    }
  });
}

/** Outline the faces on a canvas laid over the (mirrored) camera view. */
function draw(canvas, video, m) {
  if (!canvas.isConnected || !video || !video.videoWidth) return;
  const w = video.videoWidth;
  const hgt = video.videoHeight;
  if (canvas.width !== w || canvas.height !== hgt) {
    canvas.width = w;
    canvas.height = hgt;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, hgt);
  const others = m.others || [];
  if (!m.you && !others.length) return;
  const box = ([x0, y0, x1, y1], colour, label, dashed) => {
    ctx.strokeStyle = colour;
    ctx.lineWidth = Math.max(2, w / 160);
    ctx.setLineDash(dashed ? [10, 8] : []);
    ctx.strokeRect(x0 * w, y0 * hgt, (x1 - x0) * w, (y1 - y0) * hgt);
    ctx.setLineDash([]);
    // The canvas is mirrored like the video: mirror the text back.
    ctx.save();
    ctx.scale(-1, 1);
    ctx.font = `600 ${Math.round(w / 28)}px system-ui, sans-serif`;
    ctx.fillStyle = colour;
    ctx.textAlign = 'right';
    ctx.fillText(label, -x0 * w, y0 * hgt - 8);
    ctx.restore();
  };
  if (m.you) box(m.you, 'rgba(94, 234, 212, 0.95)', 'You', false);
  for (const b of others) box(b, 'rgba(255, 255, 255, 0.55)', 'Ignored', true);
}
