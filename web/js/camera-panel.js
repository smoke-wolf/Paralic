// The small camera preview in the corner (plus the eye overlay drawn from the
// landmarks the server sends back) and the blink meter.

import { $ } from './dom.js';

/**
 * Draw eye outlines and irises (normalised coordinates) on a canvas. An eye
 * held closed in a wink (`winking`: "left" / "right", the person's own eye)
 * is drawn in gold.
 */
export function drawEyes(canvas, eyes, width, height, winking = null) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  if (!eyes) return;
  ctx.lineWidth = Math.max(2, width / 150);
  for (const [eye, contour] of [['right', eyes.r], ['left', eyes.l]]) {
    ctx.strokeStyle = eye === winking ? 'rgba(251, 191, 36, 1)' : 'rgba(94, 234, 212, 0.95)';
    ctx.beginPath();
    contour.forEach(([x, y], i) => (i ? ctx.lineTo(x * width, y * height) : ctx.moveTo(x * width, y * height)));
    ctx.closePath();
    ctx.stroke();
  }
  ctx.fillStyle = 'rgba(167, 139, 250, 0.9)';
  for (const [x, y, r] of [eyes.ri, eyes.li]) {
    ctx.beginPath();
    ctx.arc(x * width, y * height, Math.max(2, r * width), 0, Math.PI * 2);
    ctx.fill();
  }
  if (eyes.box) {
    const [x0, y0, x1, y1] = eyes.box;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(x0 * width, y0 * height, (x1 - x0) * width, (y1 - y0) * height);
    ctx.setLineDash([]);
  }
}

export class CameraPanel {
  constructor(tracker) {
    this.tracker = tracker;
    this.panel = $('#camera-panel');
    this.video = $('#camera-video');
    this.canvas = $('#camera-overlay');
    this.dot = $('#status-dot');
    this.text = $('#status-text');
    this.fill = $('#blink-fill');
    this.thr = $('#blink-thr');
    this.extraViews = new Set();
    this.lastFace = 0;
    tracker.on('frame', (m) => this.update(m));
    tracker.on('connection', ({ connected }) => {
      if (!connected) this.setStatus('bad', 'Reconnecting…');
    });
  }

  setVisible(on) {
    this.panel.hidden = !on || this.tracker.simulated;
    document.body.classList.toggle('camera-on', !this.panel.hidden);
    this.syncOverlayRequest();
  }

  /** Extra (bigger) preview, e.g. on the start screen. */
  addView(canvas, video) {
    const view = { canvas, video };
    this.extraViews.add(view);
    this.syncOverlayRequest();
    return () => {
      this.extraViews.delete(view);
      this.syncOverlayRequest();
    };
  }

  syncOverlayRequest() {
    this.tracker.overlay = !this.panel.hidden || this.extraViews.size > 0;
  }

  setStatus(kind, text) {
    this.dot.className = `status-dot ${kind}`;
    this.text.textContent = text;
  }

  update(m) {
    if (m.face) {
      this.lastFace = performance.now();
      const lat = Math.round(this.tracker.latencyMs || 0);
      this.setStatus(m.gaze ? 'ok' : 'warn', `${Math.round(m.fps || 0)} fps`);
      this.panel.title = `${Math.round(m.fps || 0)} frames/s, ${lat} ms round trip`;
      this.fill.style.width = `${Math.min(100, (m.closure || 0) * 100)}%`;
      this.fill.classList.toggle('closed', !!m.closing);
      if (m.thr) this.thr.style.left = `${Math.min(100, m.thr[0] * 100)}%`;
    } else {
      this.setStatus('warn', m.error ? 'Frame error' : 'No face');
      this.fill.style.width = '0%';
    }
    const draw = (canvas, video) => {
      const w = video.videoWidth || 640;
      const hgt = video.videoHeight || 480;
      drawEyes(canvas, m.face ? m.eyes : null, w, hgt, m.winking);
    };
    if (!this.panel.hidden) draw(this.canvas, this.video);
    for (const v of this.extraViews) draw(v.canvas, v.video);
  }
}
