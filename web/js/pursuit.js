// Smooth-pursuit trails and the Snake tracking game.
//
// A dot glides along a procedurally generated path (served by /api/trail, see
// paralic/trails.py) and you follow it with your eyes. Two uses:
//   * practice / warm-up for smooth-pursuit control,
//   * a tracking *test* that measures how closely (and how promptly) your gaze
//     follows motion.
//
// It does not feed fine-tuning: the server labels the frames *before* a label
// event with its target, which suits a dot you rest your eyes on but not one
// that moves - the eyes trail a moving dot, so every label would carry that
// lag and teach the network an offset along the direction of motion.
//
// It reuses the live tracker and the app overlay and never changes the gaze
// model.

import { h } from './dom.js';
import { say } from './mode.js';
import { screenToClient } from './screen-space.js';
import { speak, canSpeak } from './speech.js';
import { sounds } from './sound.js';

async function fetchTrail(kind, seed) {
  const res = await fetch(`/api/trail/${kind}?n=400&seed=${seed}`);
  if (!res.ok) throw new Error(`trail ${kind} unavailable`);
  const data = await res.json();
  return (data.points || []).map((p) => [p.x, p.y]);
}

// Position along a polyline at parameter u in [0, 1] (linear between waypoints).
function sampleTrail(points, u) {
  if (points.length === 0) return [0.5, 0.5];
  const f = Math.max(0, Math.min(1, u)) * (points.length - 1);
  const i = Math.floor(f);
  const t = f - i;
  const a = points[i];
  const b = points[Math.min(points.length - 1, i + 1)];
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

export class PursuitTrainer {
  constructor(app) {
    this.app = app;
    this.cancelled = false;
    window.addEventListener('keydown', (e) => { if (this.running && e.key === 'Escape') this.cancelled = true; });
  }

  get tracker() { return this.app.tracker; }

  /**
   * Run one pursuit session.
   * @param kind  'lissajous' | 'spline' | 'snake'
   * @param opts  { durationMs, loops, game }
   */
  async run(kind = 'lissajous', { durationMs = 22000, loops = 1, game = false } = {}) {
    if (this.running || !this.tracker) return null;
    this.running = true;
    this.cancelled = false;
    this.app.gaze.setSuspended(true);
    const ov = this.app.openOverlay('calib solid pursuit');
    const dot = h('div', { class: 'pursuit-dot' });
    const trail = h('div', { class: 'pursuit-trail' });
    const hud = h('div', { class: 'pursuit-hud' });
    const text = h('div', { class: 'calib-text top' });
    ov.append(trail, dot, hud, text);

    // In hand mode the "gaze" is the fingertip, so the same game trains pointing.
    const title = game ? 'Snake chase' : 'Follow the dot';
    const sub = game ? say('Keep your eyes on the dot as it slithers. Stay locked on to score.',
                           'Keep pointing at the dot as it slithers. Stay locked on to score.')
                     : say('Let your eyes glide along with the dot — smooth and relaxed.',
                           'Let your fingertip glide along with the dot — smooth and relaxed.');
    text.append(h('h2', {}, title), h('p', {}, sub));
    if (canSpeak()) speak(`${title}. ${sub}`);
    await this._sleep(1800);

    let points;
    try {
      points = await fetchTrail(kind, (Date.now() % 997));
    } catch (err) {
      this.app.closeOverlay(ov); this.running = false; this.app.gaze.setSuspended(false);
      return { ok: false, error: String(err) };
    }

    // Live gaze from the tracker.
    let lastGaze = null;      // client {x,y}
    const off = this.tracker.on('frame', (m) => {
      lastGaze = (m.gaze && m.face) ? screenToClient(m.gaze[0], m.gaze[1]) : null;
    });

    const errors = [];        // px distance gaze<->dot while a face is seen
    let score = 0;
    const start = performance.now();
    const total = durationMs * Math.max(1, loops);

    try {
      for (;;) {
        if (this.cancelled) break;
        const now = performance.now();
        const elapsed = now - start;
        if (elapsed >= total) break;
        // Loop the path; ease within each loop for a natural speed.
        const u = (elapsed % durationMs) / durationMs;
        const [fx, fy] = sampleTrail(points, u);
        const cx = fx * window.innerWidth;
        const cy = fy * window.innerHeight;
        dot.style.translate = `${cx}px ${cy}px`;

        if (lastGaze) {
          const d = Math.hypot(lastGaze.x - cx, lastGaze.y - cy);
          errors.push(d);
          const lockRadius = 0.12 * Math.hypot(window.innerWidth, window.innerHeight);
          const locked = d <= lockRadius;
          dot.classList.toggle('locked', locked);
          if (locked && game) { score += 1; }
          if (game) hud.textContent = `Score ${score}`;
        }
        await this._frame();
      }
    } finally {
      off();
    }

    const meanErr = errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : NaN;
    const result = { ok: true, kind, meanErrorPx: Math.round(meanErr) || null, samples: errors.length, score };
    await this._showResult(ov, text, hud, result, game);

    this.app.closeOverlay(ov);
    this.running = false;
    this.app.gaze.setSuspended(false);
    return result;
  }

  runSnake(opts = {}) { return this.run('snake', { durationMs: 28000, game: true, ...opts }); }

  async _showResult(ov, text, hud, result, game) {
    hud.textContent = '';
    const diag = Math.hypot(window.innerWidth, window.innerHeight);
    const rel = result.meanErrorPx / diag;
    const rating = !result.meanErrorPx ? 'no data'
      : rel < 0.06 ? 'excellent' : rel < 0.1 ? 'good' : rel < 0.16 ? 'fair' : 'keep practising';
    const line = game
      ? `Score ${result.score}. Tracking was ${rating}` + (result.meanErrorPx ? `, about ${result.meanErrorPx} px off on average.` : '.')
      : `Pursuit tracking was ${rating}` + (result.meanErrorPx ? `, about ${result.meanErrorPx} px off on average.` : '.');
    text.innerHTML = '';
    text.append(h('h2', {}, game ? 'Nice chasing!' : 'All done'), h('p', {}, line));
    if (canSpeak()) speak(line);
    sounds.success && sounds.success();
    await this._sleep(2600);
  }

  _sleep(ms) {
    return new Promise((resolve) => {
      const end = performance.now() + ms;
      const tick = () => (this.cancelled || performance.now() >= end) ? resolve() : requestAnimationFrame(tick);
      tick();
    });
  }
  _frame() { return new Promise((r) => requestAnimationFrame(() => r())); }
}
