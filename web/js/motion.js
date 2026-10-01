// Cursor motion: how the drawn cursor moves between gaze estimates.
//
// Two small, DOM-free pieces used by the gaze controller (tested with node,
// see tests/js/motion.test.mjs):
//
// * CursorMotion - "graceful" movement. The cursor glides towards its goal on
//   a critically damped spring (eases in and out, never overshoots) instead of
//   jumping. With "hold still while you look" on, the goal during a fixation is
//   the average of that fixation's gaze samples (a running average that turns
//   into a 0.6 s moving average, so slow drifts are still followed): the cursor
//   rests steadily - and, because the noise averages out, closer to where you
//   actually look. A real eye jump (a sample well outside the fixation, twice
//   in a row) starts a new fixation at once.
//
// * HeadNudge - fine control with small head tilts. Tilting the head up /
//   down / left / right past a dead zone moves the cursor that way, faster the
//   further you tilt (like a joystick). The neutral pose adapts slowly, so the
//   way you happen to sit never moves the cursor; the offset stays while you
//   keep looking at the same place and is dropped when the eyes jump elsewhere.

export const MOTION_STYLES = Object.freeze({
  snappy: { omega: 30 },   // settles in ~0.15 s
  balanced: { omega: 17 }, // ~0.3 s
  glide: { omega: 9.5 },   // ~0.5 s, very smooth
});

const HOLD_TAU = 0.6;      // s: how far back the fixation average reaches

export class CursorMotion {
  constructor() {
    this.pos = null;          // drawn position
    this.vel = { x: 0, y: 0 };
    this.goal = null;         // where the cursor is heading
    this.style = 'balanced';
    this.hold = true;
    this.radius = 40;         // fixation radius (px)
    this.fix = null;          // {cx, cy, n, t} average of the current fixation
    this.pending = null;      // a sample outside the fixation, waiting for the next one
    this.locked = false;      // goal held still (the head is nudging the cursor)
  }

  configure({ style, hold, radius } = {}) {
    if (style && MOTION_STYLES[style]) this.style = style;
    if (typeof hold === 'boolean') this.hold = hold;
    if (Number.isFinite(radius)) this.radius = Math.max(15, Math.min(120, radius));
  }

  /** Jump straight to a point (e.g. when tracking (re)starts). */
  reset(p) {
    this.pos = p ? { x: p.x, y: p.y } : null;
    this.goal = p ? { x: p.x, y: p.y } : null;
    this.vel = { x: 0, y: 0 };
    this.fix = p ? { cx: p.x, cy: p.y, n: 1, t: null } : null;
    this.pending = null;
  }

  /** Hold the goal where it is (samples still tell when the eyes jump away). */
  lock(on) {
    this.locked = !!on;
  }

  /**
   * A new gaze estimate (page coordinates) at time `t` (seconds). Returns true
   * when it starts a new fixation (the eyes jumped somewhere else).
   */
  addSample(p, t = performance.now() / 1000) {
    if (!this.pos) {
      this.reset(p);
      this.fix.t = t;
      return true;
    }
    const f = this.fix;
    const d = Math.hypot(p.x - f.cx, p.y - f.cy);
    // While locked the gaze estimate itself may shift a little (the head is
    // tilting), so only a big move counts as looking somewhere else.
    const r = this.locked ? 4 * this.radius : this.radius;
    let jumped = false;
    if (d <= r) {
      // Still the same fixation: refine its average (a running average at
      // first, then a moving average over the last HOLD_TAU seconds).
      this.pending = null;
      const dt = f.t === null ? 0 : Math.max(0, t - f.t);
      f.n += 1;
      const k = Math.max(1 / f.n, 1 - Math.exp(-dt / HOLD_TAU));
      f.cx += (p.x - f.cx) * k;
      f.cy += (p.y - f.cy) * k;
      f.t = t;
    } else if (!this.pending) {
      // A single stray sample: wait for the next one before moving.
      this.pending = { x: p.x, y: p.y };
      return false;
    } else {
      // The eyes moved: a new fixation, seeded with the last two samples.
      const q = this.pending;
      this.fix = { cx: (p.x + q.x) / 2, cy: (p.y + q.y) / 2, n: 2, t };
      this.pending = null;
      this.locked = false;
      jumped = true;
    }
    if (this.locked) return jumped;
    this.goal = this.hold ? { x: this.fix.cx, y: this.fix.cy } : { x: p.x, y: p.y };
    return jumped;
  }

  /** Advance the animation by dt seconds; returns the position to draw. */
  step(dt) {
    if (!this.pos || !this.goal) return this.pos ? { ...this.pos } : null;
    const w = MOTION_STYLES[this.style].omega;
    // Critically damped spring, integrated in small steps for stability.
    let t = Math.min(dt, 0.1);
    while (t > 0) {
      const h = Math.min(t, 1 / 240);
      const ax = w * w * (this.goal.x - this.pos.x) - 2 * w * this.vel.x;
      const ay = w * w * (this.goal.y - this.pos.y) - 2 * w * this.vel.y;
      this.vel.x += ax * h;
      this.vel.y += ay * h;
      this.pos.x += this.vel.x * h;
      this.pos.y += this.vel.y * h;
      t -= h;
    }
    if (Math.hypot(this.goal.x - this.pos.x, this.goal.y - this.pos.y) < 0.3 && Math.hypot(this.vel.x, this.vel.y) < 2) {
      this.pos = { ...this.goal };
      this.vel = { x: 0, y: 0 };
    }
    return { ...this.pos };
  }
}

export class HeadNudge {
  constructor() {
    this.enabled = false;
    this.deadzone = 5;        // degrees of tilt that do nothing
    this.speed = 60;          // px/s per degree beyond the dead zone
    // Which way of yaw / pitch means right / up for this person. With the face
    // transform's convention yaw grows when the head turns to the person's
    // left, so "right" is negative yaw; the head test can flip either.
    this.signs = { right: -1, up: 1 };
    this.neutral = null;      // {yaw, pitch}
    this.offset = { x: 0, y: 0 };
    this.active = false;      // currently tilted past the dead zone
    this.maxOffset = 400;
  }

  configure({ enabled, deadzone, speed, signs, maxOffset } = {}) {
    if (typeof enabled === 'boolean') this.enabled = enabled;
    if (Number.isFinite(deadzone)) this.deadzone = Math.max(1, Math.min(15, deadzone));
    if (Number.isFinite(speed)) this.speed = Math.max(10, Math.min(400, speed));
    if (signs) this.signs = { right: signs.right < 0 ? -1 : 1, up: signs.up < 0 ? -1 : 1 };
    if (Number.isFinite(maxOffset)) this.maxOffset = maxOffset;
    if (!this.enabled) this.clear();
  }

  clear() {
    this.offset = { x: 0, y: 0 };
    this.active = false;
  }

  /** Tilt beyond the dead zone (degrees) for a head pose. */
  tilt(yaw, pitch) {
    if (!this.neutral) return { x: 0, y: 0 };
    const beyond = (v) => Math.sign(v) * Math.max(0, Math.abs(v) - this.deadzone);
    return { x: beyond(yaw - this.neutral.yaw), y: beyond(pitch - this.neutral.pitch) };
  }

  /**
   * One frame of head pose ([yaw, pitch, ...] in degrees, or null when the
   * face is not seen) after `dt` seconds. `jumped`: the eyes just moved far
   * (the offset belonged to the previous place). Returns the offset (px) to
   * add to the gaze point.
   */
  update(head, dt, jumped = false) {
    if (jumped) this.offset = { x: 0, y: 0 };
    if (!this.enabled || !head) {
      this.active = false;
      return { ...this.offset };
    }
    const [yaw, pitch] = head;
    if (!this.neutral) this.neutral = { yaw, pitch };
    const e = this.tilt(yaw, pitch);
    this.active = e.x !== 0 || e.y !== 0;
    if (this.active) {
      const vx = this.signs.right * e.x * this.speed;
      const vy = -this.signs.up * e.y * this.speed;      // screen y grows downwards
      this.offset.x += vx * dt;
      this.offset.y += vy * dt;
      const m = Math.hypot(this.offset.x, this.offset.y);
      if (m > this.maxOffset) {
        this.offset.x *= this.maxOffset / m;
        this.offset.y *= this.maxOffset / m;
      }
    } else {
      // The way you sit becomes the new neutral (slowly).
      const k = 1 - Math.exp(-dt / 4);
      this.neutral.yaw += (yaw - this.neutral.yaw) * k;
      this.neutral.pitch += (pitch - this.neutral.pitch) * k;
    }
    return { ...this.offset };
  }
}
