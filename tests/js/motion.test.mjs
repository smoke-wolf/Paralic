// Unit tests for the DOM-free browser modules (run by tests/test_js_units.py:
// node --test tests/js). They have no imports, so they load from source.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

async function load(name) {
  const src = readFileSync(new URL(`../../web/js/${name}`, import.meta.url), 'utf8');
  return import(`data:text/javascript;base64,${Buffer.from(src).toString('base64')}`);
}

const { CursorMotion, HeadNudge } = await load('motion.js');
const { assessPosition } = await load('position.js');

// Deterministic "gaze noise".
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}
function gauss(r) {
  return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
}

function timeToReach(style, fraction) {
  const m = new CursorMotion();
  m.configure({ style, hold: false });
  m.reset({ x: 0, y: 0 });
  m.addSample({ x: 300, y: 0 }, 0);
  m.addSample({ x: 300, y: 0 }, 0.033);
  let t = 0;
  let last = 0;
  while (t < 3) {
    const p = m.step(1 / 60);
    t += 1 / 60;
    assert.ok(p.x <= 300 + 1e-6, 'never overshoots');
    assert.ok(p.x >= last - 1e-6, 'moves one way only');
    last = p.x;
    if (p.x >= fraction * 300) return t;
  }
  return Infinity;
}

test('the cursor glides to a new place without overshooting', () => {
  const glide = timeToReach('glide', 0.9);
  const balanced = timeToReach('balanced', 0.9);
  const snappy = timeToReach('snappy', 0.9);
  assert.ok(snappy < balanced && balanced < glide, `${snappy} < ${balanced} < ${glide}`);
  assert.ok(balanced < 0.35 && glide < 0.6 && snappy < 0.2);
});

test('holding still averages the noise of a fixation', () => {
  const r = rng(1);
  const m = new CursorMotion();
  m.configure({ hold: true, radius: 45 });
  m.reset({ x: 500, y: 400 });
  const samples = [];
  const goals = [];
  for (let i = 0; i < 90; i++) {
    const p = { x: 500 + 12 * gauss(r), y: 400 + 12 * gauss(r) };
    m.addSample(p, i / 30);
    if (i >= 30) {
      samples.push(Math.hypot(p.x - 500, p.y - 400));
      goals.push(Math.hypot(m.goal.x - 500, m.goal.y - 400));
    }
  }
  const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
  assert.ok(mean(goals) < 0.4 * mean(samples), `${mean(goals)} vs ${mean(samples)}`);
  // Without holding, the goal is simply the latest sample.
  m.configure({ hold: false });
  m.addSample({ x: 510, y: 395 }, 3.1);
  assert.deepEqual(m.goal, { x: 510, y: 395 });
});

test('a stray sample waits; a real jump starts a new fixation at once', () => {
  const m = new CursorMotion();
  m.configure({ hold: true, radius: 40 });
  m.reset({ x: 200, y: 200 });
  for (let i = 0; i < 10; i++) m.addSample({ x: 200, y: 200 }, i / 30);
  assert.equal(m.addSample({ x: 300, y: 200 }, 0.4), false);       // one stray sample
  assert.equal(m.addSample({ x: 202, y: 200 }, 0.43), false);      // back: nothing happened
  assert.ok(Math.abs(m.goal.x - 200) < 1);
  assert.equal(m.addSample({ x: 800, y: 600 }, 0.5), false);
  assert.equal(m.addSample({ x: 804, y: 600 }, 0.53), true);       // two in a row: the eyes moved
  assert.ok(Math.abs(m.goal.x - 802) < 1 && Math.abs(m.goal.y - 600) < 1);
});

test('a slow drift within a fixation is still followed', () => {
  const m = new CursorMotion();
  m.configure({ hold: true, radius: 40 });
  m.reset({ x: 500, y: 300 });
  for (let i = 0; i <= 90; i++) m.addSample({ x: 500 + i / 3, y: 300 }, i / 30);  // 30 px in 3 s
  assert.ok(m.goal.x > 520, `goal ${m.goal.x}`);
});

test('while locked the goal stays put until the eyes really look elsewhere', () => {
  const m = new CursorMotion();
  m.configure({ hold: true, radius: 40 });
  m.reset({ x: 400, y: 400 });
  m.addSample({ x: 400, y: 400 }, 0);
  m.lock(true);
  for (let i = 1; i < 10; i++) assert.equal(m.addSample({ x: 470, y: 400 }, i / 30), false);  // head tilt shifts the estimate
  assert.deepEqual(m.goal, { x: 400, y: 400 });
  assert.equal(m.addSample({ x: 1200, y: 700 }, 0.4), false);
  assert.equal(m.addSample({ x: 1200, y: 700 }, 0.43), true);
  assert.equal(m.locked, false);
  assert.deepEqual(m.goal, { x: 1200, y: 700 });
});

test('the head nudge moves the cursor past a dead zone, like a joystick', () => {
  const n = new HeadNudge();
  assert.deepEqual(n.update([10, 0, 0], 0.1), { x: 0, y: 0 });       // off by default
  n.configure({ enabled: true, deadzone: 5, speed: 60 });
  n.update([2, -3, 0], 0.033);                                         // the neutral pose
  // Within the dead zone: nothing moves.
  for (let i = 0; i < 30; i++) n.update([5, -1, 0], 0.033);
  assert.deepEqual(n.offset, { x: 0, y: 0 });
  assert.equal(n.active, false);
  // Turning to the person's right (yaw goes down) by 10 degrees for 0.5 s.
  const start = { ...n.neutral };
  for (let i = 0; i < 15; i++) n.update([start.yaw - 10, start.pitch, 0], 1 / 30);
  assert.equal(n.active, true);
  assert.ok(n.offset.x > 120 && n.offset.x < 180 && Math.abs(n.offset.y) < 1, JSON.stringify(n.offset));
  // Tipping the head up moves it up (screen y decreases).
  for (let i = 0; i < 15; i++) n.update([start.yaw, start.pitch + 10, 0], 1 / 30);
  assert.ok(n.offset.y < -120);
  // Back to neutral: the offset stays; a jump of the eyes drops it.
  const kept = { ...n.offset };
  n.update([start.yaw, start.pitch, 0], 1 / 30);
  assert.deepEqual(n.offset, kept);
  assert.deepEqual(n.update([start.yaw, start.pitch, 0], 1 / 30, true), { x: 0, y: 0 });
  // A person whose camera sees it the other way round.
  n.configure({ signs: { right: 1, up: -1 } });
  for (let i = 0; i < 15; i++) n.update([start.yaw + 10, start.pitch, 0], 1 / 30);
  assert.ok(n.offset.x > 120);
  // Never further than maxOffset.
  for (let i = 0; i < 300; i++) n.update([start.yaw + 15, start.pitch, 0], 1 / 30);
  assert.ok(Math.hypot(n.offset.x, n.offset.y) <= n.maxOffset + 1e-6);
});

test('the neutral head pose follows how the person sits', () => {
  const n = new HeadNudge();
  n.configure({ enabled: true, deadzone: 5 });
  n.update([0, 0, 0], 0.033);
  for (let i = 0; i < 600; i++) n.update([4, 3, 0], 1 / 30);   // sitting a bit differently, within the dead zone
  assert.ok(Math.abs(n.neutral.yaw - 4) < 0.5 && Math.abs(n.neutral.pitch - 3) < 0.5);
  assert.deepEqual(n.offset, { x: 0, y: 0 });
});

test('position check: one hint at a time, most important first', () => {
  assert.equal(assessPosition(null).face, false);
  assert.match(assessPosition({ face: false }).hint, /can’t see your face/);
  const good = { face: true, head: [2, -6, 0], dist: 60, pos: [1, -5], light: { face: 130, frame: 110, balance: 0.05 } };
  assert.deepEqual(assessPosition(good), { face: true, place: '', light: '', ok: true, hint: '' });
  assert.match(assessPosition({ ...good, light: { face: 40, frame: 40, balance: 0 } }).hint, /dark/);
  assert.match(assessPosition({ ...good, light: { face: 90, frame: 170, balance: 0 } }).hint, /behind you/);
  assert.match(assessPosition({ ...good, light: { face: 120, frame: 110, balance: -0.4 } }).hint, /one side/);
  assert.match(assessPosition({ ...good, dist: 25 }).hint, /lean back/);
  assert.match(assessPosition({ ...good, dist: 120 }).hint, /closer/);
  // Camera x grows towards the person's left: too far that way means "move right".
  assert.match(assessPosition({ ...good, pos: [20, -5] }).hint, /to your right/);
  assert.match(assessPosition({ ...good, pos: [-20, -5] }).hint, /to your left/);
  // Position comes before light.
  const both = assessPosition({ ...good, dist: 25, light: { face: 40, frame: 40, balance: 0 } });
  assert.match(both.hint, /lean back/);
  assert.ok(both.light && both.place && !both.ok);
});

test('position check: back to where the calibration was done', () => {
  const ref = { x: 1, y: -5, dist: 58, yaw: 2, pitch: -6 };
  const m = { face: true, head: [2, -6, 0], dist: 58, pos: [1, -5] };
  assert.equal(assessPosition(m, ref).ok, true);
  assert.equal(assessPosition({ ...m, dist: 61, pos: [3, -3] }, ref).ok, true);     // close enough
  assert.match(assessPosition({ ...m, dist: 68 }, ref).hint, /closer/);
  assert.match(assessPosition({ ...m, dist: 50 }, ref).hint, /Lean back/);
  assert.match(assessPosition({ ...m, pos: [6, -5] }, ref).hint, /to your right/);
  assert.match(assessPosition({ ...m, pos: [-4, -5] }, ref).hint, /to your left/);
  assert.match(assessPosition({ ...m, pos: [1, 0] }, ref).hint, /lower/);
  assert.match(assessPosition({ ...m, pos: [1, -10] }, ref).hint, /higher/);
  assert.match(assessPosition({ ...m, head: [16, -6, 0] }, ref).hint, /towards the screen/);
});
