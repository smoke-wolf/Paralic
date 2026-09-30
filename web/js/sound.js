// Tiny feedback sounds made with Web Audio (no audio files needed).

import { getSettings } from './settings.js';

let ctx = null;

/** Must be called from a real user gesture once (browsers block audio otherwise). */
export function unlockAudio() {
  try {
    ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === 'suspended') ctx.resume();
  } catch {
    ctx = null;
  }
}

function tone(freq, duration, { type = 'sine', gain = 0.06, endFreq = null, delay = 0 } = {}) {
  if (!ctx || !getSettings().sounds) return;
  const t0 = ctx.currentTime + delay;
  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, t0);
  if (endFreq) osc.frequency.exponentialRampToValueAtTime(endFreq, t0 + duration);
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(gain, t0 + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
  osc.connect(g).connect(ctx.destination);
  osc.start(t0);
  osc.stop(t0 + duration + 0.02);
}

export const sounds = {
  firstBlink: () => tone(880, 0.07, { gain: 0.035 }),
  click: () => { tone(740, 0.06, { gain: 0.07 }); tone(1180, 0.08, { gain: 0.06, delay: 0.05 }); },
  miss: () => tone(300, 0.12, { type: 'triangle', gain: 0.05, endFreq: 200 }),
  point: () => tone(660, 0.05, { gain: 0.03 }),
  success: () => { tone(660, 0.1, { gain: 0.05 }); tone(880, 0.1, { gain: 0.05, delay: 0.09 }); tone(1320, 0.16, { gain: 0.05, delay: 0.18 }); },
  pop: () => tone(520, 0.1, { type: 'triangle', gain: 0.08, endFreq: 1400 }),
};
