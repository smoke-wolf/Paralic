// User settings, persisted in localStorage.

const KEY = 'paralic.settings.v1';

export const DEFAULTS = Object.freeze({
  smoothing: 'medium',         // low | medium | high           (server: One Euro filter)
  blinkSensitivity: 'normal',  // low | normal | high           (server: blink threshold)
  doubleBlink: 'normal',       // fast | normal | relaxed       (server: max pause between blinks)
  snap: 'normal',              // off | normal | strong         (magnetic buttons)
  scrollSpeed: 'normal',       // slow | normal | fast
  cursorSize: 'normal',        // small | normal | large
  sounds: true,
  showCamera: true,
  driftCorrection: true,       // learn small offsets from your clicks
  speechRate: 'normal',        // slow | normal | fast
});

const listeners = new Set();
let current = load();

function load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    /* storage unavailable */
  }
  return { ...DEFAULTS };
}

export function getSettings() {
  return current;
}

export function updateSettings(patch) {
  current = { ...current, ...patch };
  try {
    localStorage.setItem(KEY, JSON.stringify(current));
  } catch {
    /* ignore */
  }
  for (const fn of listeners) fn(current, patch);
}

export function onSettingsChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** The subset of settings the Python server needs. */
export function serverSettings(s = current) {
  return { smoothing: s.smoothing, blink_sensitivity: s.blinkSensitivity, double_blink: s.doubleBlink };
}
