// User settings, persisted in localStorage.

const KEY = 'paralic.settings.v1';

// "auto" / "personal" use the values learned for the current person (from
// their calibration, blink recording and A/B experiments); the named presets
// override them.
export const DEFAULTS = Object.freeze({
  smoothing: 'auto',           // auto | low | medium | high              (server: One Euro filter)
  blinkSensitivity: 'personal',// personal | low | normal | high          (server: blink threshold)
  doubleBlink: 'personal',     // personal | fast | normal | relaxed      (server: max pause between blinks)
  snap: 'auto',                // auto | off | normal | strong            (magnetic buttons)
  scrollSpeed: 'normal',       // slow | normal | fast
  cursorSize: 'normal',        // small | normal | large
  sounds: true,
  showCamera: true,
  driftCorrection: true,       // nudge the cursor to line up with what you click
  learning: true,              // turn practice hits and clicks into fine-tuning data
  speechRate: 'normal',        // slow | normal | fast
});

const listeners = new Set();
let current = load();

// Values that were the defaults before personalisation existed: people who
// never changed them get the new personal / auto defaults.
const OLD_DEFAULTS = { smoothing: 'medium', blinkSensitivity: 'normal', doubleBlink: 'normal', snap: 'normal' };

function load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const saved = JSON.parse(raw);
      if (!saved._v) {
        for (const [key, old] of Object.entries(OLD_DEFAULTS)) if (saved[key] === old) delete saved[key];
      }
      return { ...DEFAULTS, ...saved, _v: 2 };
    }
  } catch {
    /* storage unavailable */
  }
  return { ...DEFAULTS, _v: 2 };
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
  return { smoothing: s.smoothing, blink_sensitivity: s.blinkSensitivity, double_blink: s.doubleBlink,
    learning: !!s.learning };
}
