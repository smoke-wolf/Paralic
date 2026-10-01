// Text to speech via the browser's Web Speech API.
//
// Naturalness comes from the *voice* that is chosen: modern systems ship neural
// voices (macOS "Premium"/"Enhanced"/Siri voices, Windows "Natural" voices) that
// sound far better than the default. We score the installed voices and pick the
// best one in the user's language, preferring local voices (no network, private,
// reliable). The choice can be overridden and is remembered.

import { getSettings } from './settings.js';
import { toast } from './dom.js';

const RATES = { slow: 0.82, normal: 1.0, fast: 1.22 };
const STORAGE_KEY = 'paralic.voiceURI';

// Known high-quality voice names (macOS/iOS and common natural voices).
const GOOD_NAMES = ['ava', 'samantha', 'allison', 'zoe', 'serena', 'tessa', 'karen',
  'moira', 'daniel', 'evan', 'joelle', 'nathan', 'noelle', 'siri', 'aaron', 'nicky', 'flo'];

let voices = [];
let chosen = null;

export function canSpeak() {
  return 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
}

function uiLang() {
  return (navigator.language || 'en').slice(0, 2).toLowerCase();
}

/** Higher = more natural / more appropriate. */
function score(v) {
  const name = (v.name || '').toLowerCase();
  const uri = (v.voiceURI || '').toLowerCase();
  const lang = (v.lang || '').toLowerCase();
  let s = 0;
  if (lang.startsWith(uiLang())) s += 6;
  else if (lang.startsWith('en')) s += 3;
  if (/(premium|enhanced|neural|natural)/.test(name + ' ' + uri)) s += 9;
  if (name.includes('siri') || uri.includes('siri')) s += 7;
  if (GOOD_NAMES.some((n) => name.includes(n))) s += 4;
  if (v.localService) s += 2;             // local: private, no network, reliable
  if (lang === 'en-us' || lang === 'en-gb') s += 1;
  if (/(compact|eloquence|novelty|whisper|zarvox|albert|bad|bells)/.test(name)) s -= 6;
  return s;
}

function rankedVoices() {
  return voices.slice().sort((a, b) => score(b) - score(a));
}

function pickVoice() {
  let saved = null;
  try { saved = localStorage.getItem(STORAGE_KEY); } catch { /* private mode */ }
  if (saved) {
    const v = voices.find((x) => x.voiceURI === saved);
    if (v) return v;
  }
  return rankedVoices()[0] || null;
}

function loadVoices() {
  try { voices = window.speechSynthesis.getVoices() || []; } catch { voices = []; }
  if (voices.length && (!chosen || !voices.includes(chosen))) chosen = pickVoice();
}

if (canSpeak()) {
  loadVoices();
  try { window.speechSynthesis.addEventListener('voiceschanged', loadVoices); } catch { /* ignore */ }
}

/** Best voices first (for a chooser). */
export function listVoices() {
  if (!voices.length) loadVoices();
  return rankedVoices();
}

/** The voice currently used. */
export function currentVoice() {
  if (!chosen) loadVoices();
  return chosen;
}

/** Override the voice (remembered). Pass a voiceURI. */
export function setVoice(uri) {
  const v = voices.find((x) => x.voiceURI === uri);
  if (v) {
    chosen = v;
    try { localStorage.setItem(STORAGE_KEY, uri); } catch { /* ignore */ }
  }
  return chosen;
}

/**
 * Say something (interrupting whatever is being said). Resolves when the
 * sentence has been spoken, could not be spoken, or was interrupted — so a
 * caller can wait for an instruction to finish before moving on.
 */
export function speak(text) {
  const clean = String(text || '').trim();
  if (!clean) return Promise.resolve();
  if (!canSpeak()) {
    toast('Speech is not supported in this browser', 'warn');
    return Promise.resolve();
  }
  if (!chosen) loadVoices();
  return new Promise((resolve) => {
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(clean);
      u.rate = RATES[getSettings().speechRate] || 1;
      u.pitch = 1.0;
      if (chosen) { u.voice = chosen; u.lang = chosen.lang; }
      // If speech never starts (no voices / no audio device), resolve anyway so
      // a caller waiting on an instruction does not hang.
      let started = false;
      u.onstart = () => { started = true; };
      setTimeout(() => { if (!started) resolve(); }, 1200);
      u.onend = () => resolve();
      u.onerror = (e) => {
        if (e.error === 'not-allowed') toast('Click anywhere once to allow speech in this browser', 'warn', 5000);
        resolve();
      };
      window.speechSynthesis.speak(u);
    } catch {
      toast('Could not speak', 'warn');
      resolve();
    }
  });
}

/** Stop speaking now. */
export function stopSpeaking() {
  try {
    if (canSpeak()) window.speechSynthesis.cancel();
  } catch {
    /* ignore */
  }
}
