// Text to speech via the browser's Web Speech API.

import { getSettings } from './settings.js';
import { toast } from './dom.js';

const RATES = { slow: 0.8, normal: 1.0, fast: 1.25 };

export function canSpeak() {
  return 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
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
  return new Promise((resolve) => {
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(clean);
      u.rate = RATES[getSettings().speechRate] || 1;
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
