// Eyes or hands: how the person controls Paralic in this tab.
//
// main.js sets `body.hand-mode` once the mode is chosen. Text that names a
// gesture follows it:
//   - markup rendered before the mode is known (or that should follow a later
//     switch) uses both wordings - sayHtml('blink twice', 'pinch') - and CSS
//     shows the right one (.say-eyes / .say-hands);
//   - text made at the moment it is shown (toasts, speech) uses say().
// Whole elements can be marked .eyes-only or .hands-only.

import { esc } from './dom.js';

export const handMode = () => document.body.classList.contains('hand-mode');

/** The wording for the current mode. */
export function say(eyes, hands) {
  return handMode() ? hands : eyes;
}

/** Both wordings as HTML; CSS shows the one for the current mode. */
export function sayHtml(eyes, hands) {
  return `<span class="say-eyes">${esc(eyes)}</span><span class="say-hands">${esc(hands)}</span>`;
}

/** The click gesture, e.g. "Blink twice" / "Pinch" (capitalised when `cap`). */
export function clickWord(cap = false) {
  const w = say('blink twice', 'pinch');
  return cap ? w[0].toUpperCase() + w.slice(1) : w;
}

/** One line about the saved hand setup (app.state.hand), e.g. "Pointing accuracy ≈ 38 px · pinch tuned". */
export function handSummary(hand) {
  if (!hand) return 'Not set up yet';
  const parts = [];
  if (typeof hand.pointing_error_px === 'number') parts.push(`pointing accuracy ≈ ${Math.round(hand.pointing_error_px)} px`);
  parts.push(hand.pinch_tuned ? 'pinch tuned' : 'standard pinch');
  const line = parts.join(' · ');
  return line[0].toUpperCase() + line.slice(1);
}
