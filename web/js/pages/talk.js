// Talk: speak common phrases (grouped by topic), or type with your eyes (with
// word + next-word predictions) and have the computer read it aloud in a natural
// voice.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { speak, listVoices, setVoice, currentVoice } from '../speech.js';
import { WORDS, STARTERS, NEXT } from '../data/words.js';
import { PHRASE_GROUPS } from '../data/phrases.js';

const ROWS = ['QWERTYUIOP', 'ASDFGHJKL\'', 'ZXCVBNM,.?'];

// Kept between visits to the page.
let text = '';
let mode = 'phrases';
let group = 0;

function sentenceStart(t) {
  return /^\s*$/.test(t) || /[.?!]\s*$/.test(t);
}
function capitalise(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}
function fixI(word) {
  return word === 'i' || word.startsWith("i'") ? capitalise(word) : word;
}

export function predict(current) {
  const m = current.match(/([A-Za-z']+)$/);
  const prefix = m ? m[1].toLowerCase() : '';
  const before = current.slice(0, current.length - prefix.length);
  const atStart = sentenceStart(before);
  if (!prefix) {
    if (!current.trim()) return STARTERS;
    // No partial word: suggest likely *next* words from the previous word.
    const lastWord = (before.trim().match(/([A-Za-z']+)[\s.,?!]*$/) || [, ''])[1].toLowerCase();
    const next = (NEXT[lastWord] || ['the', 'to', 'you', 'please', 'and', 'a', 'is', 'it']).slice(0, 4);
    return atStart ? next.map(capitalise) : next.map(fixI);
  }
  const matches = WORDS.filter((w) => w.startsWith(prefix) && w !== prefix).slice(0, 4);
  return matches.map((w) => (atStart ? capitalise(w) : fixI(w)));
}

function shortVoiceName(v) {
  if (!v) return 'Default';
  return (v.name || 'Voice').replace(/\s*\(.*?\)\s*/g, '').trim() || v.name;
}

export default {
  title: 'Talk',
  render(el) {
    const display = h('div', { class: 'text-display', 'aria-live': 'polite' });
    const speakBtn = h('button', { class: 'btn primary', type: 'button', html: `${icon('speaker')}<span>Speak</span>` });
    speakBtn.addEventListener('click', () => speak(text));
    const phrasesTab = h('button', { class: 'btn', type: 'button', html: `${icon('grid')}<span>Phrases</span>` });
    const keyboardTab = h('button', { class: 'btn', type: 'button', html: `${icon('keyboard')}<span>Keyboard</span>` });
    const voiceBtn = h('button', { class: 'btn', type: 'button', title: 'Change the voice' });
    const refreshVoiceLabel = () => { voiceBtn.innerHTML = `${icon('speaker')}<span>Voice: ${shortVoiceName(currentVoice())}</span>`; };
    refreshVoiceLabel();
    voiceBtn.addEventListener('click', () => {
      const vs = listVoices();
      if (!vs.length) return;
      const cur = currentVoice();
      const i = Math.max(0, vs.findIndex((v) => cur && v.voiceURI === cur.voiceURI));
      const nextV = vs[(i + 1) % vs.length];
      setVoice(nextV.voiceURI);
      refreshVoiceLabel();
      speak('Hello, this is my voice.');
    });
    const body = h('div');

    const renderDisplay = () => {
      display.innerHTML = '';
      if (!text) {
        display.append(h('span', { class: 'placeholder' },
          mode === 'keyboard' ? 'Look at letters and blink twice to type…' : 'Pick a phrase to say it out loud…'));
      } else {
        const shown = text.length > 70 ? `…${text.slice(-70)}` : text;
        display.append(document.createTextNode(shown));
      }
      if (mode === 'keyboard') display.append(h('span', { class: 'caret' }));
    };

    const renderPhrases = () => {
      body.innerHTML = '';
      // Topic chips.
      const cats = h('div', { class: 'phrase-cats' },
        PHRASE_GROUPS.map((g, i) => {
          const chip = h('button', { class: `btn chip ${i === group ? 'active' : ''}`, type: 'button' },
            h('span', { class: 'emoji' }, g.icon || '💬'), h('span', {}, g.name));
          chip.addEventListener('click', () => { group = i; renderPhrases(); });
          return chip;
        }));
      const grid = h('div', { class: 'phrases' },
        PHRASE_GROUPS[group].phrases.map(([emoji, phrase, cls]) => {
          const b = h('button', { class: `phrase ${cls || ''}`, type: 'button' },
            h('span', { class: 'emoji' }, emoji), h('span', {}, phrase));
          b.addEventListener('click', () => { text = phrase; renderDisplay(); speak(phrase); });
          return b;
        }));
      body.append(cats, grid);
    };

    const suggestionsRow = h('div', { class: 'suggestions' });
    const renderSuggestions = () => {
      suggestionsRow.innerHTML = '';
      const words = predict(text);
      for (let i = 0; i < 4; i++) {
        const w = words[i];
        const b = h('button', { class: 'suggestion', type: 'button', disabled: !w }, w || '');
        if (w) b.addEventListener('click', () => {
          text = text.replace(/[A-Za-z']*$/, '') + w + ' ';
          update();
        });
        suggestionsRow.append(b);
      }
    };

    const update = () => {
      renderDisplay();
      if (mode === 'keyboard') renderSuggestions();
    };

    const type = (ch) => {
      if (ch === ',' || ch === '.' || ch === '?') text = `${text.replace(/\s+$/, '')}${ch} `;
      else if (ch === "'") text += "'";
      else text += sentenceStart(text) ? ch.toUpperCase() : ch.toLowerCase();
      update();
    };

    const deleteWord = () => {
      text = text.replace(/\s*[^\s]*\s*$/, '');  // drop trailing spaces + last token
      update();
    };

    const key = (label, onClick, cls = '', html = null) => {
      const b = h('button', { class: `key ${cls}`, type: 'button', 'aria-label': typeof label === 'string' ? label : undefined });
      if (html) b.innerHTML = html;
      else b.textContent = label;
      b.addEventListener('click', onClick);
      return b;
    };

    const renderKeyboard = () => {
      body.innerHTML = '';
      const rows = ROWS.map((row) => h('div', { class: 'key-row' },
        [...row].map((ch) => key(ch, () => type(ch)))));
      const special = h('div', { class: 'key-row special' },
        key('Space', () => { text += ' '; update(); }, 'small-text', `${icon('space')}<span>Space</span>`),
        key('Delete', () => { text = text.slice(0, -1); update(); }, 'small-text', `${icon('del')}<span>Delete</span>`),
        key('Delete word', deleteWord, 'small-text', `${icon('del')}<span>Word</span>`),
        key('Clear', () => { text = ''; update(); }, 'small-text', `${icon('trash')}<span>Clear</span>`),
        key('Speak', () => speak(text), 'speak small-text', `${icon('speaker')}<span>Speak</span>`));
      body.append(h('div', { class: 'keyboard' }, suggestionsRow, ...rows, special));
      renderSuggestions();
    };

    const setMode = (m) => {
      mode = m;
      phrasesTab.classList.toggle('active', m === 'phrases');
      keyboardTab.classList.toggle('active', m === 'keyboard');
      if (m === 'phrases') renderPhrases();
      else renderKeyboard();
      renderDisplay();
    };
    phrasesTab.addEventListener('click', () => setMode('phrases'));
    keyboardTab.addEventListener('click', () => setMode('keyboard'));

    el.append(h('div', { class: 'talk' },
      h('div', { class: 'talk-top' }, display,
        h('div', { class: 'tabs' }, phrasesTab, keyboardTab, voiceBtn, speakBtn)),
      body));
    setMode(mode);
  },
};
