// Recording a session for the Paralic Inspector (see docs/recording-format.md).
//
// A "● Rec" button in the top bar (and Shift+R) starts and stops recording on
// the server: the configuration, every frame's measurements and what the
// pipeline made of them, commands and events, and camera images go into a
// folder on this computer (data/recordings/). While it records, the button
// pulses red and shows for how long and how much - as the server reports it.
// It can be used with the eyes (or a hand) like any button, but it asks first
// - "Record this session?" / "Stop recording?" - so a stray double blink never
// starts or stops a recording; Shift+R switches at once. With
// `python -m paralic --record` every session records from the start, and the
// button shows that too.

import { h, toast } from './dom.js';
import { say } from './mode.js';

const POLL_MS = 3000;          // how often the size is asked for while recording

let current = null;            // the controls of the current tracker (createTracker can run again)

/** Add the recording controls for app.tracker (not in demo mode: no camera, nothing to record). */
export function attachRecordingUi(app) {
  if (current) current.detach();
  current = app.tracker && !app.tracker.simulated ? new RecordingUi(app.tracker) : null;
}

class RecordingUi {
  constructor(tracker) {
    this.tracker = tracker;
    this.status = { on: false };   // the server's last `recording` message...
    this.since = 0;                // ...and when it arrived
    this.wanted = false;           // turned on here: record the new session after a reconnect too
    this.pending = false;          // asked the server, no answer yet
    this.polledAt = 0;
    addStyle();
    this.info = h('small', { class: 'rec-info' });
    this.button = h('button', { class: 'nav-btn rec-btn', type: 'button', 'aria-pressed': 'false', 'data-no-learn': '' },
      h('span', { class: 'rec-label' }, h('span', { class: 'rec-dot', 'aria-hidden': 'true' }), h('span', {}, 'Rec')),
      this.info);
    this.button.addEventListener('click', () => this.confirm());
    const bar = document.getElementById('topbar') || document.querySelector('.topbar');
    if (bar) bar.insertBefore(this.button, bar.querySelector('#pause-btn'));
    this.offs = [
      tracker.on('recording', (m) => this.onStatus(m)),
      tracker.on('connection', ({ connected }) => this.onConnection(connected)),
    ];
    this.onKey = (e) => {
      if (!e.shiftKey || e.ctrlKey || e.metaKey || e.altKey || e.repeat || e.key.toLowerCase() !== 'r') return;
      if (e.target.closest && e.target.closest('input, textarea, [contenteditable]')) return;
      e.preventDefault();
      this.toggle();
    };
    window.addEventListener('keydown', this.onKey);
    this.timer = setInterval(() => this.tick(), 1000);
    this.render();
  }

  detach() {
    this.offs.forEach((off) => off());
    window.removeEventListener('keydown', this.onKey);
    clearInterval(this.timer);
    this.button.remove();
  }

  toggle() {
    if (!this.pending) this.set(!this.status.on);
  }

  /** Ask before starting or stopping (the button is a gaze target). */
  confirm() {
    if (this.pending) return;
    document.querySelector('.rec-confirm')?.remove();
    const on = !this.status.on;
    const yes = h('button', { class: 'btn primary', type: 'button' }, on ? 'Start recording' : 'Stop recording');
    const no = h('button', { class: 'btn', type: 'button' }, on ? 'Cancel' : 'Keep recording');
    const box = h('div', { class: 'face-banner rec-confirm', role: 'dialog', 'aria-label': 'Recording' },
      h('span', {}, on ? 'Record this session? Camera images are included.' : 'Stop recording?'), yes, no);
    document.body.append(box);
    const close = () => {
      clearTimeout(timer);
      box.remove();
    };
    const timer = setTimeout(close, 15000);
    yes.addEventListener('click', () => {
      close();
      this.set(on);
    });
    no.addEventListener('click', close);
  }

  set(on) {
    this.wanted = on;
    this.pending = true;
    this.render();
    if (!this.tracker.send({ type: 'recording', on })) {
      this.pending = false;
      this.render();
      toast(on ? 'Recording starts as soon as the tracker is connected' : 'Not connected to the tracker', 'warn');
    }
  }

  onStatus(m) {
    const was = !!this.status.on;
    this.pending = false;
    this.status = m;
    this.since = performance.now();
    if (m.on && !was) {
      toast('Recording this session on this computer — camera images included', 'warn', 7000);
    } else if (!m.on && (was || m.ok === false)) {
      this.wanted = false;
      if (m.error) toast(`Recording stopped: ${m.error}`, 'bad', 9000);
      else if (m.id) toast(`Recording saved (${duration(m.seconds)}, ${megabytes(m.bytes)}) in ${m.dir}`, 'ok', 9000);
    }
    if (m.on && m.video_stopped && m.video_stopped !== this.videoNote) {
      toast(`Recording without camera images from now on: ${m.video_stopped}`, 'warn', 9000);
    }
    this.videoNote = m.on ? m.video_stopped : null;
    this.render();
  }

  onConnection(connected) {
    if (!connected) {
      // The server's session - and its recording - ended with the connection.
      this.status = { on: false };
      this.pending = false;
      this.render();
    } else if (this.wanted) {
      this.set(true);              // the new session records into a new folder
    }
  }

  tick() {
    if (!this.status.on) return;
    if (performance.now() - this.polledAt >= POLL_MS) {
      this.polledAt = performance.now();
      this.tracker.send({ type: 'recording' });     // how is it going? (not recorded)
    }
    this.render();
  }

  render() {
    const s = this.status;
    const on = !!s.on;
    this.button.classList.toggle('recording', on);
    this.button.classList.toggle('pending', this.pending);
    this.button.setAttribute('aria-pressed', String(on));
    this.button.setAttribute('aria-label', on ? 'Stop recording' : 'Start recording');
    this.info.textContent = on
      ? `${duration((s.seconds || 0) + (performance.now() - this.since) / 1000)} · ${megabytes(s.bytes)}` : '';
    this.button.title = on
      ? `Recording in ${s.dir} — click or press Shift+R to stop`
      : say('Record this session for the Inspector: camera images, your eyes’ measurements and what Paralic made of them (Shift+R)',
        'Record this session for the Inspector: camera images, your hand’s measurements and what Paralic made of them (Shift+R)');
  }
}

function duration(seconds) {
  const s = Math.max(0, Math.floor(Number(seconds) || 0));
  const two = (v) => String(v).padStart(2, '0');
  const hours = Math.floor(s / 3600);
  return hours ? `${hours}:${two(Math.floor(s / 60) % 60)}:${two(s % 60)}` : `${Math.floor(s / 60)}:${two(s % 60)}`;
}

function megabytes(bytes) {
  const b = Number(bytes) || 0;
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  return b >= 1e7 ? `${Math.round(b / 1e6)} MB` : `${(b / 1e6).toFixed(1)} MB`;
}

const STYLE = `
.rec-btn { flex: 0 0 auto; min-width: clamp(72px, 6.2vw, 116px); background: #2a2140; }
.rec-btn .rec-label { display: flex; align-items: center; gap: 0.45em; }
.rec-btn .rec-dot { width: 0.8em; height: 0.8em; border-radius: 50%; background: var(--faint); }
.rec-btn .rec-info { font-size: 0.72em; font-weight: 600; color: var(--muted); font-variant-numeric: tabular-nums; }
.rec-btn .rec-info:empty { display: none; }
.rec-btn.recording { border-color: var(--danger); background: rgba(251, 113, 133, 0.14); }
.rec-btn.recording .rec-dot { background: var(--danger); animation: rec-pulse 1.4s ease-in-out infinite; }
.rec-btn.pending { opacity: 0.6; }
@keyframes rec-pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(251, 113, 133, 0.6); }
  50% { opacity: 0.35; box-shadow: 0 0 0 0.4em rgba(251, 113, 133, 0); }
}`;

function addStyle() {
  if (!document.getElementById('recording-ui-style')) {
    document.head.append(h('style', { id: 'recording-ui-style' }, STYLE));
  }
}
