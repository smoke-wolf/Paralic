// One recording: the transport bar and the panels, all following the player.
//
// The current frame's details (landmarks, the networks' internals) come from
// the server one request at a time, always for the newest frame, so playing at
// 4x skips what it cannot show instead of falling behind. Plain frame fields
// come in windows of frames for the screen map's trail and the frame facts.

import { FrameWindows, Latest, api } from './api.js';
import { CalibrationPanel } from './calib.js';
import { CameraPanel } from './camera.js';
import { ConfigPanel } from './config.js';
import { EventsPanel } from './events.js';
import { FactsPanel } from './facts.js';
import { HoodPanel } from './hood.js';
import { rating, recordingStatus } from './list.js';
import { Player, SPEEDS, Transport } from './player.js';
import { ScreenMap } from './screenmap.js';
import { SignalsPanel } from './signals.js';
import { bytes, clock, count, debounce, duration, h, isNum, num } from './util.js';

const SECTIONS = {
  Camera: 'sec-camera', Signals: 'sec-signals', 'Under the hood': 'sec-hood', Calibration: 'sec-calibration',
  Events: 'sec-events', Config: 'sec-config',
};
const PLAYING_DETAIL_MS = 45;     // while playing, at most ~20 frame details a second

export class RecordingView {
  constructor(root, id) {
    this.root = root;
    this.id = id;
    this.closed = false;
  }

  async open(n) {
    this.root.replaceChildren(h('div', { class: 'loading' }, `Opening ${this.id}…`));
    let info;
    let index;
    try {
      [info, index] = await Promise.all([api.info(this.id), api.index(this.id)]);
    } catch (err) {
      this.root.replaceChildren(h('div', { class: 'list-page' }, h('p', { class: 'error' }, `Could not open ${this.id}: ${err.message}`),
        h('a', { href: '#/' }, '← Recordings')));
      return;
    }
    if (this.closed) return;
    this.info = info;
    const player = new Player(index);
    this.player = player;
    this.frames = new FrameWindows(this.id, player.length);
    const transport = new Transport(player, { mode: info.mode, onSection: (name) => this.scrollTo(name) });
    this.transport = transport;
    this.scrubber = transport.scrubber;
    this.camera = new CameraPanel(this);
    this.screen = new ScreenMap(this);
    this.facts = new FactsPanel(this);
    this.signals = new SignalsPanel(this);
    this.hood = new HoodPanel(this);
    this.calib = new CalibrationPanel(this);
    this.events = new EventsPanel(this);
    this.config = new ConfigPanel(this);
    document.title = `${info.person || 'Recording'} · ${info.mode} · Paralic Inspector`;
    this.root.replaceChildren(h('div', { class: 'rec' },
      this._header(info),
      transport.el,
      h('main', { class: 'rec-grid' },
        this.camera.el, this.screen.el, this.facts.el,
        this.signals.el,
        this.hood.el,
        this.calib.el, this.events.el,
        this.config.el)));
    // Transport buttons should not keep the focus, so Space keeps meaning play / pause.
    transport.el.addEventListener('mousedown', (e) => {
      if (e.target.closest('button')) e.preventDefault();
    });

    this.detail = null;
    this.lastDetailAt = 0;
    this.detailLoader = new Latest((k) => api.detail(this.id, k), (k, d) => {
      this.detail = d;
      this.camera.onDetail(d);
      this.screen.onDetail(d);
      this.facts.onDetail(d);
      this.hood.onDetail(d);
    });
    player.on('frame', () => this._onFrame());
    player.on('time', () => {
      this.events.onTime();
      this._schedule();
    });
    player.on('state', () => {
      if (!player.playing) {
        this.detailLoader.want(player.n);
        this._saveHash();
      }
    });
    this.frames.listeners.add(() => {
      this.screen.draw();
      this.facts.render();
    });
    this._loadTimeline();
    this.events.load().then((evs) => this.signals.setEvents(evs));
    this.calib.load();
    this.config.load();
    this.onKey = (e) => this._key(e);
    document.addEventListener('keydown', this.onKey);
    this.saveHash = debounce(() => this._saveHash(), 250);
    player.go(isNum(n) ? n : 0);
    this._onFrame();
    this.scrubber.setView(...this.signals.view);
  }

  _header(info) {
    const acc = info.accuracy;
    const r = acc && isNum(acc.mean_error_px) ? rating(acc.mean_error_px, info.screen) : null;
    const facts = [
      ['started', info.started ? new Date(info.started).toLocaleString() : '—'],
      ['duration', duration(info.duration_s)],
      ['frames', count(info.frames)],
      ['video', `${count(info.video_frames)} images${info.meta && info.meta.video ? ` (every ${info.meta.video.every})` : ''}`],
      ['size', bytes(info.bytes)],
      ['screen', `${info.screen.w}×${info.screen.h}`],
      ['models', count(info.models.length)],
      ['accuracy', acc ? `${num(acc.mean_error_px, 1)} px${r ? ` (${r.label})` : ''}` : '—'],
      ['glasses seen', info.glasses === true ? 'yes' : info.glasses === false ? 'no' : '—'],
      ['status', recordingStatus(info)],
    ];
    if (info.dropped) facts.push(['dropped', `${count(info.dropped)} frames (the disk could not keep up)`]);
    const stopped = info.video_stopped;
    if (stopped && stopped.reason) {
      facts.push(['images stopped', `${isNum(stopped.t) ? `at ${clock(stopped.t - info.t0)}: ` : ''}${stopped.reason}`]);
    }
    return h('header', { class: 'rec-head' },
      h('a', { class: 'back', href: '#/' }, '← Recordings'),
      h('div', { class: 'rec-title' },
        h('h1', {}, info.person || info.id, ' ', h('span', { class: `badge mode-${info.mode}` }, info.mode)),
        h('div', { class: 'rec-id mono' }, info.id)),
      h('dl', { class: 'rec-facts' }, facts.map(([k, v]) => h('div', {}, h('dt', {}, k), h('dd', {}, v)))));
  }

  /** Markers and stretches for the scrubber; while the server is still reading the recording, ask again. */
  _loadTimeline() {
    api.timeline(this.id).then((tl) => {
      if (this.closed) return;
      this.timeline = tl;
      this.scrubber.setTimeline(tl);
      this.signals.setTimeline(tl);
      if (!tl.complete) setTimeout(() => this._loadTimeline(), 1500);
    }).catch((err) => console.warn(err));
  }

  _onFrame() {
    const p = this.player;
    this.camera.onFrame();
    this.screen.draw();
    this.facts.onFrame();
    this.calib.onFrame();
    this.frames.get(p.n);
    this._schedule(true);
    if (!p.playing) this.saveHash();
  }

  /** Ask for the current frame's details (rate-limited while playing). */
  _schedule(changed = false) {
    const p = this.player;
    if (!changed && !p.playing) return;
    const now = performance.now();
    if (p.playing && now - this.lastDetailAt < PLAYING_DETAIL_MS) return;
    if (this.detailLoader.busy && p.playing) return;
    if (this.detail && this.detail.n === p.n) return;
    this.lastDetailAt = now;
    this.detailLoader.want(p.n);
  }

  _saveHash() {
    if (this.closed) return;
    const want = `#/rec/${encodeURIComponent(this.id)}?n=${this.player.n}`;
    if (location.hash !== want) history.replaceState(null, '', want);
  }

  _key(e) {
    if (e.target.closest && e.target.closest('input, select, textarea')) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const p = this.player;
    const onButton = e.target.closest && e.target.closest('button, a, summary');
    switch (e.key) {
      case ' ':
        if (onButton) return;
        e.preventDefault();
        p.toggle();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        p.step(e.shiftKey ? -10 : -1);
        break;
      case 'ArrowRight':
        e.preventDefault();
        p.step(e.shiftKey ? 10 : 1);
        break;
      case 'Home':
        e.preventDefault();
        p.pause();
        p.go(0);
        break;
      case 'End':
        e.preventDefault();
        p.pause();
        p.go(p.length - 1);
        break;
      case '[':
      case ']': {
        const k = SPEEDS.indexOf(p.speed) + (e.key === ']' ? 1 : -1);
        p.setSpeed(SPEEDS[Math.max(0, Math.min(SPEEDS.length - 1, k))]);
        break;
      }
      default:
        break;
    }
  }

  scrollTo(name) {
    const el = document.getElementById(SECTIONS[name]);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  destroy() {
    this.closed = true;
    if (this.player) this.player.pause();
    if (this.onKey) document.removeEventListener('keydown', this.onKey);
  }
}
