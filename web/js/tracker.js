// Camera capture + WebSocket link to the Python eye tracker.
//
// Frames are captured from the webcam, JPEG-encoded and sent to the server as
// binary messages:  uint32 header length | JSON header | JPEG bytes.
// Only one frame is in flight at a time, so the stream automatically runs as
// fast as the server can process it without building up lag.
//
// The server replies with a "frame" message per frame (gaze position, blink
// state, ...) plus events such as "blink", "double_blink", winks
// ("wink_start" / "wink_end" while one eye is held closed, "wink" for a short
// one) and "long_close" (both eyes closed for about a second). It also answers
// commands (calibration, people, personalisation) and pushes the results of
// background jobs such as fine-tuning ("finetune_result").

import { screenInfo, clientToScreen } from './screen-space.js';
import { serverSettings } from './settings.js';

class Emitter extends EventTarget {
  on(type, fn) {
    const handler = (e) => fn(e.detail);
    this.addEventListener(type, handler);
    return () => this.removeEventListener(type, handler);
  }
  emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}

/** The WebSocket command channel shared by the real tracker and demo mode. */
class Channel extends Emitter {
  constructor() {
    super();
    this.ws = null;
    this.running = false;
    this.connected = false;
    this.pending = new Map();
    this.fatal = null;
  }

  connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.ws = ws;
    ws.onopen = () => {
      this.connected = true;
      this.send({ type: 'hello', screen: screenInfo(), settings: serverSettings() });
      this.emit('connection', { connected: true });
    };
    ws.onmessage = (e) => {
      let msg;
      try {
        msg = JSON.parse(e.data);
      } catch {
        return;
      }
      this.onMessage(msg);
    };
    ws.onclose = () => {
      const wasConnected = this.connected;
      this.connected = false;
      this.onDisconnect();
      for (const [, waiters] of this.pending) waiters.forEach((w) => w.reject(new Error('Connection lost')));
      this.pending.clear();
      if (wasConnected) this.emit('connection', { connected: false });
      if (this.running && !this.fatal) setTimeout(() => this.running && this.connect(), 1500);
    };
    ws.onerror = () => {};
  }

  onDisconnect() {}

  onMessage(msg) {
    const waiters = this.pending.get(msg.type);
    if (waiters && waiters.length) {
      const w = waiters.shift();
      clearTimeout(w.timer);
      w.resolve(msg);
    }
    this.emit(msg.type, msg);
  }

  send(cmd) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(cmd));
      return true;
    }
    return false;
  }

  /** Send a command and wait for a reply of type `replyType`. */
  request(cmd, replyType, timeoutMs = 30000) {
    return new Promise((resolve, reject) => {
      const waiters = this.pending.get(replyType) || [];
      const entry = { resolve, reject, timer: null };
      entry.timer = setTimeout(() => {
        const list = this.pending.get(replyType) || [];
        const i = list.indexOf(entry);
        if (i >= 0) list.splice(i, 1);
        reject(new Error(`Timed out waiting for ${replyType}`));
      }, timeoutMs);
      waiters.push(entry);
      this.pending.set(replyType, waiters);
      if (!this.send(cmd)) {
        clearTimeout(entry.timer);
        waiters.pop();
        reject(new Error('Not connected to the eye tracker'));
      }
    });
  }
}

export class EyeTracker extends Channel {
  constructor({ frameWidth = 960, quality = 0.82 } = {}) {
    super();
    this.frameWidth = frameWidth;
    this.quality = quality;
    this.video = null;
    this.stream = null;
    this.frameId = 0;
    this.inFlight = null;
    this.inFlightSince = 0;
    this.sentAt = new Map();
    this.latencyMs = 0;
    this.label = null;
    this.gesture = null;
    this.overlay = false;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    this.simulated = false;
  }

  /** Ask for the camera and start streaming. Throws if the camera is unavailable. */
  async start(videoEl) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser cannot access the camera here. Open the page as http://localhost (not an IP address) in Chrome, Edge or Firefox.');
    }
    this.video = videoEl;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    videoEl.srcObject = this.stream;
    videoEl.muted = true;
    await videoEl.play();
    this.running = true;
    this.connect();
    this.lastVideoCallback = 0;
    this.scheduleCapture();
    // requestVideoFrameCallback may pause when the preview is hidden; poll as a fallback.
    this.pollTimer = setInterval(() => {
      if (performance.now() - this.lastVideoCallback > 120) this.onVideoFrame();
    }, 33);
  }

  stop() {
    this.running = false;
    clearInterval(this.pollTimer);
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.ws) this.ws.close();
  }

  /** Label attached to every captured frame (used during calibration). */
  setLabel(label) {
    this.label = label;
  }

  /** Phase of the wink test attached to every frame: "rest" | "left" | "right" | null. */
  setGesturePhase(phase) {
    this.gesture = phase;
  }

  onDisconnect() {
    this.inFlight = null;
  }

  onMessage(msg) {
    if (msg.type === 'frame') {
      const sent = this.sentAt.get(msg.id);
      if (sent) {
        this.latencyMs = performance.now() - sent;
        this.sentAt.delete(msg.id);
      }
      if (this.sentAt.size > 50) this.sentAt.clear();
      this.inFlight = null;
    } else if (msg.type === 'fatal') {
      this.fatal = msg.error;
    }
    super.onMessage(msg);
  }

  // -- frame capture --------------------------------------------------------
  scheduleCapture() {
    if (!this.running || !this.video.requestVideoFrameCallback) return;
    this.video.requestVideoFrameCallback(() => {
      this.lastVideoCallback = performance.now();
      this.onVideoFrame();
      this.scheduleCapture();
    });
  }

  onVideoFrame() {
    if (!this.running || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    // Backpressure: wait for the previous frame's reply (with a safety timeout).
    if (this.inFlight !== null && performance.now() - this.inFlightSince < 4000) return;
    this.captureAndSend();
  }

  captureAndSend() {
    const video = this.video;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;
    const w = Math.min(this.frameWidth, vw);
    const hgt = Math.round((vh * w) / vw);
    if (this.canvas.width !== w || this.canvas.height !== hgt) {
      this.canvas.width = w;
      this.canvas.height = hgt;
    }
    this.ctx.drawImage(video, 0, 0, w, hgt);
    const id = ++this.frameId;
    this.inFlight = id;
    this.inFlightSince = performance.now();
    const header = { id };
    if (this.label) header.label = this.label;
    if (this.gesture) header.gesture = this.gesture;
    if (this.overlay) header.overlay = true;
    this.canvas.toBlob((blob) => {
      if (!blob || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
        if (this.inFlight === id) this.inFlight = null;
        return;
      }
      const hb = new TextEncoder().encode(JSON.stringify(header));
      const len = new Uint8Array(4);
      new DataView(len.buffer).setUint32(0, hb.length, true);
      this.sentAt.set(id, performance.now());
      this.ws.send(new Blob([len, hb, blob]));
    }, 'image/jpeg', this.quality);
  }
}

/**
 * Demo mode: the mouse plays the role of your eyes and keys play the role of
 * the eyelids:
 *   B        a blink (press it twice quickly to "double blink"; hold it about
 *            a second for a long close of both eyes)
 *   Q / E    hold to keep your left / right eye closed (a wink: hold to drag)
 * Handy to try the website without a webcam. It emits the same events as
 * EyeTracker and still talks to the server for people and personalisation.
 */
export class SimTracker extends Channel {
  constructor() {
    super();
    this.simulated = true;
    this.frameId = 0;
    this.latencyMs = 0;
    this.overlay = false;
    this.label = null;
    this.mouse = null;
    this.closedUntil = 0;
    this.pendingBlink = null;
    this.blinkGapMs = 550;
    this.holdMs = 350;          // personal wink hold time (set by the app)
    this.longCloseMs = 1000;
    this.winkEnabled = { left: true, right: true };
    this.both = null;           // B held: {start, preFrame, ready}
    this.wink = null;           // Q / E held: {eye, start, preFrame, started}
  }

  async start() {
    this.running = true;
    // Only the real mouse is the "gaze": not the pointer events that eye presses dispatch.
    this._onMove = (e) => { if (e.isTrusted) this.mouse = clientToScreen(e.clientX, e.clientY); };
    const key = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return null;
      if (e.target && e.target.closest && e.target.closest('input, textarea, [contenteditable]')) return null;
      return { b: 'both', q: 'left', e: 'right' }[e.key.toLowerCase()] || null;
    };
    this._onKey = (e) => {
      const which = key(e);
      if (!which || e.repeat) return;
      if (which === 'both') this.closeBoth();
      else this.closeOne(which);
    };
    this._onKeyUp = (e) => {
      const which = key(e);
      if (!which) return;
      if (which === 'both') this.openBoth();
      else this.openOne(which);
    };
    window.addEventListener('pointermove', this._onMove);
    window.addEventListener('keydown', this._onKey);
    window.addEventListener('keyup', this._onKeyUp);
    this.timer = setInterval(() => this.tick(), 33);
    this.connect();
  }

  stop() {
    this.running = false;
    clearInterval(this.timer);
    window.removeEventListener('pointermove', this._onMove);
    window.removeEventListener('keydown', this._onKey);
    window.removeEventListener('keyup', this._onKeyUp);
    if (this.ws) this.ws.close();
  }

  onMessage(msg) {
    // Without a camera the server may report that face tracking is unavailable,
    // and it never sends frames: neither matters in demo mode.
    if (msg.type === 'fatal' || msg.type === 'frame') return;
    if (msg.type === 'hello') msg = { ...msg, simulated: true };
    super.onMessage(msg);
  }

  setLabel(label) {
    this.label = label;
  }

  setGesturePhase() {}

  tick() {
    const id = ++this.frameId;
    const now = performance.now();
    const closed = now < this.closedUntil || !!this.both;
    const gaze = this.mouse ? [this.mouse.x, this.mouse.y] : null;
    const w = this.wink;
    if (w && !w.started && now - w.start >= this.holdMs) {
      w.started = true;
      this.emit('wink_start', { type: 'wink_start', eye: w.eye, frame: id, pre_frame: w.preFrame, at: null,
        duration_ms: Math.round(now - w.start) });
    }
    const b = this.both;
    if (b && !b.ready && now - b.start >= this.longCloseMs) {
      b.ready = true;
      this.emit('long_close_ready', { type: 'long_close_ready', frame: id });
    }
    const shut = 0.9;
    const cl = closed || (w && w.eye === 'left') ? shut : 0.12;
    const cr = closed || (w && w.eye === 'right') ? shut : 0.12;
    this.emit('frame', {
      type: 'frame', id, face: true, gaze, raw: gaze, frozen: closed, labeled: !!this.label && !closed && !w,
      closure: closed ? shut : 0.12, cl, cr, closed, closing: closed, thr: [0.5, 0.35], fps: 30, ms: 0,
      head: [0, 0, 0], dist: 60, winking: w ? w.eye : null, wink: w && w.started ? w.eye : null,
      net: w ? (w.eye === 'left' ? 'right' : 'left') : 'both',
    });
    if (this.pendingBlink && !closed && now - this.pendingBlink.end > this.blinkGapMs) {
      this.pendingBlink = null;
      this.emit('blink_expired', { type: 'blink_expired', frame: id });
    }
  }

  // -- both eyes (B) ---------------------------------------------------------------
  closeBoth() {
    if (this.both) return;
    this.both = { start: performance.now(), preFrame: this.frameId, ready: false };
  }

  openBoth() {
    const b = this.both;
    if (!b) return;
    this.both = null;
    const now = performance.now();
    const held = now - b.start;
    if (held < 700) {
      // A blink: keep the "eyes" closed for a natural ~140 ms, then report it.
      this.closedUntil = Math.max(now, b.start + 140);
      setTimeout(() => this.blinkDone(b.start, b.preFrame), Math.max(0, this.closedUntil - now) + 10);
    } else if (held >= this.longCloseMs && held <= 6000) {
      this.pendingBlink = null;
      this.emit('long_close', { type: 'long_close', frame: this.frameId, pre_frame: b.preFrame, at: null,
        duration_ms: Math.round(held) });
    }
  }

  blinkDone(start, preFrame) {
    const end = performance.now();
    if (this.pendingBlink && start - this.pendingBlink.end <= this.blinkGapMs) {
      const first = this.pendingBlink;
      this.pendingBlink = null;
      this.emit('blink', { type: 'blink', n: 2, frame: this.frameId });
      this.emit('double_blink', { type: 'double_blink', frame: this.frameId, pre_frame: first.preFrame, at: null });
    } else {
      this.pendingBlink = { start, end, preFrame };
      this.emit('blink', { type: 'blink', n: 1, frame: this.frameId });
    }
  }

  // -- one eye (Q / E) ----------------------------------------------------------------
  closeOne(eye) {
    if (this.wink || !this.winkEnabled[eye]) return;
    this.wink = { eye, start: performance.now(), preFrame: this.frameId, started: false };
  }

  openOne(eye) {
    const w = this.wink;
    if (!w || w.eye !== eye) return;
    this.wink = null;
    const held = performance.now() - w.start;
    if (w.started) {
      this.emit('wink_end', { type: 'wink_end', eye, frame: this.frameId, duration_ms: Math.round(held), at: null,
        cancelled: false });
    } else if (held >= 120) {
      this.emit('wink', { type: 'wink', eye, frame: this.frameId, pre_frame: w.preFrame, at: null,
        duration_ms: Math.round(held) });
    }
  }
}
