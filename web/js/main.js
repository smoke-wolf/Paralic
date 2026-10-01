// Paralic — application bootstrap and top-level flows (start screen, face
// check, calibration, pause, navigation).

import { $, $$, esc, h, toast } from './dom.js';
import { clientToScreen } from './screen-space.js';
import { icon } from './icons.js';
import { EyeTracker, SimTracker } from './tracker.js';
import { GazeController } from './gaze.js';
import { GestureController } from './gestures.js';
import { Calibrator, rateAccuracy } from './calibration.js';
import { HandCalibrator } from './hand-calibration.js';
import { CameraPanel } from './camera-panel.js';
import { getSettings, onSettingsChange, serverSettings, updateSettings } from './settings.js';
import { unlockAudio } from './sound.js';
import { Router } from './router.js';

import home from './pages/home.js';
import explore from './pages/explore.js';
import read from './pages/read.js';
import talk from './pages/talk.js';
import practice from './pages/practice.js';
import settingsPage from './pages/settings.js';
import help from './pages/help.js';
import arrange from './pages/arrange.js';
import draw from './pages/draw.js';
import lab from './pages/lab.js';

const ROUTES = { home, explore, read, talk, practice, settings: settingsPage, help, arrange, draw, lab };
const params = new URLSearchParams(location.search);

class App {
  constructor() {
    this.appEl = $('#app');
    this.pageEl = $('#page');
    this.overlayRoot = $('#overlay-root');
    this.events = new EventTarget();
    this.tracker = null;
    this.gaze = null;
    this.camera = null;
    this.state = {
      started: false, calibrated: false, accuracy: null, simulated: false, profile: null,
      person: null, people: [], personal: null,
    };

    this.decorateNav();
    this.router = new Router(this.pageEl, ROUTES, this);
    this.calibrator = new Calibrator(this);
    this.bindChrome();
    this.router.start();
    this.showStart();
  }

  // -- helpers used by pages ------------------------------------------------------
  on(type, fn) {
    const handler = (e) => fn(e.detail);
    this.events.addEventListener(type, handler);
    return () => this.events.removeEventListener(type, handler);
  }

  emit(type, detail) {
    this.events.dispatchEvent(new CustomEvent(type, { detail }));
  }

  navigate(hash) {
    location.hash = hash;
  }

  openOverlay(cls = '') {
    const ov = h('div', { class: `overlay ${cls}` });
    this.overlayRoot.append(ov);
    this.appEl.inert = true;
    return ov;
  }

  closeOverlay(ov) {
    ov.remove();
    if (!this.overlayRoot.children.length) this.appEl.inert = false;
  }

  /** Giant gaze-friendly choice buttons. Resolves with the chosen id. */
  choose(container, { title, subtitle, choices, rowStyle } = {}) {
    return new Promise((resolve) => {
      const wrap = h('div', { class: 'choice-wrap' });
      if (title) wrap.append(h('div', { class: 'choice-title' }, h('h2', {}, title), subtitle ? h('p', {}, subtitle) : null));
      const row = h('div', { class: 'choice-row', style: rowStyle || {} });
      for (const c of choices) {
        const b = h('button', { class: `choice ${c.primary ? 'primary' : ''}`, type: 'button', 'data-choice': c.id });
        b.innerHTML = `${icon(c.icon || 'check')}<span>${esc(c.label)}</span>${c.sub ? `<small>${esc(c.sub)}</small>` : ''}`;
        b.addEventListener('click', () => {
          wrap.remove();
          resolve(c.id);
        });
        row.append(b);
      }
      wrap.append(row);
      container.append(wrap);
    });
  }

  /**
   * Scanning choice: the options light up one after another and a double blink
   * picks the lit one. Works before any calibration (e.g. choosing who you are).
   */
  scanChoose(container, { title, subtitle, choices, dwellMs = 1800 } = {}) {
    return new Promise((resolve) => {
      const wrap = h('div', { class: 'choice-wrap scan' });
      if (title) wrap.append(h('div', { class: 'choice-title' }, h('h2', {}, title), subtitle ? h('p', {}, subtitle) : null));
      const grid = h('div', { class: 'scan-grid' });
      const buttons = choices.map((c) => {
        const b = h('button', { class: 'choice', type: 'button', 'data-choice': c.id });
        b.innerHTML = `${icon(c.icon || 'head')}<span>${esc(c.label)}</span>${c.sub ? `<small>${esc(c.sub)}</small>` : ''}`;
        b.addEventListener('click', () => finish(c.id));
        grid.append(b);
        return b;
      });
      wrap.append(grid);
      container.append(wrap);
      let index = 0;
      const light = () => buttons.forEach((b, i) => b.classList.toggle('scan-focus', i === index));
      light();
      const timer = setInterval(() => { index = (index + 1) % buttons.length; light(); }, dwellMs);
      const wasSuspended = this.gaze ? this.gaze.suspended : false;
      if (this.gaze) this.gaze.setSuspended(true);
      const off = this.gaze ? this.gaze.onDoubleBlinkFirst(() => { finish(choices[index].id); return true; }) : () => {};
      let done = false;
      function finish(id) {
        if (done) return;
        done = true;
        clearInterval(timer);
        off();
        wrap.remove();
        resolve(id);
      }
      this._afterScan = () => { if (this.gaze) this.gaze.setSuspended(wasSuspended); };
    }).finally(() => this._afterScan && this._afterScan());
  }

  // -- people & personalisation ----------------------------------------------------------
  setPersonal(view) {
    if (!view) return;
    this.state.personal = view;
    if (this.gaze) {
      this.gaze.personalMagnet = view.magnet || null;
      this.gaze.gestures = view.gestures || null;
      // The cursor's jitter after smoothing sizes the "hold still" radius.
      this.gaze.precisionPx = (view.smoothing_profile || {}).expected_jitter_px || null;
    }
    const g = view.gestures;
    if (this.tracker && this.tracker.simulated && g) {
      // Demo mode imitates the server's gesture detection with the same settings.
      this.tracker.holdMs = Number(g.hold_ms) || 350;
      this.tracker.longCloseMs = Number(g.long_close_ms) || 1000;
      this.tracker.headKeys = !!g.head_nudge;
      if (view.winks) this.tracker.winkEnabled = { ...view.winks };
    }
    this.emit('personal', view);
  }

  applyUsersReply(reply) {
    if (!reply || reply.ok === false) return;
    this.state.people = reply.users || [];
    this.state.person = reply.user || null;
    this.state.profile = reply.profile || null;
    this.setPersonal(reply.personal);
    this.emit('people', reply);
  }

  /** Ask who is using Paralic (scanning, so it works before calibration). */
  async pickPerson(ov) {
    const people = this.state.people || [];
    const choices = people.map((u) => ({
      id: u.id, label: u.name, icon: 'head',
      sub: u.calibrated ? 'Calibrated' : 'Not calibrated yet',
    }));
    choices.push({ id: '__new', label: 'New person', sub: 'Start a fresh profile', icon: 'sparkle' });
    const id = await this.scanChoose(ov, {
      title: 'Who’s using Paralic?',
      subtitle: 'Blink twice when your name lights up (or click it).',
      choices,
    });
    const reply = id === '__new'
      ? await this.tracker.request({ type: 'user_create' }, 'users')
      : await this.tracker.request({ type: 'user_select', id }, 'users');
    this.applyUsersReply(reply);
    this.state.calibrated = false;
    this.state.accuracy = null;
    if (this.gaze) {
      this.gaze.setActive(false);
      this.gaze.resetBias();
    }
    return reply;
  }

  /** Switch person from anywhere (Home, Lab): pick, then load or calibrate. */
  async switchPerson() {
    if (!this.tracker || this.calibrator.running) return;
    const ov = this.openOverlay('solid');
    await this.pickPerson(ov);
    await this.enterAsPerson(ov);
  }

  // -- chrome: nav, pause, keyboard ---------------------------------------------------
  decorateNav() {
    for (const el of $$('.nav-btn[data-icon]')) {
      const label = el.textContent.trim();
      el.innerHTML = `${icon(el.dataset.icon)}<span>${label}</span>`;
    }
  }

  bindChrome() {
    $('[data-action="back"]').addEventListener('click', () => this.router.back());
    $('#pause-btn').addEventListener('click', () => this.gaze && this.gaze.setPaused(!this.gaze.paused));
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.gestures) this.gestures.cancelAll();
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
      if (!this.state.started || this.calibrator.running || this.overlayRoot.children.length) return;
      if (e.target.closest && e.target.closest('input, textarea, [contenteditable]')) return;
      const key = e.key.toLowerCase();
      if (key === 'c' && !this.state.simulated) this.calibrate('full');
      else if (key === 'a' && !this.state.simulated && this.state.calibrated) this.calibrate('adjust');
      else if (key === 'p' && this.gaze) this.gaze.setPaused(!this.gaze.paused);
    });
    onSettingsChange((s, patch) => {
      if (this.tracker) this.tracker.send({ type: 'settings', ...serverSettings(s) });
      if ('showCamera' in patch && this.camera) this.camera.setVisible(s.showCamera);
    });
  }

  updatePauseUi(paused) {
    const btn = $('#pause-btn');
    btn.innerHTML = `${icon(paused ? 'play' : 'pause')}<span>${paused ? 'Resume' : 'Pause'}</span>`;
    $('#paused-banner').hidden = !paused;
  }

  // -- tracker setup -------------------------------------------------------------------
  createTracker(simulated, mode = 'eyes') {
    if (this.gestures) this.gestures.detach();
    if (this.gaze) this.gaze.destroy();
    const frameWidth = Number(params.get('fw')) || 960;
    this.tracker = simulated ? new SimTracker() : new EyeTracker({ frameWidth, mode });
    this.state.simulated = simulated;
    this.state.handMode = mode === 'hand';
    document.body.classList.toggle('hand-mode', this.state.handMode);
    this.gaze = new GazeController(this.tracker, { cursorEl: $('#gaze-cursor'), pageEl: this.pageEl, railEl: $('#scroll-rail') });
    this.gaze.addEventListener('activate', (e) => this.emit('activate', e.detail));
    this.gestures = new GestureController(this);
    this.gestures.attach(this.tracker);
    this.gaze.addEventListener('pausechange', (e) => {
      this.updatePauseUi(e.detail);
      this.emit('pausechange', e.detail);
    });
    this.camera = new CameraPanel(this.tracker);
    // Hand mode: pinch-drag scroll and open-palm pause arrive as their own
    // messages (clicks reuse the double-blink path, so the cursor code is shared).
    this.tracker.on('hand_scroll', (m) => {
      const sc = (this.gaze && this.gaze.scrollContainer && this.gaze.scrollContainer())
        || document.scrollingElement || document.documentElement;
      if (sc && typeof m.dy === 'number') sc.scrollBy(0, m.dy);
    });
    this.tracker.on('hand_pause', (m) => {
      if (this.gaze && this.gaze.setPaused) this.gaze.setPaused(!!m.paused);
      toast(m.paused ? 'Paused — open palm again to resume' : 'Resumed', 'ok');
    });
    this.tracker.on('fatal', (m) => this.showFatal(m.error));
    this.tracker.on('connection', ({ connected }) => {
      if (!connected && this.state.started) toast('Lost connection to the eye tracker — reconnecting…', 'warn');
      if (connected && this.state.started && this.state.calibrated && !simulated) this.reloadProfileAfterReconnect();
    });
    this.tracker.on('hello', (m) => {
      this.state.hello = m;
      this.state.profile = m.profile || null;
      this.state.people = m.users || [];
      this.state.person = m.user || null;
      this.setPersonal(m.personal);
    });
    this.tracker.on('users', (m) => this.applyUsersReply(m));
    for (const type of ['profile', 'validation_result', 'calibration_result', 'blink_calibration_result',
      'wink_calibration_result', 'experiment_result', 'personal', 'experiment_plan']) {
      this.tracker.on(type, (m) => m && m.personal && this.setPersonal(m.personal));
    }
    this.tracker.on('face_changed', (m) => this.offerSwitch(m));
    this.tracker.on('finetune_result', (m) => {
      if (m.personal) this.setPersonal(m.personal);
      if (m.ok && m.accepted) {
        this.gaze.resetBias(); // the new network already includes the correction
        toast(`Your gaze model improved: ${Math.round(m.champion_error_px)} → ${Math.round(m.candidate_errors_px[m.winner])} px (v${m.version})`, 'ok', 6000);
      } else if (!m.auto) {
        toast(m.ok ? 'Fine-tuning finished: the current model is still the best' : (m.error || 'Fine-tuning failed'), m.ok ? '' : 'warn', 6000);
      }
      this.emit('finetune', m);
    });
    this.tracker.on('system_control', (m) => this.onSystemControl(m));
    // Learn from clicks: the frames before each double-blink click are training data.
    this.gaze.addEventListener('activate', (e) => this.learnFromActivation(e.detail));
  }

  /** Turn whole-computer desktop control on or off. */
  toggleSystemControl(enable) {
    const want = enable === undefined ? !(this.state.systemControl && this.state.systemControl.enabled) : enable;
    this.tracker.send({ type: 'system_control', enabled: want });
  }

  onSystemControl(m) {
    this.state.systemControl = m;
    this.emit('system_control', m);
    if (m.enabled) {
      this.showSystemBanner(true);
      if (typeof speak === 'function') speak('Desktop control on');
    } else {
      this.showSystemBanner(false);
      if (m.reason) toast(m.reason, m.needs_permission ? 'warn' : '', 7000);
      if (m.needs_permission) {
        toast('Open System Settings → Privacy & Security → Accessibility, enable your browser’s helper (or Paralic), then turn it on again.', 'warn', 11000);
      }
    }
  }

  showSystemBanner(on) {
    let b = document.getElementById('system-banner');
    if (on) {
      if (!b) {
        b = h('div', { id: 'system-banner', class: 'system-banner' },
          h('span', {}, '🖱️ Desktop control is ON — the cursor follows your eyes everywhere. '),
          h('b', {}, 'To stop: look at the top-left corner for a second'), h('span', {}, ', or turn it off in Settings.'));
        document.body.append(b);
      }
    } else if (b) {
      b.remove();
    }
  }

  learnFromActivation({ element, preFrame }) {
    if (!element || preFrame === null || preFrame === undefined || this.state.simulated) return;
    if (!getSettings().learning || !this.state.calibrated) return;
    if (element.closest('.scroll-rail, .choice-row, .scan-grid, [data-no-learn]')) return;
    const r = element.getBoundingClientRect();
    const a = clientToScreen(r.left, r.top);
    const b = clientToScreen(r.right, r.bottom);
    if (element.dataset.learn === 'practice') {
      this.tracker.send({ type: 'label_event', kind: 'practice', pre_frame: preFrame,
        target: [(a.x + b.x) / 2, (a.y + b.y) / 2] });
    } else {
      this.tracker.send({ type: 'label_event', kind: 'click', pre_frame: preFrame, rect: [a.x, a.y, b.x, b.y] });
    }
  }

  async reloadProfileAfterReconnect() {
    // The server lost its in-memory model (e.g. it was restarted): reload the saved one.
    try {
      const p = await this.tracker.request({ type: 'profile_load' }, 'profile', 15000);
      if (p.loaded) toast('Reconnected', 'ok');
      else {
        // No saved calibration to restore: ask for a new one (a double blink
        // works without calibration, so this stays usable with the eyes).
        this.state.calibrated = false;
        this.gaze.setActive(false);
        toast('Reconnected — the calibration has to be redone', 'warn', 6000);
        if (!this.overlayRoot.children.length && !this.calibrator.running) {
          this.askToCalibrate(this.openOverlay('solid'));
        }
      }
    } catch {
      /* next reconnect will retry */
    }
  }

  // -- start screen ---------------------------------------------------------------------
  showStart() {
    const ov = this.openOverlay('solid');
    const errorBox = h('div', { class: 'error-box', hidden: true });
    const startBtn = h('button', { class: 'btn primary', type: 'button', id: 'start-btn' });
    startBtn.innerHTML = `${icon('eye')}<span>Start eye tracking</span>`;
    const handBtn = h('button', { class: 'btn', type: 'button', id: 'hand-btn' });
    handBtn.innerHTML = `${icon('mouse')}<span>Use your hands</span>`;
    const demoBtn = h('button', { class: 'btn', type: 'button', id: 'demo-btn' });
    demoBtn.innerHTML = `${icon('mouse')}<span>Try with a mouse</span>`;
    const tips = [
      ['camera', 'Allow camera access when your browser asks. Video never leaves this computer.'],
      ['sun', 'Light your face evenly from the front and sit about an arm’s length from the screen.'],
      ['eye', 'Look to move the cursor. Blink twice to click. Look at the arrows on the right to scroll.'],
    ];
    const card = h('div', { class: 'overlay-card' },
      h('div', { class: 'start-hero' },
        h('div', {},
          h('div', { class: 'eyebrow' }, 'Hands-free browsing'),
          h('h1', {}, 'Browse the web with your eyes.'),
          h('p', { class: 'lead' }, 'Paralic uses your webcam and neural networks to follow your gaze. Look to move, blink twice to click.'),
          h('div', { class: 'start-actions' }, startBtn, handBtn, demoBtn),
          h('ul', { class: 'start-list' }, tips.map(([ic, text]) => h('li', { html: `${icon(ic)}<span>${text}</span>` }))),
          errorBox),
        h('div', { class: 'start-visual' }, h('div', { class: 'start-eye' }))));
    ov.append(card);

    const statusReady = fetch('/api/status').then((r) => r.json()).then((status) => {
      if (status.error || !status.tracker) {
        errorBox.hidden = false;
        errorBox.textContent = `The eye tracker could not start:\n${status.error || 'unknown error'}\n\nYou can still try the website with a mouse.`;
        startBtn.disabled = true;
        return false;
      }
      return true;
    }).catch(() => true);

    startBtn.addEventListener('click', () => {
      unlockAudio();
      if (!params.has('nofs')) document.documentElement.requestFullscreen?.().catch(() => {});
      this.startTracking(ov, card, errorBox);
    });
    handBtn.addEventListener('click', () => {
      unlockAudio();
      if (!params.has('nofs')) document.documentElement.requestFullscreen?.().catch(() => {});
      this.startTracking(ov, card, errorBox, 'hand');
    });
    demoBtn.addEventListener('click', () => {
      unlockAudio();
      this.startDemo(ov);
    });

    if (params.has('demo')) {
      this.startDemo(ov);
      return;
    }
    if (params.has('hands')) {
      this.startTracking(ov, card, errorBox, 'hand');
      return;
    }
    // If the camera is already allowed (and the tracker is OK), start right away: no click needed.
    statusReady.then((trackerOk) => {
      if (!trackerOk || params.has('manual')) return;
      navigator.permissions?.query({ name: 'camera' }).then((p) => {
        if (p.state === 'granted' && !this.state.started && ov.isConnected) this.startTracking(ov, card, errorBox);
      }).catch(() => {});
    });
  }

  async startTracking(ov, card, errorBox, mode = 'eyes') {
    if (this.state.started) return;
    this.state.started = true;
    unlockAudio();
    errorBox.hidden = true;
    if (!this.tracker || this.tracker.simulated || this.tracker.mode !== mode) this.createTracker(false, mode);
    try {
      await this.tracker.start($('#camera-video'));
    } catch (err) {
      this.state.started = false;
      this.tracker.stop();
      errorBox.hidden = false;
      const name = err && err.name;
      errorBox.textContent =
        name === 'NotAllowedError' ? 'Camera access was blocked. Allow the camera for this page (click the camera icon in the address bar), then press Start again.'
          : name === 'NotFoundError' ? 'No camera was found. Connect a webcam and press Start again.'
            : name === 'NotReadableError' ? 'The camera is being used by another app. Close it and press Start again.'
              : `Could not start the camera: ${err && err.message ? err.message : err}`;
      return;
    }
    card.remove();
    await this.faceCheck(ov);
  }

  startDemo(ov) {
    if (this.state.started) return;
    this.state.started = true;
    unlockAudio();
    this.createTracker(true);
    this.tracker.start();
    this.state.calibrated = true;
    this.gaze.setActive(true);
    this.closeOverlay(ov);
    this.camera.setVisible(false);
    toast('Demo mode: the mouse is your gaze. Press B twice quickly to "double blink"; hold Q or E to keep your left or right eye closed (drag).', 'ok', 8000);
  }

  /** A large mirrored camera preview with the eye outlines drawn on top. */
  makePreview() {
    const video = h('video', { playsinline: true, muted: true, autoplay: true });
    video.srcObject = this.tracker.stream;
    video.play().catch(() => {});
    const canvas = h('canvas');
    const guide = h('div', { class: 'face-guide' });
    const el = h('div', { class: 'face-preview' }, video, canvas, guide);
    const removeView = this.camera.addView(canvas, video);
    return { el, guide, remove: () => { removeView(); video.srcObject = null; } };
  }

  /** Show the camera, wait until a face is found, then load or create a calibration. */
  async faceCheck(ov) {
    const hand = this.state.handMode;
    const preview = this.makePreview();
    const status = h('p', { class: 'big-status' }, hand ? 'Looking for your hand…' : 'Looking for your face…');
    const hint = h('p', { class: 'muted' }, hand ? 'Raise one hand into view.' : 'Centre your face in the oval.');
    const card = h('div', { class: 'overlay-card' },
      h('div', { class: 'face-check' },
        preview.el,
        h('div', {}, h('div', { class: 'eyebrow' }, 'Camera check'), status, hint)));
    ov.append(card);

    let good = 0;
    const shownAt = performance.now();
    let hello = this.state.hello || null;
    const offHello = this.tracker.on('hello', (m) => { hello = m; });
    await new Promise((resolve) => {
      const off = this.tracker.on('frame', (m) => {
        if (!m.face) {
          good = 0;
          status.textContent = hand ? 'Looking for your hand…' : 'Looking for your face…';
          hint.textContent = hand ? 'Raise one hand into view, fingers spread.'
            : 'Centre your face in the oval and make sure it is well lit.';
          preview.guide.classList.remove('ok');
          return;
        }
        preview.guide.classList.add('ok');
        if (hand) {
          hint.textContent = 'Great — hold still…';
        } else {
          const [yaw, pitch] = m.head || [0, 0];
          if (m.dist && m.dist < 30) hint.textContent = 'You are quite close — lean back a little.';
          else if (m.dist && m.dist > 95) hint.textContent = 'You are far away — move a little closer.';
          else if (Math.abs(yaw) > 22 || Math.abs(pitch) > 22) hint.textContent = 'Face the screen straight on.';
          else hint.textContent = 'Great — hold still…';
        }
        status.textContent = hand ? 'Hand found' : 'Face found';
        good += 1;
        if (good >= 20 && hello && performance.now() - shownAt > 1500) {
          off();
          resolve();
        }
      });
    });
    offHello();
    preview.remove();
    card.remove();
    this.camera.setVisible(getSettings().showCamera);

    // Who is it? Recognised by their face print (eye mode) - else, when
    // several people use this computer, ask.
    const known = this.state.handMode ? null : await this.recognise(ov);
    if (!known && (this.state.people || []).length > 1) await this.pickPerson(ov);
    if (this.state.handMode) {
      // Hand mode: run (or reuse) the person's pointing + pinch setup.
      await this.runHandSetup(ov);
      this.state.calibrated = true;
      this.gaze.setActive(true);
      this.closeOverlay(ov);
      this.welcome();
      return;
    }
    await this.enterAsPerson(ov, known);
  }

  /** Recognise the person at the camera from the face prints and switch to
   *  them. Resolves to their name, or null when unsure (or no prints yet). */
  async recognise(ov) {
    if (this.state.simulated || !(this.state.people || []).some((u) => u.face)) return null;
    const card = h('div', { class: 'overlay-card' },
      h('div', { class: 'eyebrow' }, 'Face print'), h('h2', {}, 'Recognising you…'), h('div', { class: 'spinner' }));
    ov.append(card);
    try {
      const r = await this.tracker.request({ type: 'face_recognize' }, 'face_recognition', 9000).catch(() => null);
      if (!r || !r.user) return null;
      if (!this.state.person || this.state.person.id !== r.user) {
        this.applyUsersReply(await this.tracker.request({ type: 'user_select', id: r.user }, 'users'));
      }
      return r.name || (this.state.person ? this.state.person.name : null);
    } finally {
      card.remove();
    }
  }

  /** Someone else with a face print sat down: offer to switch to them. */
  offerSwitch(m) {
    if (!m || !m.user || !m.name || this.calibrator.running || this.state.simulated) return;
    document.querySelector('.face-banner')?.remove();
    const yes = h('button', { class: 'btn primary', type: 'button', html: `${icon('head')}<span>Switch to ${esc(m.name)}</span>` });
    const no = h('button', { class: 'btn', type: 'button' }, 'No');
    const banner = h('div', { class: 'face-banner', role: 'alert' }, h('span', {}, `Is that ${m.name}?`), yes, no);
    document.body.append(banner);
    const timer = setTimeout(() => banner.remove(), 15000);
    no.addEventListener('click', () => { clearTimeout(timer); banner.remove(); });
    yes.addEventListener('click', async () => {
      clearTimeout(timer);
      banner.remove();
      const ov = this.openOverlay('solid');
      try {
        this.applyUsersReply(await this.tracker.request({ type: 'user_select', id: m.user }, 'users'));
        this.state.calibrated = false;
        await this.enterAsPerson(ov, m.name);
      } catch (err) {
        this.closeOverlay(ov);
        toast(`Could not switch: ${err.message || err}`, 'bad');
      }
    });
  }

  /** Hand mode: run a fresh pointing+pinch setup, or reuse a saved one. */
  async runHandSetup(ov) {
    const cal = new HandCalibrator(this);
    const saved = this.state.hello && this.state.hello.calibration;
    let mode = 'full';
    if (saved) {
      this.gaze.setActive(true);
      this.gaze.setSuspended(false);
      const choice = await this.choose(ov, {
        choices: [
          { id: 'use', label: 'Use saved setup', sub: 'Keep your last hand calibration', icon: 'check', primary: true },
          { id: 'point', label: 'Quick re-point', sub: 'Redo just the pointing dots', icon: 'mouse' },
          { id: 'full', label: 'Full setup', sub: 'Hand span, pointing and pinch', icon: 'refresh' },
        ],
      }).catch(() => 'use');
      this.gaze.setSuspended(true);
      if (choice === 'use') {
        await this.tracker.request({ type: 'hand_profile_use' }, 'hand_calibration_result', 8000).catch(() => {});
        return;
      }
      mode = choice === 'point' ? 'point' : 'full';
    }
    await cal.run({ mode });
  }

  /** Load the current person's calibration and offer to use, adjust or redo it.
   *  `recognised`: their name when the face print recognised them. */
  async enterAsPerson(ov, recognised = null) {
    for (;;) {
      const p = await this.tracker.request({ type: 'profile_load' }, 'profile', 15000).catch(() => ({ loaded: false }));
      if (!p.loaded) {
        // Say why a saved calibration could not be used (not just "none yet").
        if (p.error && p.error !== 'No saved calibration') toast(`${p.error} Let’s calibrate again.`, 'warn', 9000);
        await this.askToCalibrate(ov);
        return;
      }
      this.state.calibrated = true;
      this.state.accuracy = p.accuracy_px || null;
      this.gaze.setActive(true);
      const name = this.state.person ? this.state.person.name : '';
      const choice = await this.choose(ov, {
        title: recognised ? `Hello, ${recognised}! I recognised you.` : name ? `Welcome back, ${name}!` : 'Welcome back!',
        subtitle: p.legacy
          ? 'Your calibration comes from an older version of Paralic. It still works, but a new full calibration will be more accurate.'
          : 'Your saved calibration is loaded. Look at an option and blink twice.',
        choices: [
          { id: 'browse', label: 'Start browsing', sub: 'Use the saved calibration as is', icon: 'arrowRight' },
          { id: 'adjust', label: 'Quick adjust', sub: '9 dots, about 20 seconds' + (p.legacy ? '' : ' (recommended)'), icon: 'crosshair', primary: !p.legacy },
          { id: 'full', label: 'Full calibration', sub: 'About a minute and a half' + (p.legacy ? ' (recommended)' : ''), icon: 'refresh', primary: !!p.legacy },
          { id: 'switch', label: 'Switch person', sub: 'Someone else is using Paralic', icon: 'head' },
        ],
      });
      if (choice === 'switch') {
        await this.pickPerson(ov);
        recognised = null;
        continue;
      }
      // Going on as this person: their face may now be learned for the face print.
      this.tracker.send({ type: 'face_confirm' });
      this.closeOverlay(ov);
      if (choice !== 'browse') await this.calibrate(choice);
      this.welcome();
      return;
    }
  }

  /** First-time calibration prompt; a double blink works before calibration. */
  async askToCalibrate(ov) {
    const preview = this.makePreview();
    preview.guide.classList.add('ok');
    const btn = h('button', { class: 'btn primary', type: 'button' });
    btn.innerHTML = `${icon('crosshair')}<span>Start calibration</span>`;
    const feedback = h('span', { class: 'pill', style: { fontSize: '1rem', padding: '10px 18px' } },
      'Waiting for two quick blinks…');
    const name = this.state.person ? this.state.person.name : null;
    const switchBtn = h('button', { class: 'btn', type: 'button' });
    switchBtn.innerHTML = `${icon('head')}<span>${name ? `Not ${esc(name)}?` : 'Switch person'}</span>`;
    const card = h('div', { class: 'overlay-card' },
      h('div', { class: 'face-check' },
        preview.el,
        h('div', {},
          h('div', { class: 'eyebrow' }, name ? `One more step, ${name}` : 'One more step'),
          h('h1', {}, 'Blink twice to calibrate'),
          h('p', { class: 'muted', style: { fontSize: '1.15rem' } },
            'First you get comfortable, then a dot moves around the screen: look at it until it shrinks away. It takes about a minute and a half and teaches the neural network how your eyes look at your screen, and how you blink.'),
          h('p', {}, feedback),
          h('div', { class: 'btn-row' }, btn, (this.state.people || []).length > 0 ? switchBtn : null))));
    ov.append(card);
    const offBlink = this.tracker.on('blink', (m) => {
      if (m.n === 1) {
        feedback.textContent = 'One blink — now blink again, quickly';
        feedback.className = 'pill ok';
      }
    });
    const offExpired = this.tracker.on('blink_expired', () => {
      feedback.textContent = 'Almost! Blink twice a little faster';
      feedback.className = 'pill warn';
    });
    const action = await new Promise((resolve) => {
      const off = this.gaze.onDoubleBlinkFirst(() => {
        off();
        resolve('calibrate');
        return true;
      });
      btn.addEventListener('click', () => {
        off();
        resolve('calibrate');
      });
      switchBtn.addEventListener('click', () => {
        off();
        resolve('switch');
      });
    });
    offBlink();
    offExpired();
    preview.remove();
    if (action === 'switch') {
      card.remove();
      await this.pickPerson(ov);
      await this.enterAsPerson(ov);
      return;
    }
    this.closeOverlay(ov);
    const result = await this.calibrate('full');
    if (!result && !this.state.calibrated) {
      await this.askToCalibrate(this.openOverlay('solid'));
      return;
    }
    this.welcome();
  }

  welcome() {
    if (this.state.welcomed) return;
    this.state.welcomed = true;
    toast('Look at anything and blink twice to click it', 'ok', 6000);
  }

  async calibrate(mode = 'full') {
    if (!this.tracker || this.state.simulated) {
      toast('Calibration needs the camera (demo mode uses the mouse)', 'warn');
      return null;
    }
    if ((mode === 'adjust' || mode === 'adjust-mouse') && !this.state.calibrated) mode = 'full';
    if (mode === 'blink' || mode === 'wink' || mode === 'head') {
      try {
        return await this.calibrator.run(mode);
      } catch (err) {
        toast(`The ${mode} test failed: ${err.message || err}`, 'bad', 6000);
        return null;
      }
    }
    const wasPaused = this.gaze.paused;
    if (wasPaused) this.gaze.setPaused(false);
    let result = null;
    try {
      result = await this.calibrator.run(mode);
    } catch (err) {
      console.error(err);
      toast(`Calibration failed: ${err.message || err}`, 'bad', 6000);
    }
    if (result && result.fit && result.fit.ok !== false) {
      this.state.calibrated = true;
      if (result.validation) this.state.accuracy = result.validation.mean_error_px;
      this.gaze.setActive(true);
      this.gaze.resetBias();
      this.emit('calibrated', result);
    }
    return result;
  }

  /** Apply the client-side part of an adopted A/B winner (e.g. switch to "auto"). */
  adoptSettings(patch) {
    if (patch && typeof patch === 'object') updateSettings(patch);
  }

  accuracyLabel() {
    if (!this.state.accuracy) return null;
    return rateAccuracy(this.state.accuracy);
  }

  showFatal(message) {
    const ov = this.openOverlay('solid');
    ov.append(h('div', { class: 'overlay-card' },
      h('h1', {}, 'The eye tracker stopped'),
      h('div', { class: 'error-box' }, message || 'Unknown error'),
      h('p', { class: 'muted', style: { marginTop: '20px' } }, 'Check the terminal running Paralic for details, then reload this page.')));
  }
}

window.paralic = new App();
