// Paralic — application bootstrap and top-level flows (start screen, face
// check, calibration, pause, navigation).

import { $, $$, h, toast } from './dom.js';
import { icon } from './icons.js';
import { EyeTracker, SimTracker } from './tracker.js';
import { GazeController } from './gaze.js';
import { Calibrator, rateAccuracy } from './calibration.js';
import { CameraPanel } from './camera-panel.js';
import { getSettings, onSettingsChange, serverSettings } from './settings.js';
import { unlockAudio } from './sound.js';
import { Router } from './router.js';

import home from './pages/home.js';
import explore from './pages/explore.js';
import read from './pages/read.js';
import talk from './pages/talk.js';
import practice from './pages/practice.js';
import settingsPage from './pages/settings.js';
import help from './pages/help.js';

const ROUTES = { home, explore, read, talk, practice, settings: settingsPage, help };
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
    this.state = { started: false, calibrated: false, accuracy: null, simulated: false, profile: null };

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
        b.innerHTML = `${icon(c.icon || 'check')}<span>${c.label}</span>${c.sub ? `<small>${c.sub}</small>` : ''}`;
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
  createTracker(simulated) {
    if (this.gaze) this.gaze.destroy();
    const frameWidth = Number(params.get('fw')) || 960;
    this.tracker = simulated ? new SimTracker() : new EyeTracker({ frameWidth });
    this.state.simulated = simulated;
    this.gaze = new GazeController(this.tracker, { cursorEl: $('#gaze-cursor'), pageEl: this.pageEl, railEl: $('#scroll-rail') });
    this.gaze.addEventListener('activate', (e) => this.emit('activate', e.detail));
    this.gaze.addEventListener('pausechange', (e) => {
      this.updatePauseUi(e.detail);
      this.emit('pausechange', e.detail);
    });
    this.camera = new CameraPanel(this.tracker);
    this.tracker.on('fatal', (m) => this.showFatal(m.error));
    this.tracker.on('connection', ({ connected }) => {
      if (!connected && this.state.started) toast('Lost connection to the eye tracker — reconnecting…', 'warn');
      if (connected && this.state.started && this.state.calibrated && !simulated) this.reloadProfileAfterReconnect();
    });
    this.tracker.on('hello', (m) => {
      this.state.hello = m;
      this.state.profile = m.profile || null;
    });
  }

  async reloadProfileAfterReconnect() {
    // The server lost its in-memory model (e.g. it was restarted): reload the saved one.
    try {
      const p = await this.tracker.request({ type: 'profile_load' }, 'profile', 15000);
      if (p.loaded) toast('Reconnected', 'ok');
      else {
        this.state.calibrated = false;
        this.gaze.setActive(false);
        toast('Reconnected — please calibrate again (press C or use Settings)', 'warn', 6000);
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
          h('div', { class: 'start-actions' }, startBtn, demoBtn),
          h('ul', { class: 'start-list' }, tips.map(([ic, text]) => h('li', { html: `${icon(ic)}<span>${text}</span>` }))),
          errorBox),
        h('div', { class: 'start-visual' }, h('div', { class: 'start-eye' }))));
    ov.append(card);

    fetch('/api/status').then((r) => r.json()).then((status) => {
      if (status.error) {
        errorBox.hidden = false;
        errorBox.textContent = `The eye tracker could not start:\n${status.error}\n\nYou can still try the website with a mouse.`;
        startBtn.disabled = true;
      }
    }).catch(() => {});

    startBtn.addEventListener('click', () => {
      unlockAudio();
      if (!params.has('nofs')) document.documentElement.requestFullscreen?.().catch(() => {});
      this.startTracking(ov, card, errorBox);
    });
    demoBtn.addEventListener('click', () => {
      unlockAudio();
      this.startDemo(ov);
    });

    if (params.has('demo')) {
      this.startDemo(ov);
      return;
    }
    // If the camera is already allowed, start right away (no click needed).
    navigator.permissions?.query({ name: 'camera' }).then((p) => {
      if (p.state === 'granted' && !this.state.started && ov.isConnected && !params.has('manual')) {
        this.startTracking(ov, card, errorBox);
      }
    }).catch(() => {});
  }

  async startTracking(ov, card, errorBox) {
    if (this.state.started) return;
    this.state.started = true;
    errorBox.hidden = true;
    if (!this.tracker || this.tracker.simulated) this.createTracker(false);
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
    this.createTracker(true);
    this.tracker.start();
    this.state.calibrated = true;
    this.gaze.setActive(true);
    this.closeOverlay(ov);
    this.camera.setVisible(false);
    toast('Demo mode: the mouse is your gaze. Press B twice quickly to "double blink".', 'ok', 7000);
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
    const preview = this.makePreview();
    const status = h('p', { class: 'big-status' }, 'Looking for your face…');
    const hint = h('p', { class: 'muted' }, 'Centre your face in the oval.');
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
          status.textContent = 'Looking for your face…';
          hint.textContent = 'Centre your face in the oval and make sure it is well lit.';
          preview.guide.classList.remove('ok');
          return;
        }
        preview.guide.classList.add('ok');
        const [yaw, pitch] = m.head || [0, 0];
        if (m.dist && m.dist < 30) hint.textContent = 'You are quite close — lean back a little.';
        else if (m.dist && m.dist > 95) hint.textContent = 'You are far away — move a little closer.';
        else if (Math.abs(yaw) > 22 || Math.abs(pitch) > 22) hint.textContent = 'Face the screen straight on.';
        else hint.textContent = 'Great — hold still…';
        status.textContent = 'Face found';
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

    if (hello && hello.profile) {
      const p = await this.tracker.request({ type: 'profile_load' }, 'profile', 15000).catch(() => ({ loaded: false }));
      if (p.loaded) {
        this.state.calibrated = true;
        this.state.accuracy = p.accuracy_px || null;
        this.gaze.setActive(true);
        const choice = await this.choose(ov, {
          title: 'Welcome back!',
          subtitle: 'Your saved calibration is loaded. Look at an option and blink twice.',
          choices: [
            { id: 'browse', label: 'Start browsing', sub: 'Use the saved calibration as is', icon: 'arrowRight' },
            { id: 'adjust', label: 'Quick adjust', sub: '5 dots, about 8 seconds (recommended)', icon: 'crosshair', primary: true },
            { id: 'full', label: 'Full calibration', sub: 'About 30 seconds', icon: 'refresh' },
          ],
        });
        this.closeOverlay(ov);
        if (choice !== 'browse') await this.calibrate(choice);
        this.welcome();
        return;
      }
    }
    await this.askToCalibrate(ov);
  }

  /** First-time calibration prompt; a double blink works before calibration. */
  async askToCalibrate(ov) {
    const preview = this.makePreview();
    preview.guide.classList.add('ok');
    const btn = h('button', { class: 'btn primary', type: 'button' });
    btn.innerHTML = `${icon('crosshair')}<span>Start calibration</span>`;
    const feedback = h('span', { class: 'pill', style: { fontSize: '1rem', padding: '10px 18px' } },
      'Waiting for two quick blinks…');
    const card = h('div', { class: 'overlay-card' },
      h('div', { class: 'face-check' },
        preview.el,
        h('div', {},
          h('div', { class: 'eyebrow' }, 'One more step'),
          h('h1', {}, 'Blink twice to calibrate'),
          h('p', { class: 'muted', style: { fontSize: '1.15rem' } },
            'A dot will move around the screen. Follow it with your eyes — it takes about 30 seconds and teaches the neural network how your eyes look at your screen.'),
          h('p', {}, feedback),
          btn)));
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
    await new Promise((resolve) => {
      const off = this.gaze.onDoubleBlinkFirst(() => {
        off();
        resolve();
        return true;
      });
      btn.addEventListener('click', () => {
        off();
        resolve();
      });
    });
    offBlink();
    offExpired();
    preview.remove();
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
    if (mode === 'adjust' && !this.state.calibrated) mode = 'full';
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
      this.emit('calibrated', result);
    }
    return result;
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
