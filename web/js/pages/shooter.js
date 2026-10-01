// Trail Shooter: walk a trail through the night and stop the creatures that
// come at you before they reach you. Look at one to aim and blink to shoot -
// with a hand, point and pinch. Each level is a preset trail (its bends, its
// hills and where the creatures wait are always the same, so a level can be
// practised) and the levels get faster and busier.
//
// Drawn on a canvas in pseudo-3D: the trail is a row of short segments, each
// projected with a simple perspective (scale = focal / distance), painted from
// far to near. The aim is the gaze cursor's position (app.gaze.point); the
// page's own cursor hides while playing and a sight is drawn instead. While
// playing, blinks shoot instead of clicking; to pause, rest the sight on the
// pause button, close your eyes for a second, hold up an open hand (hand
// mode), or press P / Esc.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { sounds } from '../sound.js';
import { handMode, say, sayHtml } from '../mode.js';

const DRAW = 140;            // segments drawn ahead
const FOCAL = 0.85;
const CAM_HEIGHT = 1.6;      // world units above the trail (the trail is 2 wide)
const HEARTS = 5;
const PAUSE_DWELL_MS = 1200;
const STORE_KEY = 'paralic.shooter';

// Preset trails: sections of [segments, bend, hill]. Creatures wait every
// `gap` segments (give or take), and walk towards you at `rush` segments/s.
const LEVELS = [
  { name: 'Forest path', speed: 8, gap: [70, 110], rush: 3, hp: 1, size: 1.25,
    track: [[60, 0, 0], [80, 0.6, 0], [60, 0, 2], [90, -0.8, 0], [60, 0, -2], [80, 0.5, 1], [120, 0, 0], [80, -0.5, -1],
      [100, 0.7, 0], [80, 0, 0]] },
  { name: 'River valley', speed: 10, gap: [55, 90], rush: 4, hp: 1, size: 1.15,
    track: [[50, 0, 0], [70, 1.0, 1], [70, -1.0, -1], [60, 0, 3], [90, 0.8, -3], [70, -1.2, 0], [80, 0, 2], [70, 1.1, -2],
      [90, -0.6, 0], [100, 0, 0]] },
  { name: 'Mountain pass', speed: 12, gap: [45, 75], rush: 5, hp: 2, size: 1.1,
    track: [[40, 0, 2], [60, 1.5, 4], [60, -1.6, -2], [50, 0, 5], [70, 1.8, -5], [60, -1.4, 3], [70, 0, -3], [60, 1.6, 2],
      [80, -1.8, -2], [100, 0, 0]] },
  { name: 'Night ridge', speed: 14, gap: [35, 60], rush: 6, hp: 2, size: 1.05,
    track: [[40, 0, 0], [60, 2.0, 3], [50, -2.0, -3], [60, 1.2, 5], [50, -1.5, -5], [70, 2.2, 2], [60, -2.2, -2], [60, 0, 4],
      [80, 1.6, -4], [120, 0, 0]] },
];

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/** The level's trail: centre line x, height y per segment, and where creatures wait. */
function buildLevel(index) {
  const L = LEVELS[index];
  const xs = [0];
  const ys = [0];
  let dx = 0;
  let bend = 0;
  let hill = 0;
  for (const [n, curve, rise] of L.track) {
    for (let k = 0; k < n; k++) {
      // Ease into and out of each bend and hill.
      const f = Math.sin((Math.PI * (k + 0.5)) / n);
      bend = curve * 0.0016 * f;
      hill = (rise * 0.012) * f;
      dx += bend;
      xs.push(xs[xs.length - 1] + dx);
      ys.push(ys[ys.length - 1] + hill);
    }
  }
  const r = rng(1009 * (index + 1));
  const foes = [];
  for (let z = 95; z < xs.length - 40;) {
    foes.push({ z, home: z, off: (r() * 2 - 1) * 0.75, hp: L.hp, phase: r() * 6.28, alive: true, hit: 0 });
    z += L.gap[0] + r() * (L.gap[1] - L.gap[0]);
  }
  const trees = [];
  for (let i = 4; i < xs.length; i += 3 + Math.floor(r() * 3)) {
    trees.push({ i, side: r() < 0.5 ? -1 : 1, off: 1.6 + r() * 2.5, size: 0.8 + r() * 0.9 });
  }
  return { ...L, index, xs, ys, foes, trees, length: xs.length - 1 };
}

function loadBest() {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) || '{}') || {};
  } catch {
    return {};
  }
}

function saveBest(best) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(best));
  } catch {
    /* private window: the best scores are not kept */
  }
}

export default {
  title: 'Trail Shooter',
  render(el, params, app) {
    const best = loadBest();
    let levelIndex = Math.min(Number(best.unlocked || 0), LEVELS.length - 1);
    let shootWith = best.shootWith === 'double' ? 'double' : 'blink';
    let aimHelp = best.aimHelp === 'large' ? 'large' : 'normal';

    const stats = h('div', { class: 'stats' });
    const pauseBtn = h('button', { class: 'btn shooter-pause', type: 'button', html: `${icon('pause')}<span>Pause</span>` });
    const canvas = h('canvas', { class: 'shooter-canvas' });
    const arena = h('div', { class: 'arena shooter-arena' }, canvas);
    el.append(
      h('div', { class: 'hud' },
        h('div', {}, h('div', { class: 'eyebrow' }, 'Game'), h('h2', { style: { margin: 0 } }, 'Trail Shooter')),
        h('div', { class: 'stats-row' }, stats, pauseBtn)),
      arena);
    const ctx = canvas.getContext('2d');

    let game = null;            // the level being played
    let state = 'menu';         // menu | playing | paused | over
    let raf = 0;
    let last = 0;
    let pauseSince = null;
    let pausedByApp = false;     // paused because the whole app paused (see below)

    // -- screens over the canvas ---------------------------------------------------------
    const overlay = (...children) => {
      arena.querySelector('.arena-center')?.remove();
      const box = h('div', { class: 'arena-center shooter-menu' }, h('div', {}, ...children));
      arena.append(box);
      return box;
    };
    const button = (label, ic, onClick, primary = false) => {
      const b = h('button', { class: `btn ${primary ? 'primary' : ''}`, type: 'button', html: `${icon(ic)}<span>${label}</span>` });
      b.addEventListener('click', onClick);
      return b;
    };
    const choice = (label, options, current, onPick) => h('div', { class: 'shooter-option eyes-only' },
      h('span', { class: 'muted' }, label),
      ...options.map(([value, text]) => {
        const b = h('button', { class: `btn chip ${value === current ? 'active' : ''}`, type: 'button' }, text);
        b.addEventListener('click', () => { onPick(value); menu(); });
        return b;
      }));

    const menu = () => {
      state = 'menu';
      setAiming(false);
      const unlocked = Number(best.unlocked || 0);
      overlay(
        h('h1', {}, 'Trail Shooter'),
        h('p', { class: 'muted', style: { maxWidth: '50ch', margin: '0 auto 18px' },
          html: sayHtml('Walk the trail and stop the creatures before they reach you. Look at one and blink to shoot. Rest your eyes on Pause (or close them for a second) to take a break.',
            'Walk the trail and stop the creatures before they reach you. Point at one and pinch to shoot. Hold up an open hand to take a break.') }),
        h('div', { class: 'btn-row shooter-levels' }, LEVELS.map((L, i) => {
          const b = button(`${i + 1}. ${L.name}${best[`level${i}`] ? ` · best ${best[`level${i}`]}` : ''}`,
            i <= unlocked ? 'play' : 'pause', () => start(i), i === levelIndex);
          b.disabled = i > unlocked;
          return b;
        })),
        choice('Shoot with:', [['blink', 'One blink'], ['double', 'Two blinks']], shootWith, (v) => {
          shootWith = v;
          best.shootWith = v;
          saveBest(best);
        }),
        h('div', { class: 'shooter-option' },
          h('span', { class: 'muted' }, 'Aim help:'),
          ...[['normal', 'Normal'], ['large', 'Large']].map(([v, text]) => {
            const b = h('button', { class: `btn chip ${v === aimHelp ? 'active' : ''}`, type: 'button' }, text);
            b.addEventListener('click', () => { aimHelp = v; best.aimHelp = v; saveBest(best); menu(); });
            return b;
          })));
      draw(0);
    };

    // -- playing ---------------------------------------------------------------------------
    const start = (i) => {
      levelIndex = i;
      game = buildLevel(i);
      Object.assign(game, { cam: 0, hearts: HEARTS, score: 0, shots: 0, hits: 0, combo: 0, sparks: [], beams: [],
        flash: 0, started: performance.now() });
      arena.querySelector('.arena-center')?.remove();
      state = 'playing';
      setAiming(true);
      renderStats();
      sounds.point?.();
    };

    const resume = () => {
      pausedByApp = false;
      if (app.gaze && app.gaze.paused) app.gaze.setPaused(false);
      arena.querySelector('.arena-center')?.remove();
      state = 'playing';
      setAiming(true);
      last = performance.now();
    };

    const pause = () => {
      if (state !== 'playing') return;
      state = 'paused';
      setAiming(false);
      overlay(
        h('h1', {}, 'Paused'),
        h('div', { class: 'btn-row' },
          button('Resume', 'play', resume, true),
          button('Restart level', 'refresh', () => start(levelIndex)),
          button('Levels', 'grid', menu)));
    };

    const finish = (won) => {
      state = 'over';
      setAiming(false);
      const accuracy = game.shots ? Math.round((100 * game.hits) / game.shots) : 0;
      const key = `level${game.index}`;
      const record = game.score > (best[key] || 0);
      if (won) {
        best[key] = Math.max(best[key] || 0, game.score);
        best.unlocked = Math.max(Number(best.unlocked || 0), Math.min(game.index + 1, LEVELS.length - 1));
        saveBest(best);
        sounds.success?.();
      }
      const next = won && game.index + 1 < LEVELS.length;
      overlay(
        h('h1', {}, won ? `${game.name} cleared!` : 'The creatures got through'),
        h('p', { class: 'muted' }, `Score ${game.score}${record && won ? ' — a new best!' : ''} · ${game.hits} hits · accuracy ${accuracy}%`),
        h('div', { class: 'btn-row' },
          next ? button(`Next: ${LEVELS[game.index + 1].name}`, 'arrowRight', () => start(game.index + 1), true) : null,
          button(won ? 'Play again' : 'Try again', 'refresh', () => start(game.index), !next),
          button('Levels', 'grid', menu)));
    };

    const renderStats = () => {
      stats.innerHTML = '';
      if (!game) return;
      const progress = Math.min(1, game.cam / game.length);
      stats.append(
        h('span', { class: 'pill' }, `Level ${game.index + 1} · ${game.name}`),
        h('span', { class: 'pill ok' }, `Score ${game.score}`),
        h('span', { class: 'pill warn', 'aria-label': `${game.hearts} lives` }, '❤️'.repeat(Math.max(0, game.hearts)) || '–'),
        h('span', { class: 'pill' }, `${Math.round(progress * 100)}% of the trail`));
    };

    // The page's cursor hides while playing: the sight is drawn instead.
    function setAiming(on) {
      document.body.classList.toggle('game-aiming', on);
    }

    // -- the aim and the shots -----------------------------------------------------------------
    const aimPoint = () => {
      const p = app.gaze && app.gaze.point;
      if (!p) return null;
      const r = canvas.getBoundingClientRect();
      return { x: p.x - r.left, y: p.y - r.top, inside: p.x >= r.left && p.x <= r.right && p.y >= r.top && p.y <= r.bottom };
    };

    let lastShot = 0;
    const shoot = () => {
      if (state !== 'playing' || !game) return;
      const now = performance.now();
      if (now - lastShot < 120) return;
      lastShot = now;
      const aim = aimPoint();
      if (!aim || !aim.inside) return;
      game.shots++;
      const assist = (aimHelp === 'large' ? 110 : 70) * (canvas.width / canvas.clientWidth || 1);
      const x = aim.x * (canvas.width / canvas.clientWidth);
      const y = aim.y * (canvas.height / canvas.clientHeight);
      let target = null;
      let bestD = Infinity;
      for (const f of game.foes) {
        if (!f.alive || !f.screen) continue;
        const d = Math.hypot(f.screen.x - x, f.screen.y - y) - f.screen.r;
        if (d <= assist && d < bestD) {
          bestD = d;
          target = f;
        }
      }
      game.beams.push({ x, y, t: now });
      if (target) {
        target.hp -= 1;
        target.hit = now;
        game.hits++;
        if (target.hp <= 0) {
          target.alive = false;
          game.combo++;
          game.score += 100 * Math.min(5, game.combo);
          burst(target.screen.x, target.screen.y, target.screen.r);
          sounds.click?.();
        } else {
          sounds.point?.();
        }
      } else {
        game.combo = 0;
        sounds.miss?.();
      }
      renderStats();
    };

    const burst = (x, y, r) => {
      for (let k = 0; k < 26; k++) {
        const a = Math.random() * Math.PI * 2;
        const v = (0.4 + Math.random()) * r * 3;
        game.sparks.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 0.7 + Math.random() * 0.4 });
      }
    };

    // -- one frame -----------------------------------------------------------------------------
    const step = (now) => {
      raf = requestAnimationFrame(step);
      bind();
      const dt = Math.min(0.05, Math.max(0, (now - (last || now)) / 1000));
      last = now;
      resize();
      if (state === 'playing' && game) {
        if (app.gaze && app.gaze.paused) {
          pause();
          pausedByApp = true;
        }
        game.cam += game.speed * dt;
        for (const f of game.foes) {
          if (!f.alive) continue;
          // Creatures wait until you come near, then rush towards you.
          if (f.z - game.cam < DRAW * 0.7) f.z -= game.rush * dt;
          if (f.z - game.cam < 1.2) {
            f.alive = false;
            game.hearts -= 1;
            game.combo = 0;
            game.flash = 1;
            sounds.miss?.();
            renderStats();
          }
        }
        if (game.hearts <= 0) finish(false);
        else if (game.cam >= game.length - 2) finish(true);
        else if (Math.floor(game.cam) % 10 === 0) renderStats();
        updatePauseDwell(now);
      }
      draw(dt);
    };

    const updatePauseDwell = (now) => {
      const p = app.gaze && app.gaze.point;
      const r = pauseBtn.getBoundingClientRect();
      const on = p && p.x >= r.left - 10 && p.x <= r.right + 10 && p.y >= r.top - 10 && p.y <= r.bottom + 10;
      if (!on) {
        pauseSince = null;
        pauseBtn.style.removeProperty('--p');
        return;
      }
      pauseSince = pauseSince ?? now;
      const frac = Math.min(1, (now - pauseSince) / PAUSE_DWELL_MS);
      pauseBtn.style.setProperty('--p', frac.toFixed(2));
      if (frac >= 1) {
        pauseSince = null;
        pause();
      }
    };

    const resize = () => {
      const dpr = Math.min(1.5, window.devicePixelRatio || 1);
      const w = Math.round(canvas.clientWidth * dpr);
      const hgt = Math.round(canvas.clientHeight * dpr);
      if (w && hgt && (canvas.width !== w || canvas.height !== hgt)) {
        canvas.width = w;
        canvas.height = hgt;
      }
    };

    // -- drawing -------------------------------------------------------------------------------
    const draw = (dt) => {
      const W = canvas.width;
      const H = canvas.height;
      if (!W || !H) return;
      const horizon = H * 0.42;
      const sky = ctx.createLinearGradient(0, 0, 0, horizon);
      sky.addColorStop(0, '#050816');
      sky.addColorStop(1, '#1f2b52');
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, W, H);
      // Stars and the moon (fixed).
      const r = rng(7);
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      for (let k = 0; k < 90; k++) ctx.fillRect(r() * W, r() * horizon * 0.95, 1.5, 1.5);
      ctx.fillStyle = '#e8ecff';
      ctx.beginPath();
      ctx.arc(W * 0.8, H * 0.12, H * 0.045, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#0b1530';
      ctx.fillRect(0, horizon, W, H - horizon);

      const g = game || buildMenuScene();
      const cam = g.cam || 0;
      const base = Math.floor(cam);
      const frac = cam - base;
      const camX = lerpAt(g.xs, cam);
      const camY = lerpAt(g.ys, cam) + CAM_HEIGHT;
      const project = (wx, wy, dz) => {
        const s = FOCAL / Math.max(0.05, dz);
        return { x: W / 2 + (wx - camX) * s * (W / 2), y: horizon - (wy - camY) * s * (W / 2) * 0.62, s };
      };
      // Trees and creatures, by segment, to paint with the trail from far to near.
      const sprites = new Map();
      for (const t of g.trees) {
        if (t.i > base && t.i < base + DRAW) (sprites.get(t.i) || sprites.set(t.i, []).get(t.i)).push({ tree: t });
      }
      if (game) {
        for (const f of g.foes) {
          f.screen = null;
          const i = Math.floor(f.z);
          if (f.alive && f.z > cam + 0.8 && i < base + DRAW) (sprites.get(i) || sprites.set(i, []).get(i)).push({ foe: f });
        }
      }
      for (let i = Math.min(g.length - 1, base + DRAW); i > base; i--) {
        const dzFar = i + 1 - cam;
        const dzNear = Math.max(0.3, i - cam);
        const a = project(g.xs[i], g.ys[i], dzNear);
        const b = project(g.xs[i + 1] ?? g.xs[i], g.ys[i + 1] ?? g.ys[i], dzFar);
        const wa = a.s * (W / 2);
        const wb = b.s * (W / 2);
        const fog = Math.min(1, dzNear / DRAW);
        const stripe = i % 6 < 3;
        ctx.fillStyle = shade(stripe ? [22, 52, 44] : [18, 44, 38], fog);
        ctx.fillRect(0, b.y, W, Math.max(1, a.y - b.y + 1));
        quad(b.x - wb * 1.18, b.y, b.x + wb * 1.18, a.x + wa * 1.18, a.y, a.x - wa * 1.18, shade([94, 234, 212], fog * 0.9 + 0.1, 0.55));
        quad(b.x - wb, b.y, b.x + wb, a.x + wa, a.y, a.x - wa, shade(stripe ? [52, 60, 84] : [46, 54, 76], fog));
        if (i % 8 < 4) quad(b.x - wb * 0.04, b.y, b.x + wb * 0.04, a.x + wa * 0.04, a.y, a.x - wa * 0.04, shade([230, 230, 255], fog, 0.6));
        for (const sp of sprites.get(i) || []) {
          if (sp.tree) drawTree(sp.tree, g, project, i, cam, fog);
          else drawFoe(sp.foe, g, project, cam);
        }
      }
      if (!game) return;
      // Sparks, beams, the damage flash and the sight.
      ctx.globalCompositeOperation = 'lighter';
      for (const s of game.sparks) {
        s.life -= dt;
        s.x += s.vx * dt;
        s.y += s.vy * dt;
        s.vy += 400 * dt;
        if (s.life > 0) {
          ctx.fillStyle = `rgba(196, 181, 253, ${Math.min(1, s.life)})`;
          ctx.fillRect(s.x - 3, s.y - 3, 6, 6);
        }
      }
      game.sparks = game.sparks.filter((s) => s.life > 0);
      const now = performance.now();
      for (const bm of game.beams) {
        const age = (now - bm.t) / 160;
        if (age > 1) continue;
        ctx.strokeStyle = `rgba(94, 234, 212, ${1 - age})`;
        ctx.lineWidth = 6 * (1 - age) + 1;
        ctx.beginPath();
        ctx.moveTo(W / 2, H);
        ctx.lineTo(bm.x, bm.y);
        ctx.stroke();
      }
      game.beams = game.beams.filter((bm) => now - bm.t < 160);
      ctx.globalCompositeOperation = 'source-over';
      if (game.flash > 0) {
        ctx.fillStyle = `rgba(248, 113, 113, ${0.35 * game.flash})`;
        ctx.fillRect(0, 0, W, H);
        game.flash = Math.max(0, game.flash - dt * 2.5);
      }
      if (state === 'playing') drawSight();
    };

    function lerpAt(arr, z) {
      const i = Math.max(0, Math.min(arr.length - 2, Math.floor(z)));
      const f = Math.min(1, Math.max(0, z - i));
      return arr[i] * (1 - f) + arr[i + 1] * f;
    }

    let menuScene = null;
    function buildMenuScene() {
      if (!menuScene) menuScene = { ...buildLevel(levelIndex), cam: 30 };
      menuScene.cam += 0.05;
      if (menuScene.cam > menuScene.length - DRAW - 2) menuScene.cam = 30;
      return menuScene;
    }

    function shade([r, gr, b], fog, alpha = 1) {
      const k = 1 - 0.75 * fog;
      return `rgba(${Math.round(r * k + 11 * (1 - k))}, ${Math.round(gr * k + 21 * (1 - k))}, ${Math.round(b * k + 48 * (1 - k))}, ${alpha})`;
    }

    function quad(x1, y1, x2, x3, y3, x4, colour) {
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x3, y3);
      ctx.lineTo(x4, y3);
      ctx.closePath();
      ctx.fill();
    }

    function drawTree(t, g, project, i, cam, fog) {
      const p = project(g.xs[i] + t.side * t.off, g.ys[i], Math.max(0.3, i - cam));
      const s = p.s * (canvas.width / 2) * t.size;
      ctx.fillStyle = shade([16, 70, 58], fog);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y - s * 2.6);
      ctx.lineTo(p.x + s * 0.8, p.y - s * 0.3);
      ctx.lineTo(p.x - s * 0.8, p.y - s * 0.3);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = shade([60, 40, 30], fog);
      ctx.fillRect(p.x - s * 0.1, p.y - s * 0.3, s * 0.2, s * 0.3);
    }

    function drawFoe(f, g, project, cam) {
      const t = performance.now() / 1000;
      const wobble = Math.sin(t * 3 + f.phase) * 0.15;
      const p = project(lerpAt(g.xs, f.z) + f.off + wobble, lerpAt(g.ys, f.z) + 1.1, f.z - cam);
      const r = Math.max(9 * (canvas.width / canvas.clientWidth || 1), p.s * (canvas.width / 2) * 0.55 * g.size);
      f.screen = { x: p.x, y: p.y, r };
      const hurt = performance.now() - f.hit < 150;
      const glow = ctx.createRadialGradient(p.x, p.y, r * 0.2, p.x, p.y, r * 1.6);
      glow.addColorStop(0, hurt ? 'rgba(255,255,255,0.9)' : 'rgba(167,139,250,0.85)');
      glow.addColorStop(1, 'rgba(167,139,250,0)');
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r * 1.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = hurt ? '#ffffff' : (f.hp > 1 ? '#7c3aed' : '#a78bfa');
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
      // Eyes that look at you.
      ctx.fillStyle = '#fff';
      for (const side of [-1, 1]) {
        ctx.beginPath();
        ctx.arc(p.x + side * r * 0.35, p.y - r * 0.15, r * 0.22, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = '#111827';
      for (const side of [-1, 1]) {
        ctx.beginPath();
        ctx.arc(p.x + side * r * 0.35, p.y - r * 0.1, r * 0.1, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function drawSight() {
      const aim = aimPoint();
      if (!aim || !aim.inside) return;
      const k = canvas.width / canvas.clientWidth;
      const x = aim.x * k;
      const y = aim.y * k;
      const assist = (aimHelp === 'large' ? 110 : 70) * k;
      const locked = game.foes.some((f) => f.alive && f.screen && Math.hypot(f.screen.x - x, f.screen.y - y) - f.screen.r <= assist);
      ctx.strokeStyle = locked ? 'rgba(251, 191, 36, 0.95)' : 'rgba(94, 234, 212, 0.9)';
      ctx.lineWidth = 3 * k;
      ctx.beginPath();
      ctx.arc(x, y, 22 * k, 0, Math.PI * 2);
      ctx.moveTo(x - 34 * k, y);
      ctx.lineTo(x - 12 * k, y);
      ctx.moveTo(x + 12 * k, y);
      ctx.lineTo(x + 34 * k, y);
      ctx.moveTo(x, y - 34 * k);
      ctx.lineTo(x, y - 12 * k);
      ctx.moveTo(x, y + 12 * k);
      ctx.lineTo(x, y + 34 * k);
      ctx.stroke();
    }

    // -- controls ------------------------------------------------------------------------------
    // The tracker may not exist yet (the page opened from a link starts before
    // the camera) or may be replaced: listen to whichever is current.
    const offs = [];
    let bound = { tracker: null, gaze: null, offs: [] };
    const bind = () => {
      if (bound.tracker === app.tracker && bound.gaze === app.gaze) return;
      bound.offs.forEach((off) => off && off());
      bound = { tracker: app.tracker, gaze: app.gaze, offs: [] };
      if (app.tracker) {
        bound.offs.push(app.tracker.on('blink', (m) => {
          if (state !== 'playing') return;
          if (m.hand) {
            if (m.n === 1) shoot();        // a pinch starts: fire at once
          } else if (shootWith === 'blink') {
            shoot();                       // every blink is a shot
          }
        }));
        bound.offs.push(app.tracker.on('long_close', () => { if (state === 'playing') pause(); }));
      }
      if (app.gaze) {
        // While playing, a double blink (or a pinch's click) shoots or does nothing - it never clicks the page.
        bound.offs.push(app.gaze.onDoubleBlinkFirst((msg) => {
          if (state !== 'playing') return false;
          if (!msg.hand && shootWith === 'double') shoot();
          return true;
        }));
      }
    };
    bind();
    offs.push(() => bound.offs.forEach((off) => off && off()));
    // The app's own pause (its button, P, a double blink to resume, the open
    // hand) pauses the game too, and resuming it resumes the game.
    offs.push(app.on('pausechange', (paused) => {
      if (paused && state === 'playing') {
        pause();
        pausedByApp = true;
      } else if (!paused && state === 'paused' && pausedByApp) {
        pausedByApp = false;
        resume();
      }
    }));
    const onKey = (e) => {
      if (e.key === 'Escape' && state === 'playing') pause();
      else if (e.key === ' ' && state === 'playing') {
        e.preventDefault();
        shoot();                           // keyboard and mouse players
      }
    };
    window.addEventListener('keydown', onKey);
    canvas.addEventListener('click', () => shoot());
    pauseBtn.addEventListener('click', () => (state === 'playing' ? pause() : state === 'paused' ? resume() : null));
    const onHide = () => { if (document.hidden) pause(); };
    document.addEventListener('visibilitychange', onHide);

    // For tests and the curious (the browser console): what is going on.
    arena.shooter = { get state() { return state; }, get game() { return game; }, shoot };

    menu();
    raf = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(raf);
      setAiming(false);
      offs.forEach((off) => off && off());
      window.removeEventListener('keydown', onKey);
      document.removeEventListener('visibilitychange', onHide);
    };
  },
};

// For tests: the levels' trails are fixed.
export { LEVELS, buildLevel };
