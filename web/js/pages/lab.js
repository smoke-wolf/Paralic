// Personalization Lab: everything Paralic has learned about the current
// person's eyes, and blind A/B experiments that tune it further.
//
//   #/lab                    overview: gaze networks, blinks, winks, experiments
//   #/lab/run/<experiment>   one round of blind, randomised target trials
//
// In an experiment round every trial secretly uses one of the arms (e.g. the
// current smoothing, a smoother one and a snappier one) in a shuffled,
// balanced order. You just pop targets; the server compares the arms with a
// permutation test and makes the winner your new default once it is clear.

import { h, toast } from '../dom.js';
import { icon } from '../icons.js';
import { sounds } from '../sound.js';

const TRIALS_PER_ARM = 4;
const TRIAL_TIMEOUT_MS = 10000;
const NETWORK_NAMES = { both: 'Both eyes', left: 'Left eye', right: 'Right eye' };
const SOURCES = { calibration: 'calibration', 'fine-tune': 'fine-tuning', 'fine-tuned': 'fine-tuning',
  retrained: 'model search', adjust: 'quick adjust' };
const DECISIONS = {
  need_more: 'Needs more trials', keep: 'Current setting is best', adopt: 'Winner adopted', inconclusive: 'No clear winner yet',
};

function stat(value, label) {
  return h('div', { class: 'lab-stat' }, h('b', {}, value), h('span', {}, label));
}

function px(v) {
  return v === null || v === undefined ? '–' : `${Math.round(v)} px`;
}

function overview(el, app) {
  const view = app.state.personal;
  const name = app.state.person ? app.state.person.name : 'you';
  const head = h('div', { class: 'page-head' },
    h('div', {},
      h('div', { class: 'eyebrow' }, 'Personalization Lab'),
      h('h1', {}, `Tuned for ${name}`),
      h('p', { class: 'muted' }, 'Everything here was learned from your own eyes on this computer. Look at a button and blink twice to use it.')),
    h('div', { class: 'btn-row' }, button('Switch person', 'head', () => app.switchPerson())));
  el.append(head);
  if (!view) {
    el.append(h('p', { class: 'muted' }, 'Connecting to the eye tracker…'));
    return;
  }

  // -- the gaze networks ----------------------------------------------------------
  const m = view.model;
  const nets = view.eye_models;
  const model = h('section', { class: 'card lab-card' },
    h('h3', { html: `${icon('brain')}<span>Your gaze network</span>` }));
  if (!m) {
    model.append(h('p', { class: 'muted' }, app.state.simulated ? 'Demo mode uses the mouse instead of a gaze network.'
      : 'Not calibrated yet.'));
  } else {
    model.append(h('div', { class: 'lab-stats' },
      stat(`v${m.version ?? '–'}`, `from ${SOURCES[m.source] || m.source || 'calibration'}`),
      stat(px(view.accuracy_px), 'accuracy (validation)'),
      stat(px(m.cv_error_px), 'cross-validated error'),
      stat(m.config || '–', 'architecture')));
    if (nets && nets.available && nets.available.length) {
      const cv = nets.cv_px || {};
      model.append(h('p', { class: 'muted' }, 'Three networks are trained: both eyes, and each eye alone (that one keeps the cursor moving while the other eye winks). The leader is chosen by cross-validation.'),
        h('div', { class: 'lab-nets' }, ['both', 'left', 'right'].map((k) => h('div', {
          class: `lab-net ${nets.preferred === k ? 'lead' : ''}`,
        }, h('b', {}, NETWORK_NAMES[k]), h('span', {}, px(cv[k])), nets.preferred === k ? h('small', {}, 'leads') : null))));
    }
    const learn = view.learning
      ? `${view.ft_new_events} new practice hits and clicks since the last update (${view.ft_events} in total). Fine-tuning runs by itself every ${15} new ones and only keeps a network that beats the current one on your latest data.`
      : 'Learning from use is off (Settings → Keep learning my eyes).';
    model.append(h('p', { class: 'muted' }, learn));
    const tune = button(view.job_running ? 'Training…' : 'Fine-tune now', 'bolt', async () => {
      try {
        const r = await app.tracker.request({ type: 'finetune' }, 'finetune_started', 8000);
        toast(r.ok ? 'Fine-tuning in the background — I’ll tell you how it went' : (r.error || 'Could not start'), r.ok ? 'ok' : 'warn');
      } catch (err) {
        toast(err.message, 'bad');
      }
    }, 'primary');
    if (view.job_running) tune.disabled = true;
    model.append(h('div', { class: 'btn-row' }, tune,
      app.state.simulated ? null : button('Quick adjust', 'crosshair', () => app.calibrate('adjust'))));
    const history = (view.model_history || []).slice(-4).reverse();
    if (history.length) {
      model.append(h('ul', { class: 'lab-history' }, history.map((e) => h('li', {},
        h('b', {}, `v${e.version}`), ` ${SOURCES[e.source] || e.source}`,
        e.error_px !== undefined ? ` · ${Math.round(e.error_px)} px` : '',
        e.accuracy_px !== undefined ? ` · ${Math.round(e.accuracy_px)} px` : '',
        h('span', { class: 'muted' }, ` · ${(e.time || '').replace('T', ' ')}`)))));
    }
  }

  // -- blinks and winks -------------------------------------------------------------
  const b = view.blink || {};
  const bp = view.blink_profile;
  const signals = { both: 'both eyes must close', mean: 'average of both eyes', left: 'left eye', right: 'right eye' };
  const blinks = h('section', { class: 'card lab-card' },
    h('h3', { html: `${icon('blink')}<span>Blinks</span>` }),
    h('div', { class: 'lab-stats' },
      stat(b.personal ? 'Personal' : 'Standard', 'thresholds'),
      stat(`${b.double_gap_ms ?? '–'} ms`, 'double-blink window'),
      stat(signals[view.blink_signal] || '–', 'watching')),
    h('p', { class: 'muted' }, bp ? `Learned from ${bp.n_blinks} blinks (typical blink ${bp.blink_ms} ms, pause ${bp.gap_ms} ms).`
      : 'Run the blink test so double blinks match how you blink.'),
    app.state.simulated ? null : h('div', { class: 'btn-row' }, button('Blink test', 'blink', () => app.calibrate('blink'))));

  const w = view.wink_profile;
  const g = view.gestures || {};
  const eyeLine = (eye) => {
    const r = w && w[eye];
    const on = view.winks ? view.winks[eye] : true;
    const mapping = { drag: 'press & drag', menu: 'right-click menu', off: 'off' }[g[`${eye}_hold`]] || '–';
    return h('li', {}, h('b', {}, eye === 'left' ? 'Left eye: ' : 'Right eye: '),
      !on ? `ignored${r && !r.ok ? ` (${r.reason})` : ''}` : `${mapping}${r && r.ok ? ' · tested ✓' : ''}`);
  };
  const winks = h('section', { class: 'card lab-card' },
    h('h3', { html: `${icon('wink')}<span>Winks and gestures</span>` }),
    h('ul', { class: 'lab-list' }, eyeLine('left'), eyeLine('right'),
      h('li', {}, h('b', {}, 'Both eyes ~1 s: '), { off: 'nothing', menu: 'menu', grab: 'pick up / drop', click: 'click' }[g.long_close] || '–'),
      h('li', {}, h('b', {}, 'Dwell click: '), g.dwell ? `${(g.dwell_ms / 1000).toFixed(1)} s` : 'off')),
    h('div', { class: 'btn-row' },
      app.state.simulated ? null : button('Wink test', 'wink', () => app.calibrate('wink')),
      h('a', { class: 'btn', href: '#/settings', html: `${icon('sliders')}<span>Gesture settings</span>` })));

  // -- experiments ----------------------------------------------------------------------
  const experiments = h('section', { class: 'card lab-card lab-wide' },
    h('h3', { html: `${icon('flask')}<span>A/B experiments</span>` }),
    h('p', { class: 'muted' }, 'Each round is 8–12 targets. Every target secretly uses one variant; you just pop them. When one variant is clearly faster for you (permutation test), it becomes your default.'));
  const list = h('div', { class: 'lab-experiments' });
  const EXP = [
    ['smoothing', 'Cursor smoothing', 'current vs smoother vs snappier'],
    ['magnet', 'Button magnet', 'current vs stronger vs off'],
    ['double_blink', 'Double-blink timing', 'current vs relaxed'],
    ['dwell', 'Dwell-click time', 'current vs faster vs slower'],
  ];
  for (const [id, title, arms] of EXP) {
    if (id === 'dwell' && !g.dwell) continue;
    const r = (view.experiments || {})[id];
    const status = r ? `${DECISIONS[r.decision] || r.decision}${r.best ? ` · best: ${r.best}` : ''}${r.p_value !== null && r.p_value !== undefined ? ` · p = ${r.p_value}` : ''}`
      : 'Not run yet';
    list.append(h('a', { class: 'lab-experiment', href: `#/lab/run/${id}` },
      h('b', {}, title), h('span', {}, arms), h('small', {}, status)));
  }
  experiments.append(list);
  if (app.state.simulated) experiments.append(h('p', { class: 'muted' }, 'In mouse demo mode only the button magnet changes anything.'));

  el.append(h('div', { class: 'lab-grid' }, model, blinks, winks, experiments));
}

function button(label, ic, onClick, cls = '') {
  const b = h('button', { class: `btn ${cls}`, type: 'button', html: `${icon(ic)}<span>${label}</span>` });
  b.addEventListener('click', onClick);
  return b;
}

/** Balanced, shuffled order of arm ids. */
function schedule(arms) {
  const order = [];
  for (let i = 0; i < TRIALS_PER_ARM; i++) order.push(...arms.map((a) => a.id));
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

function runner(el, app, experiment) {
  let cancelled = false;
  let cleanupTrial = () => {};
  const title = h('h1', {}, 'Experiment');
  const status = h('p', { class: 'muted' });
  const arena = h('div', { class: 'arena lab-arena' });
  el.append(h('div', { class: 'hud' },
    h('div', {}, h('div', { class: 'eyebrow' }, 'Personalization Lab'), title, status),
    h('a', { class: 'btn', href: '#/lab', html: `${icon('back')}<span>Back to the Lab</span>` })), arena);

  const restore = () => {
    app.gaze.magnetOverride = null;
    app.gaze.dwellOverride = null;
    app.tracker.send({ type: 'settings', clear_overrides: true });
  };

  const apply = (arm) => {
    app.gaze.magnetOverride = arm.magnet || null;
    app.gaze.dwellOverride = arm.dwell_ms || null;
    const patch = {};
    if (arm.smoothing_level !== undefined) patch.smoothing_level = arm.smoothing_level;
    if (arm.double_gap_ms !== undefined) patch.double_gap_ms = arm.double_gap_ms;
    app.tracker.send({ type: 'settings', clear_overrides: true, ...patch });
  };

  const center = (...children) => {
    arena.innerHTML = '';
    arena.append(h('div', { class: 'arena-center' }, h('div', {}, ...children)));
  };

  /** One trial: show a target somewhere, measure time to pop it and the misses. */
  const trial = (arm) => new Promise((resolve) => {
    apply(arm);
    arena.innerHTML = '';
    const size = 120;
    const w = arena.clientWidth;
    const hgt = arena.clientHeight;
    const x = size / 2 + 20 + Math.random() * Math.max(1, w - size - 40);
    const y = size / 2 + 20 + Math.random() * Math.max(1, hgt - size - 40);
    let misses = 0;
    let shownAt = 0;
    let done = false;
    const target = h('button', {
      class: 'target', type: 'button', 'aria-label': 'Target', 'data-learn': 'practice',
      style: { left: `${x - size / 2}px`, top: `${y - size / 2}px`, width: `${size}px`, height: `${size}px` },
    });
    const finish = (result) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      offActivate();
      target.remove();
      resolve(result);
    };
    target.addEventListener('click', () => {
      sounds.pop();
      finish({ arm: arm.id, time_ms: performance.now() - shownAt, misses, timeout: false });
    });
    const offActivate = app.on('activate', ({ element }) => {
      if (!done && element !== target) misses += 1;
    });
    // Let the new setting settle before the clock starts.
    let timer = setTimeout(() => {
      arena.append(target);
      shownAt = performance.now();
      timer = setTimeout(() => finish({ arm: arm.id, time_ms: TRIAL_TIMEOUT_MS, misses, timeout: true }), TRIAL_TIMEOUT_MS);
    }, 700);
    cleanupTrial = () => finish(null);
  });

  const start = async () => {
    let plan;
    try {
      plan = await app.tracker.request({ type: 'experiment_plan', experiment }, 'experiment_plan', 8000);
      if (!plan.ok) throw new Error(plan.error || 'Unknown experiment');
    } catch (err) {
      center(h('h2', {}, 'Could not load the experiment'), h('p', { class: 'muted' }, err.message));
      return;
    }
    title.textContent = plan.title;
    const a = plan.analysis || {};
    const intro = h('p', { class: 'muted', style: { maxWidth: '50ch', margin: '0 auto 24px' } },
      `${plan.arms.length * TRIALS_PER_ARM} targets. Pop each one the way you normally click (double blink, wink or dwell). `,
      `Each target secretly uses one of ${plan.arms.length} variants. So far: ${a.n_trials || 0} trials this round.`);
    const go = button('Start', 'play', () => run(plan), 'primary');
    center(h('h2', {}, plan.title), intro, h('div', { class: 'btn-row' }, go));
  };

  const run = async (plan) => {
    const arms = Object.fromEntries(plan.arms.map((a) => [a.id, a]));
    const order = schedule(plan.arms);
    const results = [];
    for (let i = 0; i < order.length && !cancelled; i++) {
      status.textContent = `Target ${i + 1} of ${order.length}`;
      const r = await trial(arms[order[i]]);
      if (!r) break;
      results.push(r);
    }
    restore();
    if (cancelled) return;
    status.textContent = '';
    center(h('h2', {}, 'Analysing…'), h('div', { class: 'spinner' }));
    let res;
    try {
      res = await app.tracker.request({ type: 'experiment_log', experiment, trials: results }, 'experiment_result', 15000);
    } catch (err) {
      center(h('h2', {}, 'Could not save the results'), h('p', { class: 'muted' }, err.message));
      return;
    }
    if (res.applied && res.applied.settings) app.adoptSettings(res.applied.settings);
    if (res.decision === 'adopt') sounds.success();
    const rows = Object.entries(res.arms || {}).map(([id, s]) => h('tr', {},
      h('td', {}, (arms[id] && arms[id].label) || id), h('td', {}, String(s.n)),
      h('td', {}, s.median_time_ms ? `${(s.median_time_ms / 1000).toFixed(2)} s` : '–'),
      h('td', {}, s.misses_per_trial !== null && s.misses_per_trial !== undefined ? s.misses_per_trial.toFixed(2) : '–')));
    center(
      h('h2', {}, DECISIONS[res.decision] || res.decision),
      h('p', { class: 'muted' }, res.decision === 'adopt'
        ? `“${(arms[res.best] && arms[res.best].label) || res.best}” was clearly better for you and is now your default.`
        : res.decision === 'need_more' ? 'A few more rounds will tell the variants apart.'
          : res.decision === 'keep' ? 'Your current setting did best — keeping it.'
            : 'The difference is too small to be sure yet. Another round helps.'),
      h('table', { class: 'lab-table' },
        h('thead', {}, h('tr', {}, h('th', {}, 'Variant'), h('th', {}, 'Targets'), h('th', {}, 'Median time'), h('th', {}, 'Misses per target'))),
        h('tbody', {}, rows)),
      h('div', { class: 'btn-row' }, button('Another round', 'refresh', () => start(), 'primary'),
        h('a', { class: 'btn', href: '#/lab', html: `${icon('back')}<span>Back to the Lab</span>` })));
  };

  start();
  return () => {
    cancelled = true;
    cleanupTrial();
    restore();
  };
}

export default {
  title: 'Personalization Lab',
  render(el, params, app) {
    if (params[0] === 'run' && params[1]) return runner(el, app, params[1]);
    const rerender = () => {
      el.innerHTML = '';
      overview(el, app);
    };
    overview(el, app);
    app.tracker?.send({ type: 'personal_get' });
    const offs = [app.on('personal', rerender), app.on('people', rerender)];
    return () => offs.forEach((off) => off());
  },
};
