// Camera panel: the recorded camera image with what the trackers found in it.
//
// Eye mode: the 478-point face mesh, the eye contours and irises (with each
// eye's axis between its corners), the face box, other faces' boxes and the
// head pose's axes. Hand mode: the 21-point skeleton, the pinch line and the
// fingertip that moves the cursor. Without an image the points are drawn alone.
//
// Eye-mode landmarks are in pixels of the frame MediaPipe analysed, which may
// be larger than the recorded image; the server measures that frame's size.

import { VideoFrames } from './api.js';
import { C, EYE, GAZE, S, alpha, fitCanvas, h, isNum, num } from './util.js';

const EYE_OPTIONS = [
  ['video', 'Video', true], ['mesh', 'Mesh (478 points)', true], ['eyes', 'Eyes & irises', true],
  ['box', 'Face box', true], ['others', 'Other faces', true], ['pose', 'Head pose', true], ['mirror', 'Mirror', false],
];
const HAND_OPTIONS = [
  ['video', 'Video', true], ['skeleton', 'Skeleton', true], ['pinch', 'Pinch line', true], ['tip', 'Fingertip', true],
  ['mirror', 'Mirror', false],
];
const TIPS = [4, 8, 12, 16, 20];

export class CameraPanel {
  constructor(rv) {
    this.rv = rv;
    this.info = rv.info;
    this.mode = rv.info.mode;
    const options = this.mode === 'hand' ? HAND_OPTIONS : EYE_OPTIONS;
    const key = `paralic.inspector.camera.${this.mode}`;
    let saved = {};
    try {
      saved = JSON.parse(localStorage.getItem(key) || '{}');
    } catch (_) { /* private mode */ }
    this.opts = Object.fromEntries(options.map(([k, , d]) => [k, saved[k] ?? d]));
    const toggles = options.map(([k, label]) => {
      const input = h('input', { type: 'checkbox', checked: this.opts[k], 'data-opt': k });
      input.addEventListener('change', () => {
        this.opts[k] = input.checked;
        try {
          localStorage.setItem(key, JSON.stringify(this.opts));
        } catch (_) { /* private mode */ }
        if (k === 'video') this.wantVideo();
        this.draw();
      });
      return h('label', { class: 'toggle' }, input, label);
    });
    this.canvas = h('canvas', { class: 'cam-canvas', 'aria-label': 'Camera frame with overlays' });
    this.note = h('div', { class: 'cam-note' });
    this.el = h('section', { class: 'panel camera', id: 'sec-camera' },
      h('header', { class: 'panel-head' }, h('h2', {}, 'Camera'), h('div', { class: 'toggles' }, toggles)),
      h('div', { class: 'cam-wrap' }, this.canvas),
      this.note);
    this.bitmap = null;
    this.bitmapId = null;
    this.detail = null;
    this.video = new VideoFrames(rv.id, (fid, bmp) => {
      this.bitmap = bmp;
      this.bitmapId = fid;
      this.draw();
    });
    new ResizeObserver(() => this.draw()).observe(this.el);
  }

  onFrame() {
    this.wantVideo();
    this.draw();
  }

  wantVideo() {
    if (this.opts.video) this.video.want(this.rv.player.videoId());
  }

  onDetail(d) {
    this.detail = d;
    this.draw();
  }

  /** Width and height of the frame the landmarks refer to, and where that came from. */
  frameSize() {
    const img = this.info.image;
    if (this.mode === 'hand') {
      const b = this.bitmap;
      return { w: b ? b.width : 640, h: b ? b.height : 480, source: 'image' };
    }
    if (img && img.w && img.h) return img;
    const b = this.bitmap;
    if (b) return { w: b.width, h: b.height, source: 'video (frame size not recorded)' };
    return { w: 960, h: 540, source: 'assumed' };
  }

  draw() {
    const wrap = this.canvas.parentElement;
    const width = wrap.clientWidth;
    if (!width) return;
    const fs = this.frameSize();
    const height = Math.min(width * (fs.h / fs.w), 560);
    const cw = Math.min(width, height * (fs.w / fs.h));
    const ctx = fitCanvas(this.canvas, cw, height);
    const s = cw / fs.w;                 // canvas px per landmark px (eye mode)
    const mirror = this.opts.mirror;
    const X = (x) => (mirror ? cw - x : x);
    ctx.clearRect(0, 0, cw, height);
    const d = this.detail;
    const showing = this.rv.player.n;
    const fresh = d && d.n === showing;

    // The image (or a plain background).
    const vid = this.rv.player.videoId();
    if (this.opts.video && this.bitmap) {
      ctx.save();
      if (mirror) {
        ctx.translate(cw, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(this.bitmap, 0, 0, cw, height);
      ctx.restore();
      if (this.bitmapId !== vid) {
        ctx.fillStyle = 'rgba(10, 15, 31, 0.35)';
        ctx.fillRect(0, 0, cw, height);
      }
    } else {
      ctx.fillStyle = '#0b1226';
      ctx.fillRect(0, 0, cw, height);
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      for (let gx = 0; gx <= 10; gx++) {
        ctx.beginPath();
        ctx.moveTo(Math.round((gx * cw) / 10) + 0.5, 0);
        ctx.lineTo(Math.round((gx * cw) / 10) + 0.5, height);
        ctx.stroke();
      }
      for (let gy = 0; gy <= 10; gy++) {
        ctx.beginPath();
        ctx.moveTo(0, Math.round((gy * height) / 10) + 0.5);
        ctx.lineTo(cw, Math.round((gy * height) / 10) + 0.5);
        ctx.stroke();
      }
    }

    const lm = d && d.landmarks;
    if (lm && lm.points) {
      ctx.globalAlpha = fresh ? 1 : 0.45;
      if (this.mode === 'hand') this.drawHand(ctx, lm.points, cw, height, X);
      else this.drawFace(ctx, lm.points, s, X, d);
      ctx.globalAlpha = 1;
    }
    if (this.mode === 'eyes' && d && this.opts.others) this.drawOthers(ctx, d, cw, height, s, X);

    // What is on screen, in words.
    const parts = [];
    if (vid == null) parts.push('no camera image recorded up to this frame');
    else if (!this.opts.video) parts.push('video hidden');
    else if (this.bitmapId === vid) {
      const dn = this.rv.player.ids[this.rv.player.n] - vid;
      parts.push(`image of frame id ${vid}${dn ? ` (${dn} frame ids earlier: video kept every n-th frame)` : ''}`);
    } else parts.push('loading image…');
    if (d && fresh) {
      if (lm) parts.push(`${lm.points.length} landmarks (${lm.chunk})`);
      else parts.push(this.mode === 'hand' ? 'no hand found in this frame' : 'no face found in this frame');
    }
    if (this.mode === 'eyes') {
      const b = this.bitmap;
      const scale = b ? fs.w / b.width : null;
      parts.push(`analysed ${Math.round(fs.w)}×${Math.round(fs.h)} (${fs.source})` +
        (b && scale && Math.abs(scale - 1) > 0.01 ? ` · video ${b.width}×${b.height} (×${num(scale, 2)})` : ''));
    }
    this.note.textContent = parts.join(' · ');
  }

  drawFace(ctx, P, s, X, d) {
    const mesh = this.info.mesh;
    const pt = (i) => [X(P[i][0] * s), P[i][1] * s];
    if (this.opts.mesh) {
      ctx.fillStyle = 'rgba(238, 242, 255, 0.55)';
      for (let i = 0; i < P.length; i++) {
        const [x, y] = pt(i);
        ctx.fillRect(x - 0.7, y - 0.7, 1.4, 1.4);
      }
      ctx.strokeStyle = 'rgba(94, 234, 212, 0.55)';
      ctx.lineWidth = 1;
      mesh.lines.forEach((loop, k) => {
        ctx.beginPath();
        loop.forEach((i, j) => {
          const [x, y] = pt(i);
          if (j) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
        });
        if (k < 4) ctx.closePath();      // face oval, lips, eyes; brows and nose stay open
        ctx.stroke();
      });
    }
    if (this.opts.eyes) {
      for (const eye of ['left', 'right']) {
        const color = EYE[eye];
        const contour = mesh[`${eye}_eye`];
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        contour.forEach((i, j) => {
          const [x, y] = pt(i);
          if (j) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
        const iris = mesh[`${eye}_iris`];
        if (P.length > iris[4]) {
          const [cx, cy] = pt(iris[0]);
          let r = 0;
          for (const i of iris.slice(1)) {
            const [x, y] = pt(i);
            r += Math.hypot(x - cx, y - cy) / 4;
          }
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = color;
          ctx.fillRect(cx - 1.5, cy - 1.5, 3, 3);
          // The eye's axis and where the iris sits along it (the features' dx).
          const [a, b] = mesh[`${eye}_corners`].map(pt);
          ctx.strokeStyle = alpha(color, 0.7);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a[0], a[1]);
          ctx.lineTo(b[0], b[1]);
          ctx.stroke();
          const ax = b[0] - a[0];
          const ay = b[1] - a[1];
          const len2 = ax * ax + ay * ay || 1;
          const u = ((cx - a[0]) * ax + (cy - a[1]) * ay) / len2;
          const qx = a[0] + u * ax;
          const qy = a[1] + u * ay;
          ctx.beginPath();
          ctx.moveTo(qx, qy);
          ctx.lineTo(cx, cy);
          ctx.stroke();
          ctx.fillStyle = C.text;
          ctx.beginPath();
          ctx.arc(qx, qy, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    let box = null;
    if (this.opts.box || this.opts.pose) {
      let x0 = Infinity; let y0 = Infinity; let x1 = -Infinity; let y1 = -Infinity;
      for (const p of P) {
        x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
      }
      box = [x0 * s, y0 * s, x1 * s, y1 * s];
    }
    if (this.opts.box && box) {
      const [a, b] = [X(box[0]), X(box[2])].sort((m, n) => m - n);
      ctx.strokeStyle = 'rgba(238, 242, 255, 0.8)';
      ctx.lineWidth = 1;
      ctx.strokeRect(Math.round(a) + 0.5, Math.round(box[1]) + 0.5, b - a, box[3] - box[1]);
      label(ctx, a, box[1] - 4, 'face');
    }
    const msg = d.frame && d.frame.msg;
    if (this.opts.pose && box && msg && Array.isArray(msg.head)) {
      const [yaw, pitch, roll] = msg.head.map((v) => (v * Math.PI) / 180);
      const R = rotation(yaw, pitch, roll);
      const [ox, oy] = pt(mesh.nose_tip);
      const L = 0.45 * (box[2] - box[0]);
      const axes = [[0, S[7], 'x'], [1, S[5], 'y'], [2, S[0], 'z']];
      for (const [k, color, name] of axes) {
        // Camera space: x to the right of the image, y up, z towards the camera.
        const dx = R[0][k] * L * (this.opts.mirror ? -1 : 1);
        const dy = -R[1][k] * L;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(ox + dx, oy + dy);
        ctx.stroke();
        label(ctx, ox + dx * 1.08, oy + dy * 1.08, name);
      }
      label(ctx, X(this.opts.mirror ? box[2] : box[0]), box[3] + 14,
        `yaw ${num(msg.head[0])}°  pitch ${num(msg.head[1])}°  roll ${num(msg.head[2])}°` +
        (isNum(msg.dist) ? `  ·  ${num(msg.dist)} cm` : ''));
    }
  }

  drawOthers(ctx, d, cw, height, s, X) {
    const msg = (d.frame && d.frame.msg) || {};
    const boxes = [];
    const others = d.landmarks && d.landmarks.others;
    const toCanvas = (b) => {
      const frac = Math.max(...b.map(Math.abs)) <= 1.5;
      return frac ? [b[0] * cw, b[1] * height, b[2] * cw, b[3] * height] : b.map((v) => v * s);
    };
    if (Array.isArray(others)) others.forEach((b) => boxes.push([toCanvas(b), 'other face', S[6]]));
    else if (Array.isArray(msg.others)) msg.others.forEach((b) => boxes.push([toCanvas(b), 'other face', S[6]]));
    if (Array.isArray(msg.you)) boxes.push([toCanvas(msg.you), 'followed face', C.accent]);
    for (const [b, text, color] of boxes) {
      const [a, z] = [X(b[0]), X(b[2])].sort((m, n) => m - n);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(a, b[1], z - a, b[3] - b[1]);
      label(ctx, a, b[1] - 4, text);
    }
    if (msg.waiting || (isNum(msg.faces) && msg.faces > 1)) {
      label(ctx, 8, 16, `${msg.faces === 1 ? 'a face' : `${msg.faces} faces`} in view${msg.waiting ? ', not this person: waiting' : ''}`);
    }
  }

  drawHand(ctx, P, cw, height, X) {
    const pt = (i) => [X(P[i][0] * cw), P[i][1] * height];
    const d = this.detail;
    const msg = (d && d.frame && d.frame.msg) || {};
    if (this.opts.skeleton) {
      ctx.strokeStyle = alpha(C.accent, 0.85);
      ctx.lineWidth = 2;
      for (const [a, b] of this.info.hand_bones) {
        const [x0, y0] = pt(a);
        const [x1, y1] = pt(b);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
      for (let i = 0; i < P.length; i++) {
        const [x, y] = pt(i);
        ctx.fillStyle = TIPS.includes(i) ? C.text : 'rgba(238, 242, 255, 0.75)';
        ctx.beginPath();
        ctx.arc(x, y, TIPS.includes(i) ? 3.5 : 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      // The palm width every distance is measured in.
      const [a, b] = [pt(5), pt(17)];
      ctx.strokeStyle = 'rgba(238, 242, 255, 0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      label(ctx, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 14, 'palm width');
    }
    if (this.opts.pinch) {
      const [a, b] = [pt(4), pt(8)];
      ctx.strokeStyle = msg.pinching ? S[4] : C.muted;
      ctx.lineWidth = msg.pinching ? 3 : 1.5;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      const hand = msg.hand || {};
      const thr = d && d.pinch;
      label(ctx, Math.min(a[0], b[0]) - 4, Math.max(a[1], b[1]) + 18,
        `pinch ${num(hand.pinch, 2)} palm widths${thr ? ` (closes below ${num(thr.on, 2)}, opens above ${num(thr.off, 2)})` : ''}` +
        (msg.pinching ? ' · pinching' : '') + (msg.scrolling ? ' · scrolling' : ''));
    }
    if (this.opts.tip) {
      const [x, y] = pt(8);
      ctx.strokeStyle = GAZE.cursor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.stroke();
      label(ctx, x + 12, y - 10, 'fingertip → cursor');
    }
    if (msg.hand && msg.hand.open) label(ctx, 8, 16, 'open hand (the "stop" palm)');
  }
}

function label(ctx, x, y, text) {
  ctx.font = '12px system-ui, sans-serif';
  ctx.textBaseline = 'alphabetic';
  const w = ctx.measureText(text).width;
  ctx.fillStyle = 'rgba(10, 15, 31, 0.72)';
  ctx.fillRect(x - 3, y - 12, w + 6, 16);
  ctx.fillStyle = C.text;
  ctx.fillText(text, x, y);
}

/** Rotation for the head pose angles (as features.head_pose_from_matrix reads them back). */
function rotation(yaw, pitch, roll) {
  const cy = Math.cos(yaw); const sy = Math.sin(yaw);
  const cp = Math.cos(-pitch); const sp = Math.sin(-pitch);
  const cr = Math.cos(roll); const sr = Math.sin(roll);
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
  const Rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]];
  const Rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]];
  const mul = (A, B) => A.map((row) => [0, 1, 2].map((j) => row.reduce((acc, v, k) => acc + v * B[k][j], 0)));
  return mul(mul(Ry, Rx), Rz);
}
