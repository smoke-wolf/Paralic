// The Inspector's server API, and loaders that keep playback quick: only the
// newest request matters (frames skipped while playing fast are never
// fetched), frames come in windows, and only the video frame on screen is
// decoded.

const enc = encodeURIComponent;

async function json(path, params) {
  const q = params ? '?' + new URLSearchParams(Object.entries(params).filter(([, v]) => v != null)).toString() : '';
  const res = await fetch(path + q);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) { /* not JSON */ }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export const api = {
  recordings: () => json('/api/recordings'),
  info: (id) => json(`/api/recordings/${enc(id)}`),
  index: (id) => json(`/api/recordings/${enc(id)}/index`),
  frames: (id, start, count, light = true) => json(`/api/recordings/${enc(id)}/frames`, { start, count, light }),
  detail: (id, n) => json(`/api/recordings/${enc(id)}/frame/${n}`),
  signals: (id, t0, t1, points) => json(`/api/recordings/${enc(id)}/signals`, { t0, t1, points }),
  timeline: (id) => json(`/api/recordings/${enc(id)}/timeline`),
  events: (id) => json(`/api/recordings/${enc(id)}/events`),
  calibrations: (id) => json(`/api/recordings/${enc(id)}/calibrations`),
  diagnosis: (id) => json(`/api/recordings/${enc(id)}/diagnosis`),
  config: (id) => json(`/api/recordings/${enc(id)}/config`),
  videoUrl: (id, frameId) => `/api/recordings/${enc(id)}/video/${frameId}`,
};

/**
 * Loads one thing at a time, always the newest wanted: while a request runs,
 * newer wishes replace each other and only the last is fetched next.
 * ``onLoad(key, value)`` gets every result that arrives.
 */
export class Latest {
  constructor(fetcher, onLoad, onError = null) {
    this.fetcher = fetcher;
    this.onLoad = onLoad;
    this.onError = onError;
    this.busy = false;
    this.wanted = undefined;
    this.current = undefined;
  }

  want(key) {
    if (key === this.current && !this.busy) return;
    this.wanted = key;
    if (!this.busy) this._next();
  }

  async _next() {
    if (this.wanted === undefined) return;
    const key = this.wanted;
    this.wanted = undefined;
    this.busy = true;
    try {
      const value = await this.fetcher(key);
      this.current = key;
      this.onLoad(key, value);
    } catch (err) {
      if (this.onError) this.onError(key, err);
      else console.warn(err);
    } finally {
      this.busy = false;
      if (this.wanted !== undefined && this.wanted !== this.current) this._next();
      else this.wanted = undefined;
    }
  }
}

/** Frames (without their bulky overlay fields) in windows of WINDOW, the last few windows kept. */
export class FrameWindows {
  static WINDOW = 256;

  constructor(id, total, keep = 16) {
    this.id = id;
    this.total = total;
    this.keep = keep;
    this.windows = new Map();     // window number -> frames array (or a pending promise)
    this.listeners = new Set();
  }

  /** The frame if loaded; otherwise starts loading its window and returns null. */
  get(n) {
    if (n < 0 || n >= this.total) return null;
    const w = Math.floor(n / FrameWindows.WINDOW);
    const win = this.windows.get(w);
    if (Array.isArray(win)) {
      this.windows.delete(w);           // most recently used last
      this.windows.set(w, win);
      return win[n - w * FrameWindows.WINDOW] || null;
    }
    if (!win) this._load(w);
    return null;
  }

  /** Frames lo..hi-1 that are loaded (missing ones as null); loads what is missing. */
  range(lo, hi) {
    const out = [];
    for (let n = Math.max(0, lo); n < Math.min(hi, this.total); n++) out.push(this.get(n));
    return out;
  }

  _load(w) {
    const start = w * FrameWindows.WINDOW;
    const p = api.frames(this.id, start, FrameWindows.WINDOW).then((res) => {
      this.windows.set(w, res.frames);
      while (this.windows.size > this.keep) this.windows.delete(this.windows.keys().next().value);
      for (const fn of this.listeners) fn(w);
    }).catch((err) => {
      this.windows.delete(w);
      console.warn(err);
    });
    this.windows.set(w, p);
  }
}

/** Decoded video frames: the newest wanted one is fetched and decoded; a few are kept for stepping. */
export class VideoFrames {
  constructor(id, onFrame, keep = 24) {
    this.id = id;
    this.onFrame = onFrame;
    this.keep = keep;
    this.cache = new Map();       // video frame id -> ImageBitmap
    this.loader = new Latest((fid) => this._fetch(fid), (fid, bmp) => {
      if (bmp) this.onFrame(fid, bmp);
    }, () => this.onFrame(null, null));
  }

  /** Show video frame ``fid`` (null: none). */
  want(fid) {
    if (fid == null) {
      this.onFrame(null, null);
      return;
    }
    const hit = this.cache.get(fid);
    if (hit) {
      this.cache.delete(fid);
      this.cache.set(fid, hit);
      this.onFrame(fid, hit);
      return;
    }
    this.loader.want(fid);
  }

  async _fetch(fid) {
    const res = await fetch(api.videoUrl(this.id, fid) + '?exact=1');
    if (!res.ok) return null;
    const bmp = await createImageBitmap(await res.blob());
    this.cache.set(fid, bmp);
    while (this.cache.size > this.keep) {
      const old = this.cache.keys().next().value;
      this.cache.get(old)?.close?.();
      this.cache.delete(old);
    }
    return bmp;
  }
}
