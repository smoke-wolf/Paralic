// Inline SVG icons (simple strokes, currentColor).

const paths = {
  back: '<path d="M15 5l-7 7 7 7"/><path d="M8 12h12"/>',
  home: '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v10h13V10"/><path d="M10 20v-6h4v6"/>',
  planet: '<circle cx="12" cy="12" r="6"/><path d="M3.5 15.5c-1.5 2.2.5 3.6 4.5 2.9 3-.5 6.8-2.2 9.8-4.6 3.1-2.5 4.5-5.1 3-6.2-.9-.7-2.6-.6-4.6.1"/>',
  book: '<path d="M4 5.5C6.5 4 9.5 4 12 5.5V20c-2.5-1.5-5.5-1.5-8 0z"/><path d="M20 5.5C17.5 4 14.5 4 12 5.5V20c2.5-1.5 5.5-1.5 8 0z"/>',
  chat: '<path d="M4 5h16v11H9l-5 4z"/><path d="M8 9.5h8M8 12.5h5"/>',
  target: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
  sliders: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2.2"/><circle cx="9" cy="17" r="2.2"/>',
  pause: '<path d="M9 5v14M15 5v14"/>',
  play: '<path d="M7 5l12 7-12 7z"/>',
  eye: '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3.2"/>',
  blink: '<path d="M2 11c3 3.5 6.3 5 10 5s7-1.5 10-5"/><path d="M5 14.5 3.5 17M9.3 16l-.8 2.8M14.7 16l.8 2.8M19 14.5l1.5 2.5"/>',
  crosshair: '<circle cx="12" cy="12" r="7"/><path d="M12 2v5M12 17v5M2 12h5M17 12h5"/>',
  scroll: '<path d="M12 3v18"/><path d="M8 7l4-4 4 4M8 17l4 4 4-4"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 0 1 4.8.9c0 1.7-2.3 2.1-2.3 3.6"/><circle cx="12" cy="17" r=".6" fill="currentColor"/>',
  speaker: '<path d="M4 9.5h4l5-4v13l-5-4H4z"/><path d="M16.5 9a4 4 0 0 1 0 6M19 6.5a7.5 7.5 0 0 1 0 11"/>',
  del: '<path d="M9 5h11v14H9l-6-7z"/><path d="M12.5 9.5l5 5M17.5 9.5l-5 5"/>',
  space: '<path d="M4 10v4h16v-4"/>',
  trash: '<path d="M4 7h16M9 7V4.5h6V7M6.5 7l1 13h9l1-13"/>',
  keyboard: '<rect x="2.5" y="6" width="19" height="12" rx="2"/><path d="M6 10h.01M9 10h.01M12 10h.01M15 10h.01M18 10h.01M7 14h10"/>',
  grid: '<rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/>',
  check: '<path d="M4.5 12.5l5 5L20 7"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
  camera: '<path d="M3 8h4l2-3h6l2 3h4v11H3z"/><circle cx="12" cy="13" r="3.5"/>',
  mouse: '<rect x="6" y="3" width="12" height="18" rx="6"/><path d="M12 7v4"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  head: '<circle cx="12" cy="9" r="5"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/>',
  arrowRight: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  arrowLeft: '<path d="M19 12H5M11 6l-6 6 6 6"/>',
  sparkle: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"/>',
  brain: '<path d="M9 4.5a3 3 0 0 0-5 2.2A3.2 3.2 0 0 0 3 12a3.3 3.3 0 0 0 2 5.3A3 3 0 0 0 9 19.5V4.5zM15 4.5a3 3 0 0 1 5 2.2 3.2 3.2 0 0 1 1 5.3 3.3 3.3 0 0 1-2 5.3 3 3 0 0 1-4 2.2V4.5z"/>',
  bolt: '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
  history: '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 8v4l3 2"/>',
  drop: '<path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z"/>',
};

export function icon(name, extraClass = '') {
  const body = paths[name] || paths.help;
  return `<svg class="icon ${extraClass}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
}

/** Create an icon as a DOM node. */
export function iconEl(name, extraClass = '') {
  const tpl = document.createElement('template');
  tpl.innerHTML = icon(name, extraClass).trim();
  return tpl.content.firstElementChild;
}
