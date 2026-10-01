// Paralic Inspector: the list of recordings (#/) and one recording (#/rec/<id>?n=<frame>).

import { renderList } from './list.js';
import { RecordingView } from './recording.js';

const root = document.getElementById('app');
let view = null;

function parse() {
  const m = location.hash.match(/^#\/rec\/([^?]+)(?:\?n=(\d+))?$/);
  return m ? { id: decodeURIComponent(m[1]), n: m[2] != null ? Number(m[2]) : null } : null;
}

async function route() {
  const r = parse();
  if (r) {
    if (view && view.id === r.id) {
      if (r.n != null && view.player) view.player.go(r.n);
      return;
    }
    if (view) view.destroy();
    view = new RecordingView(root, r.id);
    window.scrollTo(0, 0);
    await view.open(r.n);
    return;
  }
  if (view) {
    view.destroy();
    view = null;
  }
  document.title = 'Paralic Inspector';
  await renderList(root);
}

window.addEventListener('hashchange', route);
route();

// For tests and the browser console.
window.inspector = { get view() { return view; } };
