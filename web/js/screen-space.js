// Conversions between page (client) coordinates and screen coordinates.
//
// The gaze network is trained to predict *screen* positions (CSS pixels on the
// monitor), not positions inside the page. That way the calibration keeps
// working when you enter or leave full screen or move the window: we only need
// to know where the page's viewport currently sits on the screen.

export function viewportOrigin() {
  // Firefox exposes the exact value.
  if (typeof window.mozInnerScreenX === 'number') {
    return { x: window.mozInnerScreenX, y: window.mozInnerScreenY };
  }
  // Elsewhere estimate it from the window frame: equal side borders, browser
  // toolbars at the top. Exact in full screen, a close estimate otherwise.
  const border = Math.max(0, (window.outerWidth - window.innerWidth) / 2);
  const top = Math.max(0, window.outerHeight - window.innerHeight - border);
  return { x: window.screenX + border, y: window.screenY + top };
}

export function clientToScreen(x, y) {
  const o = viewportOrigin();
  return { x: x + o.x, y: y + o.y };
}

export function screenToClient(x, y) {
  const o = viewportOrigin();
  return { x: x - o.x, y: y - o.y };
}

export function screenInfo() {
  return { w: window.screen.width, h: window.screen.height, dpr: window.devicePixelRatio || 1 };
}
