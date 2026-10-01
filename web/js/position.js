// Seating position check, used before calibrating (DOM-free, so it can be
// tested with node).
//
// The gaze network is most accurate when the face is well lit, at a sensible
// distance, roughly in front of the camera - and, after a full calibration,
// when the person sits about where they sat while calibrating. The page shows
// one hint at a time until everything is fine.
//
// Frame fields used (see paralic/session.py): `face`, `head` [yaw, pitch,
// roll] in degrees, `dist` (cm from the camera), `pos` [x, y] (cm; x grows
// to the right of the camera image - the person's left - and y upwards) and,
// while the page asks for it, `light` {face, frame, balance}.

export const POSITION_LIMITS = Object.freeze({
  minDist: 35,       // cm
  maxDist: 90,
  maxAngleX: 14,     // degrees off the camera axis, side to side
  maxAngleY: 25,     // and up / down (the camera usually sits above the screen)
  maxTurn: 25,       // face turned away from the camera (degrees)
  minLight: 60,      // mean face brightness, 0-255
  backlight: 55,     // the frame this much brighter than the face = light behind
  balance: 0.3,      // one side of the face this much brighter (relative)
  refXY: 3,          // cm: how close to the calibrated position counts as "there"
  refDist: 0.08,     // fraction of the calibrated distance (at least 3 cm)
  refTurn: 10,       // degrees
});

/**
 * Assess one frame. `ref` is the calibrated position ({x, y, dist, yaw,
 * pitch}) when the person should return to it, else null. Returns
 * {face, place, light, ok, hint}: `place` and `light` are '' when fine, else
 * the hint for that item; `hint` is the most important one.
 */
export function assessPosition(m, ref = null, lim = POSITION_LIMITS) {
  if (!m || !m.face) {
    return { face: false, place: '', light: '', ok: false, hint: 'I can’t see your face — look at the screen' };
  }
  const [yaw = 0, pitch = 0] = m.head || [];
  const [px = 0, py = 0] = m.pos || [];
  const dist = Number(m.dist) || 0;

  let light = '';
  const L = m.light;
  if (L && L.face != null) {
    if (L.face < lim.minLight) light = 'It’s a bit dark — add some light in front of you';
    else if (L.frame - L.face > lim.backlight) light = 'There is bright light behind you — face the light instead';
    else if (Math.abs(L.balance) > lim.balance) light = 'Light falls from one side — try to light your face evenly';
  }

  // Camera x grows towards the person's left, so a positive offset means
  // "move to your right". Distance matters most, then side to side, then height.
  let place = '';
  if (ref && ref.dist && dist) {
    const tolD = Math.max(3, lim.refDist * ref.dist);
    const dx = px - ref.x;
    const dy = py - ref.y;
    if (dist - ref.dist > tolD) place = 'Move a little closer';
    else if (ref.dist - dist > tolD) place = 'Lean back a little';
    else if (dx > lim.refXY) place = 'Move a little to your right';
    else if (dx < -lim.refXY) place = 'Move a little to your left';
    else if (dy > lim.refXY) place = 'Move your head a little lower';
    else if (dy < -lim.refXY) place = 'Move your head a little higher';
    else if (Math.abs(yaw - (ref.yaw || 0)) > lim.refTurn || Math.abs(pitch - (ref.pitch || 0)) > lim.refTurn) {
      place = 'Turn your face back towards the screen';
    }
  } else if (dist) {
    const ax = (Math.atan2(px, dist) * 180) / Math.PI;
    const ay = (Math.atan2(py, dist) * 180) / Math.PI;
    if (dist < lim.minDist) place = 'You are quite close — lean back a little';
    else if (dist > lim.maxDist) place = 'You are quite far away — move a little closer';
    else if (ax > lim.maxAngleX) place = 'Move a little to your right';
    else if (ax < -lim.maxAngleX) place = 'Move a little to your left';
    else if (ay > lim.maxAngleY) place = 'Move your head a little lower';
    else if (ay < -lim.maxAngleY) place = 'Move your head a little higher';
    else if (Math.abs(yaw) > lim.maxTurn || Math.abs(pitch) > lim.maxTurn) place = 'Face the screen straight on';
  }
  return { face: true, place, light, ok: !place && !light, hint: place || light || '' };
}
