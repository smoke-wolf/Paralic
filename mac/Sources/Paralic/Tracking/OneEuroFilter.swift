// OneEuroFilter.swift — One-Euro filter + blink-aware cursor stabiliser.
// Port of filters.py (Casiez, Roussel & Vogel, 1€ Filter, CHI 2012).

import Foundation

private func alpha(cutoff: Double, dt: Double) -> Double {
    let tau = 1.0 / (2.0 * Double.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)
}

/// One-Euro filter for a scalar signal.
final class OneEuro {
    var minCutoff: Double
    var beta: Double
    var dCutoff: Double
    private var xPrev: Double?
    private var dxPrev: Double = 0
    private var tPrev: Double?

    init(minCutoff: Double = 1.0, beta: Double = 0.0, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff; self.beta = beta; self.dCutoff = dCutoff
    }

    func reset() { xPrev = nil; dxPrev = 0; tPrev = nil }

    func filter(_ x: Double, t: Double) -> Double {
        guard let xp = xPrev, let tp = tPrev else { xPrev = x; tPrev = t; return x }
        let dt = Swift.max(1e-3, t - tp)
        let dx = (x - xp) / dt
        let aD = alpha(cutoff: dCutoff, dt: dt)
        let dxHat = aD * dx + (1 - aD) * dxPrev
        let cutoff = minCutoff + beta * abs(dxHat)
        let a = alpha(cutoff: cutoff, dt: dt)
        let xHat = a * x + (1 - a) * xp
        xPrev = xHat; dxPrev = dxHat; tPrev = t
        return xHat
    }
}

/// 2-D One-Euro + saccade bypass + velocity lead + "freeze while the eyes are
/// closing" stabiliser. Coordinates are a screen *fraction* (0..1), so the
/// saccade threshold and dead-zone are resolution-independent.
///
/// The design goal is fast *and* precise: while you fixate, the One-Euro filter
/// with a low cutoff holds the cursor rock-steady (and a dead-zone swallows
/// sub-pixel jitter); the moment your eyes jump, the saccade detector blends
/// the output straight to the raw estimate so there is no lag, and a small
/// velocity lead hides the remaining end-to-end latency.
final class GazeStabiliser {
    private let fx: OneEuro
    private let fy: OneEuro
    private var frozen: SIMD2<Double>?
    private var last: SIMD2<Double> = .zero
    private var lastRaw: SIMD2<Double>?
    private var lastT: Double?
    private var velocity: SIMD2<Double> = .zero
    var responsiveness: Responsiveness

    /// Ring buffer of recent (t, point) so we can recover the pre-blink point.
    private var history: [(Double, SIMD2<Double>)] = []
    private let historySeconds = 0.6

    init(_ r: Responsiveness = .balanced) {
        responsiveness = r
        fx = OneEuro(minCutoff: r.minCutoff, beta: r.beta, dCutoff: r.dCutoff)
        fy = OneEuro(minCutoff: r.minCutoff, beta: r.beta, dCutoff: r.dCutoff)
    }

    func apply(_ r: Responsiveness) {
        responsiveness = r
        fx.minCutoff = r.minCutoff; fx.beta = r.beta; fx.dCutoff = r.dCutoff
        fy.minCutoff = r.minCutoff; fy.beta = r.beta; fy.dCutoff = r.dCutoff
    }

    func reset() { fx.reset(); fy.reset(); frozen = nil; history.removeAll(); lastRaw = nil; lastT = nil; velocity = .zero }

    /// Update with a raw gaze point (screen fraction). `closing` freezes the
    /// output at the last stable point. Returns the cursor point to display.
    func update(_ raw: SIMD2<Double>, t: Double, closing: Bool) -> SIMD2<Double> {
        let r = responsiveness
        // instantaneous eye speed (fraction / s)
        var speed = 0.0
        if let lr = lastRaw, let lt = lastT {
            let dt = Swift.max(1e-3, t - lt)
            let v = SIMD2((raw.x - lr.x) / dt, (raw.y - lr.y) / dt)
            velocity = v
            speed = (v.x * v.x + v.y * v.y).squareRoot()
        }
        lastRaw = raw; lastT = t

        let sx = fx.filter(raw.x, t: t)
        let sy = fy.filter(raw.y, t: t)
        var out = SIMD2(sx, sy)

        // Saccade bypass: blend toward the raw estimate proportionally to how
        // far above threshold the eye speed is, so big jumps are instant.
        if speed > r.saccadeThreshold {
            let k = Swift.min(1.0, r.saccadeSnap * (speed / r.saccadeThreshold - 1.0))
            out = SIMD2(out.x + (raw.x - out.x) * k, out.y + (raw.y - out.y) * k)
        }

        // Velocity lead to counter latency.
        let lead = r.predictionLeadMs / 1000.0
        out = SIMD2(out.x + velocity.x * lead, out.y + velocity.y * lead)

        // Dead-zone while essentially still: swallow micro-jitter for precision.
        let dz = r.deadZone
        if speed < r.saccadeThreshold {
            let d = ((out.x - last.x) * (out.x - last.x) + (out.y - last.y) * (out.y - last.y)).squareRoot()
            if d < dz { out = last }
        }
        out = SIMD2(Swift.max(0, Swift.min(1, out.x)), Swift.max(0, Swift.min(1, out.y)))

        if !closing {
            last = out
            history.append((t, out))
            while let first = history.first, t - first.0 > historySeconds { history.removeFirst() }
            frozen = nil
            return out
        } else {
            if frozen == nil { frozen = last }
            return frozen!
        }
    }

    /// The point the user was looking at `secondsAgo` before now (for clicks).
    func pointBefore(_ tEvent: Double, secondsAgo: Double) -> SIMD2<Double> {
        let target = tEvent - secondsAgo
        var best = last
        for (t, p) in history.reversed() where t <= target { best = p; break }
        return best
    }
}
