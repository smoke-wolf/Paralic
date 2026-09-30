// selftest_main.swift — offline proof that the personalization + A/B selection
// pipeline is numerically sound. NOT part of the app target; compiled into a
// separate `paralic-selftest` binary.
//
// It synthesises a plausible per-user calibration (13 dots, iris geometry that
// depends nonlinearly on gaze and is contaminated by head pose, plus per-frame
// noise), then runs GazeNet.autoTrain and prints the A/B bake-off and the
// resulting accuracy in centimetres.

import Foundation

setbuf(stdout, nil)   // unbuffered, so output survives an early trap

let W = 1920.0, H = 1080.0
let pxPerCm = 37.8

// 13-dot calibration grid (fractions of the screen).
let dots: [SIMD2<Double>] = {
    var d: [SIMD2<Double>] = []
    for gy in [0.1, 0.5, 0.9] { for gx in [0.1, 0.5, 0.9] { d.append(SIMD2(gx, gy)) } }
    d.append(SIMD2(0.3, 0.3)); d.append(SIMD2(0.7, 0.3))
    d.append(SIMD2(0.3, 0.7)); d.append(SIMD2(0.7, 0.7))
    return d
}()

let framesPerDot = 22
var rng = SeededRNG(seed: 42)

func syntheticFeatures(gaze g: SIMD2<Double>, head: (yaw: Double, pitch: Double, roll: Double, t: SIMD3<Double>),
                       noise: Double) -> [Double] {
    let gx = g.x - 0.5, gy = g.y - 0.5
    func n() -> Double { rng.gaussian(noise) }
    // iris offset: mostly linear in gaze, mild cubic curvature, plus head-pose contamination.
    let rdx = 0.30 * gx + 0.07 * (gx * gx * gx) + 0.16 * head.yaw + n()
    let rdy = 0.30 * gy + 0.07 * (gy * gy * gy) + 0.16 * head.pitch + n()
    let ldx = 0.29 * gx + 0.06 * (gx * gx * gx) + 0.15 * head.yaw + n()
    let ldy = 0.31 * gy + 0.06 * (gy * gy * gy) + 0.15 * head.pitch + n()
    let ropen = 0.30 + 0.02 * rng.gaussian(1) * noise
    let lopen = 0.30 + 0.02 * rng.gaussian(1) * noise
    func clamp01(_ x: Double) -> Double { Swift.max(0, Swift.min(1, x)) }
    let lookOut = clamp01(0.5 + gx) , lookIn = clamp01(0.5 - gx)
    let lookUp = clamp01(0.5 - gy), lookDown = clamp01(0.5 + gy)
    return [
        rdx, rdy, ldx, ldy,
        ropen, lopen,
        lookIn + n() * 2, lookOut + n() * 2, lookUp + n() * 2, lookDown + n() * 2,   // left-eye looks
        lookIn + n() * 2, lookOut + n() * 2, lookUp + n() * 2, lookDown + n() * 2,   // right-eye looks
        head.yaw, head.pitch, head.roll,
        head.t.x, head.t.y, head.t.z,
    ]
}

// Build the calibration set.
var Xrows: [[Double]] = []
var Yrows: [[Double]] = []
var groups: [Int] = []
for (di, dot) in dots.enumerated() {
    // Head pose varies per frame (simulating the head-movement calibration
    // step) around a per-dot centre, so it is NOT a dot-identity proxy.
    let centreYaw = rng.gaussian(0.06), centrePitch = rng.gaussian(0.05)
    for _ in 0..<framesPerDot {
        let head = (yaw: centreYaw + rng.gaussian(0.10), pitch: centrePitch + rng.gaussian(0.08),
                    roll: rng.gaussian(0.03),
                    t: SIMD3(rng.gaussian(1.5), rng.gaussian(1.5), -55 + rng.gaussian(2.0)))
        Xrows.append(syntheticFeatures(gaze: dot, head: head, noise: 0.012))
        Yrows.append([dot.x * W, dot.y * H])
        groups.append(di)
    }
}

let X = Mat(rowsOf: Xrows)
let Y = Mat(rowsOf: Yrows)
let minStd: [Double] = [0.01,0.01,0.01,0.01, 0.01,0.01, 0.03,0.03,0.03,0.03, 0.03,0.03,0.03,0.03,
                        0.035,0.035,0.035, 1.0,1.0,1.5]

print("Paralic personalization self-test")
print(String(repeating: "─", count: 56))
print("calibration: \(dots.count) dots × \(framesPerDot) frames = \(X.rows) samples, \(X.cols) features")
print("")

let t0 = Date()
let (model, report, ab) = GazeNet.autoTrain(X: X, Y: Y, groups: groups, minStd: minStd, pxPerCm: pxPerCm, seed: 7)
let secs = Date().timeIntervalSince(t0)

func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }
print("A/B model bake-off (grouped cross-validation — whole dots held out):")
print("  " + pad("config", 14) + pad("CV error", 12) + pad("train err", 12))
for r in ab.results.sorted(by: { ($0.cvErrorCm.isFinite ? $0.cvErrorCm : 1e9) < ($1.cvErrorCm.isFinite ? $1.cvErrorCm : 1e9) }) {
    let cv = r.cvErrorCm.isFinite ? String(format: "%.2f cm", r.cvErrorCm) : "n/a"
    let tr = String(format: "%.0f px", r.trainErrorPx)
    print("  " + pad(r.config, 14) + pad(cv, 12) + pad(tr, 12) + (r.chosen ? "◀ chosen" : ""))
}
print("")
print(String(format: "winner:            %@", ab.winner as NSString))
print(String(format: "linear baseline:   %.2f cm  (grouped-CV)", ab.linearBaselineCm))
print(String(format: "personalized:      %.2f cm  (grouped-CV)", ab.winnerCvErrorCm))
print(String(format: "improvement:       %.1f%% better than linear", ab.improvementVsLinearPct))
print(String(format: "chosen L2:         %g", report.l2))
print(String(format: "trained in:        %.2f s (ensemble of %d)", secs, model.nets.count))
print("")

// Hold-out generalisation test: brand-new gaze points the net never saw.
var errs: [Double] = []
var rng2 = SeededRNG(seed: 999)
for _ in 0..<200 {
    let g = SIMD2(Double(rng2.next() >> 11) / 9007199254740992.0,
                  Double(rng2.next() >> 11) / 9007199254740992.0)
    let head = (yaw: rng2.gaussian(0.12), pitch: rng2.gaussian(0.10), roll: rng2.gaussian(0.03),
                t: SIMD3(rng2.gaussian(1.5), rng2.gaussian(1.5), -55 + rng2.gaussian(2.0)))
    let feat = syntheticFeatures(gaze: g, head: head, noise: 0.012)
    let p = model.predict(feat)
    let dx = p.x - g.x * W, dy = p.y - g.y * H
    errs.append((dx * dx + dy * dy).squareRoot())
}
let meanCm = errs.mean() / pxPerCm
let sorted = errs.sorted()
let p90Cm = sorted[Int(0.9 * Double(sorted.count))] / pxPerCm
print(String(format: "fresh-gaze holdout (200 random points not in calibration):"))
print(String(format: "  mean error: %.2f cm   90th pct: %.2f cm", meanCm, p90Cm))

// Round-trip the saved profile (proves serialization).
let data = try! JSONEncoder().encode(model.snapshot())
let restored = GazeNet.from(try! JSONDecoder().decode(GazeNetData.self, from: data))
let a = model.predict(Xrows[0]), b = restored.predict(Xrows[0])
let drift = ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)).squareRoot()
print(String(format: "profile save/load round-trip drift: %.6f px  (%d bytes)", drift, data.count))

let ok = ab.winnerCvErrorCm < ab.linearBaselineCm + 0.01 && meanCm < 3.0 && drift < 1e-6
print("")
print(ok ? "✅ PASS — personalization is accurate, A/B selection works, profile persists."
         : "❌ FAIL — see numbers above.")
exit(ok ? 0 : 1)
