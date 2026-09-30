// GazeNet.swift — the personal gaze-estimation network (Swift port of gazenet.py).
//
// Vision tells us where the eyes and head are; GazeNet learns, from your
// calibration, where on the screen that means you are looking.
//
//   features ─► Dense(32,tanh) ─► Dense(16,tanh) ─► Dense(2) ─┐
//       │                                                     (+) ─► screen x,y
//       └───────────────── linear skip (ridge init) ──────────┘
//
// Full-batch Adam, Huber loss, L2 weight decay chosen by grouped
// cross-validation (whole calibration dots held out), ensemble of 3.

import Foundation

// MARK: - deterministic RNG (SplitMix64 + Box–Muller)

struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func gaussian(_ std: Double) -> Double {
        let u1 = Double(next() >> 11) * (1.0 / 9007199254740992.0)
        let u2 = Double(next() >> 11) * (1.0 / 9007199254740992.0)
        let mag = (-2.0 * Foundation.log(Swift.max(u1, 1e-12))).squareRoot()
        return mag * Foundation.cos(2.0 * Double.pi * u2) * std
    }
}

// MARK: - Scaler

struct Scaler: Codable {
    var mean: [Double]
    var std: [Double]

    static func fit(_ X: Mat, minStd: [Double]? = nil) -> Scaler {
        var mean = [Double](repeating: 0, count: X.cols)
        var std = [Double](repeating: 0, count: X.cols)
        for j in 0..<X.cols {
            var col = [Double](repeating: 0, count: X.rows)
            for i in 0..<X.rows { col[i] = X[i, j] }
            mean[j] = col.mean()
            let floor = (minStd != nil) ? Swift.max(minStd![j], 1e-6) : 1e-6
            std[j] = Swift.max(col.std(), floor)
        }
        return Scaler(mean: mean, std: std)
    }

    func transform(_ X: Mat) -> Mat {
        var r = X
        for i in 0..<X.rows { for j in 0..<X.cols { r[i, j] = (X[i, j] - mean[j]) / std[j] } }
        return r
    }

    func inverse(_ Z: Mat) -> Mat {
        var r = Z
        for i in 0..<Z.rows { for j in 0..<Z.cols { r[i, j] = Z[i, j] * std[j] + mean[j] } }
        return r
    }
}

// MARK: - weighted ridge (unpenalised intercept)

func ridgeFit(_ X: Mat, _ Y: Mat, lam: Double, weights: [Double]) -> (W: Mat, b: [Double]) {
    let n = X.rows
    let s = weights.reduce(0, +)
    let w = weights.map { $0 / s }
    // weighted means
    var xm = [Double](repeating: 0, count: X.cols)
    var ym = [Double](repeating: 0, count: Y.cols)
    for i in 0..<n {
        for j in 0..<X.cols { xm[j] += w[i] * X[i, j] }
        for j in 0..<Y.cols { ym[j] += w[i] * Y[i, j] }
    }
    var Xc = X, Yc = Y
    for i in 0..<n {
        for j in 0..<X.cols { Xc[i, j] -= xm[j] }
        for j in 0..<Y.cols { Yc[i, j] -= ym[j] }
    }
    // A = (Xc * w).T @ Xc + lam I ; rhs = (Xc * w).T @ Yc
    var Xw = Xc
    for i in 0..<n { for j in 0..<X.cols { Xw[i, j] *= w[i] } }
    var A = Xw.transposeTimes(Xc)
    for d in 0..<A.rows { A[d, d] += lam }
    let rhs = Xw.transposeTimes(Yc)
    let W = Linalg.solve(A, rhs)               // (cols x Yc.cols)
    // b = ym - xm @ W
    var b = ym
    for j in 0..<Y.cols {
        var acc = 0.0
        for k in 0..<X.cols { acc += xm[k] * W[k, j] }
        b[j] = ym[j] - acc
    }
    return (W, b)
}

func groupedFolds(_ groups: [Int], k: Int, seed: UInt64 = 0) -> [[Int]] {
    let uniq = Array(Set(groups)).sorted()
    var rng = SeededRNG(seed: seed)
    var shuffled = uniq; shuffled.shuffle(using: &rng)
    let kk = Swift.max(2, Swift.min(k, shuffled.count))
    var folds: [[Int]] = []
    for i in 0..<kk {
        var members = Set<Int>()
        var idx = i
        while idx < shuffled.count { members.insert(shuffled[idx]); idx += kk }
        var samples: [Int] = []
        for (gi, g) in groups.enumerated() where members.contains(g) { samples.append(gi) }
        folds.append(samples)
    }
    return folds
}

// MARK: - MLP

final class MLPRegressor {
    var W1: Mat, b1: [Double]
    var W2: Mat, b2: [Double]
    var W3: Mat
    var Ws: Mat, b: [Double]
    let nIn: Int, h1: Int, h2: Int, nOut: Int

    init(nIn: Int, nOut: Int = 2, hidden: (Int, Int) = (32, 16), seed: UInt64 = 0) {
        self.nIn = nIn; self.nOut = nOut; self.h1 = hidden.0; self.h2 = hidden.1
        var rng = SeededRNG(seed: seed)
        W1 = Mat(nIn, h1); b1 = [Double](repeating: 0, count: h1)
        W2 = Mat(h1, h2); b2 = [Double](repeating: 0, count: h2)
        W3 = Mat(h2, nOut)                     // starts at zero → pure linear model
        Ws = Mat(nIn, nOut); b = [Double](repeating: 0, count: nOut)
        for i in 0..<nIn { for j in 0..<h1 { W1[i, j] = rng.gaussian(1.0 / Double(nIn).squareRoot()) } }
        for i in 0..<h1 { for j in 0..<h2 { W2[i, j] = rng.gaussian(1.0 / Double(h1).squareRoot()) } }
    }

    @inline(__always) private func tanhInPlace(_ m: inout Mat) {
        for i in 0..<m.data.count { m.data[i] = Foundation.tanh(m.data[i]) }
    }

    func forward(_ X: Mat) -> Mat {
        var A1 = X.matmul(W1).addingRowVector(b1); tanhInPlace(&A1)
        var A2 = A1.matmul(W2).addingRowVector(b2); tanhInPlace(&A2)
        return X.matmul(Ws).addingRowVector(b) + A2.matmul(W3)
    }

    func fit(_ X: Mat, _ Y: Mat, weights: [Double]? = nil, l2: Double = 1e-2, l2skip: Double = 1e-4,
             delta: Double = 0.35, iters: Int = 600, lr: Double = 0.01, ridgeLam: Double = 1e-2) {
        let n = X.rows
        let w = weights.map { ws -> [Double] in let s = ws.reduce(0, +); return ws.map { $0 * Double(n) / s } }
            ?? [Double](repeating: 1, count: n)
        let (rW, rb) = ridgeFit(X, Y, lam: ridgeLam, weights: w)
        Ws = rW; b = rb

        // Adam state
        var mW1 = Mat(W1.rows, W1.cols), vW1 = Mat(W1.rows, W1.cols)
        var mb1 = [Double](repeating: 0, count: h1), vb1 = mb1
        var mW2 = Mat(W2.rows, W2.cols), vW2 = Mat(W2.rows, W2.cols)
        var mb2 = [Double](repeating: 0, count: h2), vb2 = mb2
        var mW3 = Mat(W3.rows, W3.cols), vW3 = Mat(W3.rows, W3.cols)
        var mWs = Mat(Ws.rows, Ws.cols), vWs = Mat(Ws.rows, Ws.cols)
        var mb = [Double](repeating: 0, count: nOut), vb = mb
        let beta1 = 0.9, beta2 = 0.999, eps = 1e-8

        // iters == 0 → leave the net as the pure linear (ridge) model (W3 = 0).
        guard iters > 0 else { return }
        for step in 1...iters {
            // forward with activations retained
            var A1 = X.matmul(W1).addingRowVector(b1); tanhInPlace(&A1)
            var A2 = A1.matmul(W2).addingRowVector(b2); tanhInPlace(&A2)
            let pred = X.matmul(Ws).addingRowVector(b) + A2.matmul(W3)
            let R = pred - Y
            // G = clip(R,-delta,delta) * w/n
            var G = R
            for i in 0..<n {
                let sc = w[i] / Double(n)
                for j in 0..<nOut {
                    let c = Swift.max(-delta, Swift.min(delta, R[i, j]))
                    G[i, j] = c * sc
                }
            }
            _ = R
            // dZ2 = (G @ W3ᵀ) * (1-A2²)
            var dZ2 = G.matmul(W3.transposed)
            for i in 0..<dZ2.data.count { let a = A2.data[i]; dZ2.data[i] *= (1 - a * a) }
            // dZ1 = (dZ2 @ W2ᵀ) * (1-A1²)
            var dZ1 = dZ2.matmul(W2.transposed)
            for i in 0..<dZ1.data.count { let a = A1.data[i]; dZ1.data[i] *= (1 - a * a) }

            var gW1 = X.transposeTimes(dZ1); addScaled(&gW1, W1, l2)
            let gb1 = colSums(dZ1)
            var gW2 = A1.transposeTimes(dZ2); addScaled(&gW2, W2, l2)
            let gb2 = colSums(dZ2)
            var gW3 = A2.transposeTimes(G); addScaled(&gW3, W3, l2)
            var gWs = X.transposeTimes(G); addScaled(&gWs, Ws, l2skip)
            let gb = colSums(G)

            let lrT = lr * (0.1 + 0.45 * (1.0 + Foundation.cos(Double.pi * Double(step) / Double(iters))))
            adam(&W1, gW1, &mW1, &vW1, step, lrT, beta1, beta2, eps)
            adamV(&b1, gb1, &mb1, &vb1, step, lrT, beta1, beta2, eps)
            adam(&W2, gW2, &mW2, &vW2, step, lrT, beta1, beta2, eps)
            adamV(&b2, gb2, &mb2, &vb2, step, lrT, beta1, beta2, eps)
            adam(&W3, gW3, &mW3, &vW3, step, lrT, beta1, beta2, eps)
            adam(&Ws, gWs, &mWs, &vWs, step, lrT, beta1, beta2, eps)
            adamV(&b, gb, &mb, &vb, step, lrT, beta1, beta2, eps)
        }
    }

    // helpers
    private func addScaled(_ g: inout Mat, _ p: Mat, _ s: Double) {
        for i in 0..<g.data.count { g.data[i] += s * p.data[i] }
    }
    private func colSums(_ m: Mat) -> [Double] {
        var s = [Double](repeating: 0, count: m.cols)
        for i in 0..<m.rows { for j in 0..<m.cols { s[j] += m[i, j] } }
        return s
    }
    private func adam(_ p: inout Mat, _ g: Mat, _ m: inout Mat, _ v: inout Mat,
                      _ step: Int, _ lrT: Double, _ b1: Double, _ b2: Double, _ eps: Double) {
        let bc1 = 1 - Foundation.pow(b1, Double(step)), bc2 = 1 - Foundation.pow(b2, Double(step))
        for i in 0..<p.data.count {
            m.data[i] = b1 * m.data[i] + (1 - b1) * g.data[i]
            v.data[i] = b2 * v.data[i] + (1 - b2) * g.data[i] * g.data[i]
            p.data[i] -= (lrT / bc1) * m.data[i] / ((v.data[i] / bc2).squareRoot() + eps)
        }
    }
    private func adamV(_ p: inout [Double], _ g: [Double], _ m: inout [Double], _ v: inout [Double],
                       _ step: Int, _ lrT: Double, _ b1: Double, _ b2: Double, _ eps: Double) {
        let bc1 = 1 - Foundation.pow(b1, Double(step)), bc2 = 1 - Foundation.pow(b2, Double(step))
        for i in 0..<p.count {
            m[i] = b1 * m[i] + (1 - b1) * g[i]
            v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]
            p[i] -= (lrT / bc1) * m[i] / ((v[i] / bc2).squareRoot() + eps)
        }
    }
}

// Codable snapshot of one network.
struct MLPData: Codable {
    var W1: [Double], b1: [Double], W2: [Double], b2: [Double]
    var W3: [Double], Ws: [Double], b: [Double]
    var nIn: Int, h1: Int, h2: Int, nOut: Int
}

extension MLPRegressor {
    func snapshot() -> MLPData {
        MLPData(W1: W1.data, b1: b1, W2: W2.data, b2: b2, W3: W3.data, Ws: Ws.data, b: b,
                nIn: nIn, h1: h1, h2: h2, nOut: nOut)
    }
    static func from(_ d: MLPData) -> MLPRegressor {
        let net = MLPRegressor(nIn: d.nIn, nOut: d.nOut, hidden: (d.h1, d.h2))
        net.W1 = Mat(rows: d.nIn, cols: d.h1, data: d.W1)
        net.b1 = d.b1
        net.W2 = Mat(rows: d.h1, cols: d.h2, data: d.W2)
        net.b2 = d.b2
        net.W3 = Mat(rows: d.h2, cols: d.nOut, data: d.W3)
        net.Ws = Mat(rows: d.nIn, cols: d.nOut, data: d.Ws)
        net.b = d.b
        return net
    }
}

// MARK: - affine correction (quick re-adjust)

struct AffineCorrection: Codable {
    var A: [Double] = [1, 0, 0, 1]   // 2x2 row-major
    var b: [Double] = [0, 0]

    func apply(_ p: SIMD2<Double>) -> SIMD2<Double> {
        SIMD2(A[0] * p.x + A[1] * p.y + b[0], A[2] * p.x + A[3] * p.y + b[1])
    }

    static func fit(pred: [SIMD2<Double>], target: [SIMD2<Double>], strength: Double = 0.15) -> AffineCorrection {
        let n = pred.count
        guard n >= 2 else { return AffineCorrection() }
        let pm = SIMD2(pred.map { $0.x }.mean(), pred.map { $0.y }.mean())
        let tm = SIMD2(target.map { $0.x }.mean(), target.map { $0.y }.mean())
        var Pc = Mat(n, 2), Tc = Mat(n, 2)
        for i in 0..<n {
            Pc[i, 0] = pred[i].x - pm.x; Pc[i, 1] = pred[i].y - pm.y
            Tc[i, 0] = target[i].x - tm.x; Tc[i, 1] = target[i].y - tm.y
        }
        let PtP = Pc.transposeTimes(Pc)
        let spread = (PtP[0, 0] + PtP[1, 1]) / 2.0
        let mu = Swift.max(strength * spread, 1e-6)
        var lhs = PtP; lhs[0, 0] += mu; lhs[1, 1] += mu
        var rhs = Pc.transposeTimes(Tc); rhs[0, 0] += mu; rhs[1, 1] += mu
        let At = Linalg.solve(lhs, rhs)          // solves for Aᵀ (2x2)
        // A = Atᵀ
        var A = [At[0, 0], At[1, 0], At[0, 1], At[1, 1]]
        // clamp singular values to keep it physically plausible (0.6..1.6)
        A = clampSingularValues(A, lo: 0.6, hi: 1.6)
        let bx = tm.x - (A[0] * pm.x + A[1] * pm.y)
        let by = tm.y - (A[2] * pm.x + A[3] * pm.y)
        return AffineCorrection(A: A, b: [bx, by])
    }

    private static func clampSingularValues(_ A: [Double], lo: Double, hi: Double) -> [Double] {
        // 2x2 SVD via eigen-decomposition of AᵀA (closed form).
        let a = A[0], b = A[1], c = A[2], d = A[3]
        let e = (a + d) / 2, f = (a - d) / 2, g = (c + b) / 2, h = (c - b) / 2
        let q = (e * e + h * h).squareRoot(), r = (f * f + g * g).squareRoot()
        var s1 = q + r, s2 = abs(q - r)
        let a1 = atan2(g, f), a2 = atan2(h, e)
        let theta = (a2 - a1) / 2, phi = (a2 + a1) / 2
        s1 = Swift.max(lo, Swift.min(hi, s1)); s2 = Swift.max(lo, Swift.min(hi, s2))
        let ct = Foundation.cos(theta), st = Foundation.sin(theta)
        let cp = Foundation.cos(phi), sp = Foundation.sin(phi)
        // A = U S Vᵀ with U = R(phi), V = R(theta)
        let u = [cp, -sp, sp, cp]
        let v = [ct, -st, st, ct]              // Vᵀ rows
        // A = U * diag(s1,s2) * Vᵀ
        let m00 = u[0] * s1, m01 = u[1] * s2
        let m10 = u[2] * s1, m11 = u[3] * s2
        return [m00 * v[0] + m01 * v[2], m00 * v[1] + m01 * v[3],
                m10 * v[0] + m11 * v[2], m10 * v[1] + m11 * v[3]]
    }
}

// MARK: - GazeNet (scalers + ensemble + correction)

struct TrainReport {
    var l2: Double
    var cvErrorPx: Double
    var linearCvErrorPx: Double
    var trainErrorPx: Double
    var nSamples: Int
    var nGroups: Int
}

struct GazeNetData: Codable {
    var xScaler: Scaler
    var yScaler: Scaler
    var nets: [MLPData]
    var correction: AffineCorrection
    var dropColumns: [Int]? = nil
}

final class GazeNet {
    static let HIDDEN = (32, 16)
    static let L2_GRID = [1e-3, 1e-2, 1e-1]

    let xScaler: Scaler
    let yScaler: Scaler
    let nets: [MLPRegressor]
    var correction: AffineCorrection
    /// Feature-vector columns the model was trained WITHOUT; inference projects
    /// the incoming full vector down to the same columns. Empty = use all.
    var dropColumns: [Int]

    init(xScaler: Scaler, yScaler: Scaler, nets: [MLPRegressor],
         correction: AffineCorrection = AffineCorrection(), dropColumns: [Int] = []) {
        self.xScaler = xScaler; self.yScaler = yScaler; self.nets = nets
        self.correction = correction; self.dropColumns = dropColumns
    }

    /// Project a full feature vector to the columns this model uses.
    private func project(_ x: [Double]) -> [Double] {
        guard !dropColumns.isEmpty else { return x }
        let drop = Set(dropColumns)
        return x.enumerated().filter { !drop.contains($0.offset) }.map { $0.element }
    }

    func predictUncorrected(_ x: [Double]) -> SIMD2<Double> {
        let xp = project(x)
        var X = Mat(rows: 1, cols: xp.count, data: xp)
        X = xScaler.transform(X)
        for i in 0..<X.data.count { X.data[i] = Swift.max(-6, Swift.min(6, X.data[i])) }
        var sum = [0.0, 0.0]
        for net in nets { let o = net.forward(X); sum[0] += o[0, 0]; sum[1] += o[0, 1] }
        let mean = Mat(rows: 1, cols: 2, data: [sum[0] / Double(nets.count), sum[1] / Double(nets.count)])
        let inv = yScaler.inverse(mean)
        return SIMD2(inv[0, 0], inv[0, 1])
    }

    func predict(_ x: [Double]) -> SIMD2<Double> { correction.apply(predictUncorrected(x)) }

    func snapshot() -> GazeNetData {
        GazeNetData(xScaler: xScaler, yScaler: yScaler, nets: nets.map { $0.snapshot() },
                    correction: correction, dropColumns: dropColumns)
    }
    static func from(_ d: GazeNetData) -> GazeNet {
        GazeNet(xScaler: d.xScaler, yScaler: d.yScaler, nets: d.nets.map { MLPRegressor.from($0) },
                correction: d.correction, dropColumns: d.dropColumns ?? [])
    }

    /// Train an ensemble, choosing weight decay by grouped cross-validation.
    static func train(X: Mat, Y: Mat, groups: [Int], weights: [Double]? = nil,
                      holdout: [Bool]? = nil, minStd: [Double]? = nil,
                      ensemble: Int = 3, iters: Int = 600, folds: Int = 3, seed: UInt64 = 0) -> (GazeNet, TrainReport) {
        let n = X.rows
        let w = weights ?? [Double](repeating: 1, count: n)
        let canHold = holdout ?? [Bool](repeating: true, count: n)

        func rowsOf(_ M: Mat, _ idx: [Int]) -> Mat {
            var r = Mat(idx.count, M.cols)
            for (ri, i) in idx.enumerated() { for j in 0..<M.cols { r[ri, j] = M[i, j] } }
            return r
        }
        func fitEval(train: [Int], test: [Int], l2: Double?) -> [Double] {
            let xs = Scaler.fit(rowsOf(X, train), minStd: minStd)
            let ys = Scaler.fit(rowsOf(Y, train))
            let Xt = xs.transform(rowsOf(X, train)), Yt = ys.transform(rowsOf(Y, train))
            var Xv = xs.transform(rowsOf(X, test))
            for i in 0..<Xv.data.count { Xv.data[i] = Swift.max(-6, Swift.min(6, Xv.data[i])) }
            let Ytrue = rowsOf(Y, test)
            var predOrig: Mat
            if let l2 = l2 {
                let net = MLPRegressor(nIn: X.cols, nOut: 2, hidden: HIDDEN, seed: seed)
                net.fit(Xt, Yt, weights: train.map { w[$0] }, l2: l2, iters: iters)
                predOrig = ys.inverse(net.forward(Xv))
            } else {
                let (W, b) = ridgeFit(Xt, Yt, lam: 1e-2, weights: train.map { w[$0] })
                predOrig = ys.inverse(Xv.matmul(W).addingRowVector(b))
            }
            var errs = [Double](repeating: 0, count: test.count)
            for i in 0..<test.count {
                let dx = predOrig[i, 0] - Ytrue[i, 0], dy = predOrig[i, 1] - Ytrue[i, 1]
                errs[i] = (dx * dx + dy * dy).squareRoot()
            }
            return errs
        }

        let holdGroups = Array(Set((0..<n).filter { canHold[$0] }.map { groups[$0] }))
        let nGroups = Set(groups).count
        var bestL2 = L2_GRID[L2_GRID.count / 2]
        var cvErr = Double.nan, linErr = Double.nan
        var candidates: [Double: Double] = [:]

        if holdGroups.count >= 6 {
            let foldSets = groupedFolds(groups.enumerated().filter { canHold[$0.offset] }.map { $0.element },
                                        k: folds, seed: seed)
            // Recompute test sets as sample indices restricted to holdable samples.
            let uniqHold = Set(holdGroups)
            let foldGroupSets: [[Int]] = {
                var rng = SeededRNG(seed: seed)
                var g = Array(uniqHold).sorted(); g.shuffle(using: &rng)
                let kk = Swift.max(2, Swift.min(folds, g.count))
                var out: [[Int]] = []
                for i in 0..<kk { var s: [Int] = []; var idx = i; while idx < g.count { s.append(g[idx]); idx += kk }; out.append(s) }
                return out
            }()
            _ = foldSets
            let allIdx = Array(0..<n)
            let testSets: [[Int]] = foldGroupSets.map { members in
                let set = Set(members)
                return (0..<n).filter { canHold[$0] && set.contains(groups[$0]) }
            }
            func cv(_ l2: Double?) -> Double {
                var errs: [Double] = [], ws: [Double] = []
                for test in testSets {
                    let train = allIdx.filter { !test.contains($0) }
                    errs.append(contentsOf: fitEval(train: train, test: test, l2: l2))
                    ws.append(contentsOf: test.map { w[$0] })
                }
                let sw = ws.reduce(0, +)
                let wsum = zip(errs, ws).reduce(0.0) { $0 + $1.0 * $1.1 }
                return sw > 0 ? wsum / sw : errs.mean()
            }
            for l2 in L2_GRID { candidates[l2] = cv(l2) }
            bestL2 = L2_GRID.min { candidates[$0]! < candidates[$1]! }!
            cvErr = candidates[bestL2]!
            linErr = cv(nil)
        }

        let xs = Scaler.fit(X, minStd: minStd)
        let ys = Scaler.fit(Y)
        let Xs = xs.transform(X), Ys = ys.transform(Y)
        var nets: [MLPRegressor] = []
        for i in 0..<ensemble {
            let net = MLPRegressor(nIn: X.cols, nOut: 2, hidden: HIDDEN, seed: seed &+ UInt64(101 * i))
            net.fit(Xs, Ys, weights: w, l2: bestL2, iters: iters)
            nets.append(net)
        }
        let model = GazeNet(xScaler: xs, yScaler: ys, nets: nets)
        // train error
        var num = 0.0, den = 0.0
        for i in 0..<n {
            let p = model.predict(X.row(i))
            let dx = p.x - Y[i, 0], dy = p.y - Y[i, 1]
            num += w[i] * (dx * dx + dy * dy).squareRoot(); den += w[i]
        }
        let report = TrainReport(l2: bestL2, cvErrorPx: cvErr, linearCvErrorPx: linErr,
                                 trainErrorPx: den > 0 ? num / den : .nan, nSamples: n, nGroups: nGroups)
        return (model, report)
    }
}
