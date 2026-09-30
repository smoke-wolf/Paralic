// ABSelection.swift — per-user A/B model selection for personalization.
//
// Accurate personalization is not one-size-fits-all: different people (and
// seating, lighting, glasses, head motion) favour different network capacity,
// regularisation and whether head pose helps. `autoTrain` runs an A/B bake-off
// over candidate configurations, scoring each by *grouped* cross-validation on
// the user's own calibration data (whole calibration dots held out, so the
// score measures interpolation to unseen screen positions, not memorisation),
// and keeps the winner. The full comparison is returned so the UI can show it.

import Foundation

/// One candidate model configuration entered into the bake-off.
struct ModelConfig: Equatable {
    var name: String
    var hidden: (Int, Int)
    var iters: Int
    var ensemble: Int
    var useHeadPose: Bool          // drop head-pose features if they only add noise for this user
    var l2Grid: [Double]

    static func == (a: ModelConfig, b: ModelConfig) -> Bool { a.name == b.name }

    /// The default grid of candidates that compete for each user.
    static let grid: [ModelConfig] = [
        ModelConfig(name: "linear",        hidden: (1, 1),   iters: 0,   ensemble: 1, useHeadPose: true,  l2Grid: [1e-2]),
        ModelConfig(name: "small",         hidden: (16, 8),  iters: 500, ensemble: 3, useHeadPose: true,  l2Grid: GazeNet.L2_GRID),
        ModelConfig(name: "standard",      hidden: (32, 16), iters: 600, ensemble: 3, useHeadPose: true,  l2Grid: GazeNet.L2_GRID),
        ModelConfig(name: "large",         hidden: (48, 24), iters: 800, ensemble: 3, useHeadPose: true,  l2Grid: GazeNet.L2_GRID),
        ModelConfig(name: "no-head-pose",  hidden: (32, 16), iters: 600, ensemble: 3, useHeadPose: false, l2Grid: GazeNet.L2_GRID),
    ]
}

struct ABResult {
    var config: String
    var cvErrorPx: Double        // grouped-CV error (pixels) — the selection metric
    var cvErrorCm: Double
    var trainErrorPx: Double
    var chosen: Bool
}

struct ABReport {
    var results: [ABResult]
    var winner: String
    var winnerCvErrorCm: Double
    var linearBaselineCm: Double
    /// Improvement of the winner over the plain linear baseline (higher = better).
    var improvementVsLinearPct: Double
}

extension GazeNet {

    /// Columns of the feature vector that are head-pose related (yaw,pitch,roll,tx,ty,tz).
    /// Matches FeatureLayout in Features.swift.
    static let headPoseColumns: Set<Int> = [14, 15, 16, 17, 18, 19]

    private static func project(_ X: Mat, dropColumns drop: Set<Int>) -> Mat {
        if drop.isEmpty { return X }
        let keep = (0..<X.cols).filter { !drop.contains($0) }
        var r = Mat(X.rows, keep.count)
        for i in 0..<X.rows { for (jj, j) in keep.enumerated() { r[i, jj] = X[i, j] } }
        return r
    }

    /// A/B bake-off across `configs`, selecting the winner by grouped CV.
    /// `pxPerCm` converts the pixel error to centimetres for reporting.
    static func autoTrain(X: Mat, Y: Mat, groups: [Int], weights: [Double]? = nil,
                          holdout: [Bool]? = nil, minStd: [Double]? = nil,
                          pxPerCm: Double = 37.8, configs: [ModelConfig] = ModelConfig.grid,
                          seed: UInt64 = 0) -> (GazeNet, TrainReport, ABReport) {

        struct Trained { let model: GazeNet; let report: TrainReport; let cfg: ModelConfig }
        var trained: [Trained] = []
        var linearCm = Double.nan

        for cfg in configs {
            let drop = cfg.useHeadPose ? Set<Int>() : headPoseColumns
            let dropSorted = drop.sorted()
            let Xp = project(X, dropColumns: drop)
            let mstd = minStd.map { ms -> [Double] in
                if drop.isEmpty { return ms }
                return (0..<ms.count).filter { !drop.contains($0) }.map { ms[$0] }
            }
            let (m, r) = GazeNet.train(X: Xp, Y: Y, groups: groups, weights: weights,
                                       holdout: holdout, minStd: mstd,
                                       ensemble: cfg.ensemble, iters: cfg.iters, folds: 3, seed: seed)
            m.dropColumns = dropSorted    // so inference projects incoming vectors to match
            if cfg.name == "linear", r.linearCvErrorPx.isFinite { linearCm = r.linearCvErrorPx / pxPerCm }
            if linearCm.isNaN && r.linearCvErrorPx.isFinite { linearCm = r.linearCvErrorPx / pxPerCm }
            trained.append(Trained(model: m, report: r, cfg: cfg))
        }

        // Score: prefer the lowest grouped-CV error. When CV is unavailable
        // (too few dots) fall back to training error (with a small penalty for
        // bigger models to avoid overfitting a tiny calibration).
        func score(_ t: Trained) -> Double {
            if t.report.cvErrorPx.isFinite { return t.report.cvErrorPx }
            let sizePenalty = Double(t.cfg.hidden.0 + t.cfg.hidden.1) * 0.05
            return t.report.trainErrorPx + sizePenalty
        }
        let winner = trained.min { score($0) < score($1) }!

        let results = trained.map { t -> ABResult in
            let cvPx = t.report.cvErrorPx
            return ABResult(config: t.cfg.name,
                            cvErrorPx: cvPx,
                            cvErrorCm: cvPx.isFinite ? cvPx / pxPerCm : .nan,
                            trainErrorPx: t.report.trainErrorPx,
                            chosen: t.cfg == winner.cfg)
        }
        let winnerCm = (winner.report.cvErrorPx.isFinite ? winner.report.cvErrorPx : winner.report.trainErrorPx) / pxPerCm
        let improvement = (linearCm.isFinite && winnerCm > 0) ? (linearCm - winnerCm) / linearCm * 100.0 : 0.0
        let ab = ABReport(results: results, winner: winner.cfg.name, winnerCvErrorCm: winnerCm,
                          linearBaselineCm: linearCm, improvementVsLinearPct: improvement)
        return (winner.model, winner.report, ab)
    }
}
