// BlinkDetector.swift — blink / double-blink detection (port of blink.py).
//
// Input is a per-frame eye *closure* score in 0 (open) … 1 (closed). Thresholds
// adapt to your eyes and to lids dropping when you look down: we track the
// median closure of recent open-eye frames (the baseline) and call the eyes
// closed when the score rises well above it. A blink lasts between
// min/maxClosedMs; a double blink is two blinks with a short pause between.

import Foundation

struct BlinkConfig {
    var sensitivity: Double = 0.30      // rise above baseline (fraction of headroom) to count as closed
    var minClosedMs: Double = 25
    var maxClosedMs: Double = 700
    var doubleGapMs: Double = 550
    var minGapMs: Double = 30
    var baselineWindowS: Double = 1.6
    var initialBaseline: Double = 0.2
    var missingResetMs: Double = 600
}

enum BlinkEventType { case blink, doubleBlink, blinkExpired }

struct BlinkEvent {
    var type: BlinkEventType
    var t: Double
    var count: Int = 0
    var firstStart: Double? = nil
}

struct BlinkStatus {
    var closing: Bool
    var closed: Bool
    var baseline: Double
    var closeThreshold: Double
    var openThreshold: Double
    var pending: Bool
}

final class BlinkDetector {
    var config: BlinkConfig
    private var closed = false
    private var closedSince = 0.0
    private var pending: (start: Double, end: Double)?
    private var samples: [(Double, Double)] = []
    private var baseline: Double
    private var lastSeen: Double?
    private var closure = 0.0

    init(config: BlinkConfig = BlinkConfig()) {
        self.config = config
        self.baseline = config.initialBaseline
    }

    func thresholds() -> (close: Double, open: Double) {
        let b = baseline
        let tClose = Swift.max(0.30, Swift.min(0.92, b + config.sensitivity * (1 - b)))
        let tOpen = b + 0.5 * (tClose - b)
        return (tClose, tOpen)
    }

    func status() -> BlinkStatus {
        let (tc, to) = thresholds()
        return BlinkStatus(closing: closed || closure >= to, closed: closed, baseline: baseline,
                           closeThreshold: tc, openThreshold: to, pending: pending != nil)
    }

    func reset() {
        closed = false; pending = nil; samples.removeAll()
        baseline = config.initialBaseline; lastSeen = nil; closure = 0
    }

    /// Call when no face was found in a frame.
    func updateMissing(t: Double) -> [BlinkEvent] {
        var events: [BlinkEvent] = []
        if let ls = lastSeen, (t - ls) * 1000 > config.missingResetMs {
            closed = false
            if pending != nil { events.append(BlinkEvent(type: .blinkExpired, t: t)) }
            pending = nil; closure = 0
        }
        return events
    }

    func update(t: Double, closure c: Double) -> [BlinkEvent] {
        var events: [BlinkEvent] = []
        lastSeen = t; closure = c
        let (tClose, tOpen) = thresholds()
        if !closed {
            addSample(t, c)
            if c >= tClose { closed = true; closedSince = t }
            else if let p = pending, (t - p.end) * 1000 > config.doubleGapMs {
                pending = nil; events.append(BlinkEvent(type: .blinkExpired, t: t))
            }
        } else {
            let durationMs = (t - closedSince) * 1000
            if durationMs > config.maxClosedMs { addSample(t, c) }   // let long closures teach the baseline
            if c < tOpen { closed = false; events.append(contentsOf: onReopen(t, durationMs)) }
        }
        return events
    }

    private func onReopen(_ t: Double, _ durationMs: Double) -> [BlinkEvent] {
        guard config.minClosedMs <= durationMs && durationMs <= config.maxClosedMs else {
            let expired = pending != nil; pending = nil
            return expired ? [BlinkEvent(type: .blinkExpired, t: t)] : []
        }
        let blink = (start: closedSince, end: t)
        if let p = pending {
            let gapMs = (blink.start - p.end) * 1000
            if config.minGapMs <= gapMs && gapMs <= config.doubleGapMs {
                pending = nil
                return [BlinkEvent(type: .blink, t: t, count: 2, firstStart: p.start),
                        BlinkEvent(type: .doubleBlink, t: t, count: 2, firstStart: p.start)]
            }
        }
        pending = blink
        return [BlinkEvent(type: .blink, t: t, count: 1, firstStart: blink.start)]
    }

    private func addSample(_ t: Double, _ c: Double) {
        samples.append((t, c))
        let horizon = t - config.baselineWindowS
        while let first = samples.first, first.0 < horizon { samples.removeFirst() }
        if samples.count >= 5 { baseline = [Double].median(samples.map { $0.1 }) }
    }
}
