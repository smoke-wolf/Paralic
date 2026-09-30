// Settings.swift — fine-grained, persisted control over tracking behaviour.
//
// The defaults are tuned for "fast and precise": steady while you fixate,
// instant when your eyes jump. Every field is individually adjustable so you
// can dial in responsiveness vs. stability exactly.

import Foundation

/// Cursor responsiveness. Ties together the One-Euro filter, a saccade
/// detector (bypass smoothing on fast eye jumps) and a small velocity lead
/// (counter end-to-end latency) so the cursor feels immediate yet steady.
struct Responsiveness: Codable, Equatable {
    /// One-Euro minimum cutoff (Hz). Lower = smoother/steadier when still.
    var minCutoff: Double = 1.1
    /// One-Euro speed coefficient. Higher = more responsive during motion.
    var beta: Double = 0.02
    /// Derivative cutoff (Hz).
    var dCutoff: Double = 1.0
    /// Eye speed (screen fraction / second) above which we treat motion as a
    /// saccade and blend the cursor straight to the raw estimate.
    var saccadeThreshold: Double = 2.2
    /// How hard we snap to raw during a saccade (0..1).
    var saccadeSnap: Double = 0.9
    /// Velocity lead in milliseconds — predicts slightly ahead to hide latency.
    var predictionLeadMs: Double = 16
    /// Ignore sub-pixel jitter below this (screen fraction) while fixating.
    var deadZone: Double = 0.0025

    /// A single 0..1 "Speed" dial that sets a sensible fast/precise balance.
    /// 0 = maximally smooth & precise, 1 = maximally quick & responsive.
    static func fromSpeed(_ s: Double) -> Responsiveness {
        let s = max(0, min(1, s))
        return Responsiveness(
            minCutoff: 0.6 + 2.4 * s,
            beta: 0.004 + 0.045 * s,
            dCutoff: 1.0,
            saccadeThreshold: 3.0 - 1.4 * s,
            saccadeSnap: 0.75 + 0.2 * s,
            predictionLeadMs: 6 + 26 * s,
            deadZone: 0.004 - 0.003 * s
        )
    }

    /// Approximate the inverse: recover a 0..1 speed from the current settings.
    var speed: Double {
        max(0, min(1, (minCutoff - 0.6) / 2.4))
    }

    static let precise = fromSpeed(0.25)
    static let balanced = fromSpeed(0.5)
    static let fast = fromSpeed(0.85)
}

enum SnapStrength: String, Codable, CaseIterable {
    case off, light, normal, strong
    var pull: Double { [.off: 0, .light: 0.35, .normal: 0.6, .strong: 0.82][self]! }
    var radius: Double { [.off: 0, .light: 0.10, .normal: 0.14, .strong: 0.20][self]! } // × screen diagonal
}

enum BlinkSensitivity: String, Codable, CaseIterable {
    case low, normal, high
    var value: Double { [.low: 0.40, .normal: 0.30, .high: 0.22][self]! }
}

enum DoubleBlinkSpeed: String, Codable, CaseIterable {
    case fast, normal, relaxed
    var gapMs: Double { [.fast: 380, .normal: 550, .relaxed: 800][self]! }
}

/// Everything the user can tune, persisted to the profile.
struct Settings: Codable, Equatable {
    var responsiveness = Responsiveness.balanced
    var snap: SnapStrength = .normal
    var blinkSensitivity: BlinkSensitivity = .normal
    var doubleBlinkSpeed: DoubleBlinkSpeed = .normal
    var scrollSpeed: Double = 1.0
    var driftCorrection = true          // each click nudges an affine correction
    var cursorSize: Double = 1.0
    var showCameraPreview = true
    var sounds = true
    var speechRate: Double = 0.5

    static let `default` = Settings()
}
