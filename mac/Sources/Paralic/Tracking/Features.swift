// Features.swift — Vision face landmarks → the 20-number feature vector.
//
// Mirrors the Python features.py contract so GazeNet and the A/B head-pose
// column mask line up exactly:
//
//   0  r_dx     1  r_dy     2  l_dx     3  l_dy       iris offset per eye (eye widths)
//   4  r_open   5  l_open                             lid opening per eye (eye widths)
//   6..13       look in/out/up/down  (L then R)       gaze "blendshapes" (geometric on macOS)
//   14 yaw      15 pitch    16 roll                   head rotation (radians)
//   17 tx       18 ty       19 tz                     head position proxy (box centre + size)
//
// macOS Vision provides pupils, eye contours and roll/yaw/pitch but no
// blendshapes, so the look features are derived from iris geometry.

import Foundation
#if canImport(Vision)
import Vision
import CoreGraphics

enum FeatureLayout {
    static let count = 20
    static let names = [
        "r_dx", "r_dy", "l_dx", "l_dy", "r_open", "l_open",
        "look_in_l", "look_out_l", "look_up_l", "look_down_l",
        "look_in_r", "look_out_r", "look_up_r", "look_down_r",
        "yaw", "pitch", "roll", "tx", "ty", "tz",
    ]
    static let headPoseColumns = [14, 15, 16, 17, 18, 19]
    static let minStd: [Double] = [
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
        0.035, 0.035, 0.035, 0.02, 0.02, 0.03,
    ]
}

struct FrameFeatures {
    var vector: [Double]          // count == FeatureLayout.count
    var closure: Double           // 0 open … 1 closed (blink signal)
    var faceBox: CGRect           // normalised
    var leftEye: [CGPoint]        // normalised contour (for the camera overlay)
    var rightEye: [CGPoint]
    var leftIris: CGPoint
    var rightIris: CGPoint
}

enum FaceFeatures {

    /// Extract features for one detected face. Returns nil if the landmarks we
    /// need (pupils + eye contours) are missing.
    static func extract(from face: VNFaceObservation, imageSize: CGSize) -> FrameFeatures? {
        guard let lm = face.landmarks,
              let leftEye = lm.leftEye, let rightEye = lm.rightEye else { return nil }

        // Vision landmark points are normalised to the face bounding box.
        let box = face.boundingBox
        func toNorm(_ p: CGPoint) -> CGPoint {   // → whole-image normalised coords
            CGPoint(x: box.origin.x + p.x * box.width, y: box.origin.y + p.y * box.height)
        }
        let leftPts = leftEye.normalizedPoints.map(toNorm)
        let rightPts = rightEye.normalizedPoints.map(toNorm)
        let leftIris = centroid(lm.leftPupil?.normalizedPoints.map(toNorm)) ?? centroid(leftPts)!
        let rightIris = centroid(lm.rightPupil?.normalizedPoints.map(toNorm)) ?? centroid(rightPts)!

        let r = eyeMeasure(contour: rightPts, iris: rightIris)
        let l = eyeMeasure(contour: leftPts, iris: leftIris)

        let yaw = (face.yaw?.doubleValue) ?? 0
        let pitch = (face.pitch?.doubleValue) ?? 0
        let roll = (face.roll?.doubleValue) ?? 0
        // Head position proxy: face-box centre and a size→distance signal.
        let tx = Double(box.midX - 0.5)
        let ty = Double(box.midY - 0.5)
        let tz = Double(1.0 - box.width)    // smaller box ≈ further away

        // Geometric "look" features from iris offset (no blendshapes on macOS).
        func clamp01(_ x: Double) -> Double { Swift.max(0, Swift.min(1, x)) }
        let lLookOut = clamp01(0.5 + l.dx), lLookIn = clamp01(0.5 - l.dx)
        let lLookUp = clamp01(0.5 - l.dy), lLookDown = clamp01(0.5 + l.dy)
        let rLookOut = clamp01(0.5 - r.dx), rLookIn = clamp01(0.5 + r.dx)
        let rLookUp = clamp01(0.5 - r.dy), rLookDown = clamp01(0.5 + r.dy)

        let vector: [Double] = [
            r.dx, r.dy, l.dx, l.dy, r.aperture, l.aperture,
            lLookIn, lLookOut, lLookUp, lLookDown,
            rLookIn, rLookOut, rLookUp, rLookDown,
            yaw, pitch, roll, tx, ty, tz,
        ]

        func geomClosure(_ ap: Double) -> Double { Swift.max(0, Swift.min(1, (0.30 - ap) / 0.24)) }
        let closure = 0.5 * (geomClosure(l.aperture) + geomClosure(r.aperture))

        return FrameFeatures(vector: vector, closure: closure, faceBox: box,
                             leftEye: leftPts, rightEye: rightPts, leftIris: leftIris, rightIris: rightIris)
    }

    private struct EyeMeasure { var dx: Double; var dy: Double; var aperture: Double }

    /// Iris offset (in eye widths) along/across the corner-to-corner eye axis,
    /// and the lid aperture. Corners are the extreme-x contour points.
    private static func eyeMeasure(contour pts: [CGPoint], iris: CGPoint) -> EyeMeasure {
        guard pts.count >= 4 else { return EyeMeasure(dx: 0, dy: 0, aperture: 0.3) }
        let a = pts.min { $0.x < $1.x }!     // left corner
        let b = pts.max { $0.x < $1.x }!     // right corner
        let axis = CGPoint(x: b.x - a.x, y: b.y - a.y)
        let width = Swift.max(1e-6, (axis.x * axis.x + axis.y * axis.y).squareRoot())
        let ex = CGPoint(x: axis.x / width, y: axis.y / width)
        let ey = CGPoint(x: -ex.y, y: ex.x)  // +90°, "down" in image space
        let centre = CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2)
        let rel = CGPoint(x: iris.x - centre.x, y: iris.y - centre.y)
        let dx = Double((rel.x * ex.x + rel.y * ex.y) / width)
        let dy = Double((rel.x * ey.x + rel.y * ey.y) / width)
        // aperture: spread of contour points perpendicular to the axis.
        var lo = Double.greatestFiniteMagnitude, hi = -Double.greatestFiniteMagnitude
        for p in pts {
            let v = Double(((p.x - centre.x) * ey.x + (p.y - centre.y) * ey.y))
            lo = Swift.min(lo, v); hi = Swift.max(hi, v)
        }
        let aperture = (hi - lo) / Double(width)
        return EyeMeasure(dx: dx, dy: dy, aperture: aperture)
    }

    private static func centroid(_ pts: [CGPoint]?) -> CGPoint? {
        guard let pts = pts, !pts.isEmpty else { return nil }
        let sx = pts.reduce(0) { $0 + $1.x }, sy = pts.reduce(0) { $0 + $1.y }
        return CGPoint(x: sx / CGFloat(pts.count), y: sy / CGFloat(pts.count))
    }
}
#endif
