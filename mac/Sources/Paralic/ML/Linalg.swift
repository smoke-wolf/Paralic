// Linalg.swift — small dense linear algebra for the GazeNet port.
//
// Matrices are row-major `[Double]` wrapped in `Mat`. Matrix products use
// Accelerate's BLAS (cblas_dgemm); the only linear *solve* we need is tiny
// (<= 20x20 ridge systems, 2x2 affine), so a plain Gauss–Jordan solver is used.

import Foundation
import Accelerate

/// Row-major dense matrix of Doubles.
struct Mat {
    var rows: Int
    var cols: Int
    var data: [Double]

    init(_ rows: Int, _ cols: Int, _ value: Double = 0) {
        self.rows = rows; self.cols = cols
        self.data = [Double](repeating: value, count: rows * cols)
    }

    init(rows: Int, cols: Int, data: [Double]) {
        precondition(data.count == rows * cols, "Mat size mismatch")
        self.rows = rows; self.cols = cols; self.data = data
    }

    /// Build from an array of row vectors.
    init(rowsOf vectors: [[Double]]) {
        let r = vectors.count
        let c = r > 0 ? vectors[0].count : 0
        var d = [Double](); d.reserveCapacity(r * c)
        for v in vectors { precondition(v.count == c); d.append(contentsOf: v) }
        self.init(rows: r, cols: c, data: d)
    }

    @inline(__always) subscript(_ i: Int, _ j: Int) -> Double {
        get { data[i * cols + j] }
        set { data[i * cols + j] = newValue }
    }

    func row(_ i: Int) -> [Double] { Array(data[i * cols ..< (i + 1) * cols]) }

    static func identity(_ n: Int) -> Mat {
        var m = Mat(n, n); for i in 0..<n { m[i, i] = 1 }; return m
    }
}

extension Mat {
    /// C = self (m x k) * B (k x n)
    func matmul(_ B: Mat) -> Mat {
        precondition(cols == B.rows, "matmul shape mismatch \(rows)x\(cols) * \(B.rows)x\(B.cols)")
        let m = rows, k = cols, n = B.cols
        var C = Mat(m, n)
        self.data.withUnsafeBufferPointer { a in
            B.data.withUnsafeBufferPointer { b in
                C.data.withUnsafeMutableBufferPointer { c in
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(m), Int32(n), Int32(k),
                                1.0, a.baseAddress, Int32(k),
                                b.baseAddress, Int32(n),
                                0.0, c.baseAddress, Int32(n))
                }
            }
        }
        return C
    }

    /// C = selfᵀ * B
    func transposeTimes(_ B: Mat) -> Mat {
        precondition(rows == B.rows, "shape mismatch")
        let m = cols, k = rows, n = B.cols
        var C = Mat(m, n)
        self.data.withUnsafeBufferPointer { a in
            B.data.withUnsafeBufferPointer { b in
                C.data.withUnsafeMutableBufferPointer { c in
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                Int32(m), Int32(n), Int32(k),
                                1.0, a.baseAddress, Int32(cols),
                                b.baseAddress, Int32(n),
                                0.0, c.baseAddress, Int32(n))
                }
            }
        }
        return C
    }

    var transposed: Mat {
        var t = Mat(cols, rows)
        for i in 0..<rows { for j in 0..<cols { t[j, i] = self[i, j] } }
        return t
    }

    static func + (lhs: Mat, rhs: Mat) -> Mat {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        var r = lhs; vDSP.add(lhs.data, rhs.data, result: &r.data); return r
    }

    static func - (lhs: Mat, rhs: Mat) -> Mat {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        var r = lhs; vDSP.subtract(lhs.data, rhs.data, result: &r.data); return r
    }

    /// Add a row vector to every row.
    func addingRowVector(_ v: [Double]) -> Mat {
        precondition(v.count == cols)
        var r = self
        for i in 0..<rows { for j in 0..<cols { r[i, j] += v[j] } }
        return r
    }

    func scaled(_ s: Double) -> Mat {
        var r = self; vDSP.multiply(s, data, result: &r.data); return r
    }
}

enum Linalg {
    /// Solve A x = B for x, with A (n x n) and B (n x m). Gauss–Jordan with
    /// partial pivoting. Returns x (n x m). Falls back to a small ridge on the
    /// diagonal if A is singular.
    static func solve(_ A: Mat, _ B: Mat) -> Mat {
        precondition(A.rows == A.cols && A.rows == B.rows)
        let n = A.rows, m = B.cols
        var a = A, b = B
        for col in 0..<n {
            // pivot
            var piv = col
            var best = abs(a[col, col])
            for r in (col + 1)..<n where abs(a[r, col]) > best { best = abs(a[r, col]); piv = r }
            if best < 1e-12 { a[col, col] += 1e-9 } // nudge near-singular
            if piv != col {
                for j in 0..<n { a.data.swapAt(col * n + j, piv * n + j) }
                for j in 0..<m { b.data.swapAt(col * m + j, piv * m + j) }
            }
            let d = a[col, col]
            for j in 0..<n { a[col, j] /= d }
            for j in 0..<m { b[col, j] /= d }
            for r in 0..<n where r != col {
                let f = a[r, col]
                if f == 0 { continue }
                for j in 0..<n { a[r, j] -= f * a[col, j] }
                for j in 0..<m { b[r, j] -= f * b[col, j] }
            }
        }
        return b
    }
}

// MARK: - vector helpers

extension Array where Element == Double {
    func mean() -> Double { isEmpty ? 0 : reduce(0, +) / Double(count) }
    func std() -> Double {
        guard count > 0 else { return 0 }
        let m = mean()
        let v = reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(count)
        return v.squareRoot()
    }
    static func median(_ xs: [Double]) -> Double {
        guard !xs.isEmpty else { return 0 }
        let s = xs.sorted()
        let n = s.count
        return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}
