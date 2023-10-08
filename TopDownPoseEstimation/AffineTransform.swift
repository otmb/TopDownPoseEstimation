// Perspective transform from quadrilateral to quadrilateral in Swift
// Gary Bartos Jan 5, 2021
// https://rethunk.medium.com/perspective-transform-from-quadrilateral-to-quadrilateral-in-swift-5a9adf2175c3

import CoreGraphics
import Foundation

func radiansToDegrees(_ radians: CGFloat) -> CGFloat {
  180.0 * radians / CGFloat.pi
}

extension CGVector {
  init(_ point: CGPoint) {
    self.init(dx: point.x, dy: point.y)
  }
  
  static func + (lhs: CGVector, rhs: CGVector) -> CGVector {
    CGVector(dx: lhs.dx + rhs.dx, dy: lhs.dy + rhs.dy)
  }
  
  static func - (lhs: CGVector, rhs: CGVector) -> CGVector {
    CGVector(dx: lhs.dx - rhs.dx, dy: lhs.dy - rhs.dy)
  }
  
  static func * (_ vector: CGVector, _ scalar: CGFloat) -> CGVector {
    CGVector(dx: vector.dx * scalar, dy: vector.dy * scalar)
  }
  
  static func * (_ scalar: CGFloat, _ vector: CGVector) -> CGVector {
    CGVector(dx: vector.dx * scalar, dy: vector.dy * scalar)
  }
  
  static func * (lhs: CGVector, rhs: CGVector) -> CGFloat {
    lhs.dx * rhs.dx + lhs.dy * rhs.dy
  }
  
  static func dotProduct(v1: CGVector, v2: CGVector) -> CGFloat {
    v1.dx * v2.dx + v1.dy * v2.dy
  }
  
  static func / (_ vector: CGVector, _ scalar: CGFloat) -> CGVector {
    CGVector(dx: vector.dx / scalar, dy: vector.dy / scalar)
  }
  
  static func / (_ scalar: CGFloat, _ vector: CGVector) -> CGVector {
    CGVector(dx: vector.dx / scalar, dy: vector.dy / scalar)
  }
  
  // Returns again between vectors in range 0 to 2 * pi [positive]
  // a * b = ||a|| ||b|| cos(theta)
  // theta = arc cos (a * b / ||a|| ||b||)
  static func angleBetweenVectors(v1: CGVector, v2: CGVector) -> CGFloat {
    acos( v1 * v2 / (v1.length() * v2.length()) )
  }
  
  func length() -> CGFloat {
    sqrt(self.dx * self.dx + self.dy * self.dy)
  }
}

extension NumberFormatter {
  func string(_ value: Double, _ digits: Int, failText: String = "[?]") -> String {
    minimumFractionDigits = max(0, digits)
    maximumFractionDigits = minimumFractionDigits
    
    guard let s = string(from: NSNumber(value: value)) else {
      return failText
    }
    
    return s
  }
  
  func string(_ value: Float, _ digits: Int, failText: String = "[?]") -> String {
    minimumFractionDigits = max(0, digits)
    maximumFractionDigits = minimumFractionDigits
    
    guard let s = string(from: NSNumber(value: value)) else {
      return failText
    }
    
    return s
  }
  
  func string(_ value: CGFloat, _ digits: Int, failText: String = "[?]") -> String {
    minimumFractionDigits = max(0, digits)
    maximumFractionDigits = minimumFractionDigits
    
    guard let s = string(from: NSNumber(value: Double(value))) else {
      return failText
    }
    
    return s
  }
  
  func string(_ point: CGPoint, _ digits: Int = 1, failText: String = "[?]") -> String {
    let sx = string(point.x, digits, failText: failText)
    let sy = string(point.y, digits, failText: failText)
    return "(\(sx), \(sy))"
  }
  
  func string(_ vector: CGVector, _ digits: Int = 1, failText: String = "[?]") -> String {
    let sdx = string(vector.dx, digits, failText: failText)
    let sdy = string(vector.dy, digits, failText: failText)
    return "(\(sdx), \(sdy))"
  }
  
  func string(_ transform: CGAffineTransform, rotationDigits: Int = 2, translationDigits: Int = 1, failText: String = "[?]") -> String {
    let sa = string(transform.a, rotationDigits)
    let sb = string(transform.b, rotationDigits)
    let sc = string(transform.c, rotationDigits)
    let sd = string(transform.d, rotationDigits)
    let stx = string(transform.tx, translationDigits)
    let sty = string(transform.ty, translationDigits)
    
    var s = "a:  \(sa)   b: \(sb)   0"
    s += "\nc:  \(sc)   d: \(sd)   0"
    s += "\ntx: \(stx)   ty: \(sty)   1"
    return s
  }
}

struct Matrix1x3: CustomStringConvertible {
  var m1: CGFloat
  var m2: CGFloat
  var m3: CGFloat
  
  init(m1: CGFloat, m2: CGFloat, m3: CGFloat) {
    self.m1 = m1
    self.m2 = m2
    self.m3 = m3
  }
  
  init(_ p: CGPoint) {
    m1 = p.x
    m2 = p.y
    m3 = 1
  }
  
  var description: String {
    let f = NumberFormatter()
    return "\(f.string(m1, 2))"
    + "\n\(f.string(m2, 2))"
    + "\n\(f.string(m3, 2))"
  }
  
  /// |  m11  m12  m13 |      | m1 |
  /// |  m21  m22  m23 |  *   | m2 |
  /// |  m31  m32  m33 |      | m3 |
  func applying(_ x: Matrix3x3) -> Matrix1x3 {
    Matrix1x3(
      m1: x.m11 * m1 + x.m12 * m2 + x.m13 * m3,
      m2: x.m21 * m1 + x.m22 * m2 + x.m23 * m3,
      m3: x.m31 * m1 + x.m32 * m2 + x.m33 * m3)
  }
  
  func to2DPoint() -> CGPoint {
    CGPoint(x: m1 / m3, y: m2 / m3)
  }
}

/// Indices are m(row,column)
/// |  m11  m12  m13 |
/// |  m21  m22  m23 |
/// |  m31  m32  m33 |
struct Matrix3x3: CustomStringConvertible {
  var m11: CGFloat    //row 1
  var m12: CGFloat
  var m13: CGFloat
  
  var m21: CGFloat    //row 2
  var m22: CGFloat
  var m23: CGFloat
  
  var m31: CGFloat    //row 3
  var m32: CGFloat
  var m33: CGFloat
  
  var description: String {
    let f = NumberFormatter()
    
    var s = "\(f.string(m11, 2))   \(f.string(m12, 2))   \(f.string(m13, 2))"
    s += "\n\(f.string(m21, 2))   \(f.string(m22, 2))   \(f.string(m23, 2))"
    s += "\n\(f.string(m31, 2))   \(f.string(m32, 2))   \(f.string(m33, 2))"
    return s
  }
  
  func inverted() -> Matrix3x3? {
    let d = determinant()
    
    //TODO pick some realistic near-zero number here
    if abs(d) < 0.0000001 {
      return nil
    }
    
    //transpose matrix first
    let t = self.transpose()
    
    //determinants of 2x2 minor matrices
    let a11 = t.m22 * t.m33 - t.m32 * t.m23
    let a12 = t.m21 * t.m33 - t.m31 * t.m23
    let a13 = t.m21 * t.m32 - t.m31 * t.m22
    
    let a21 = t.m12 * t.m33 - t.m32 * t.m13
    let a22 = t.m11 * t.m33 - t.m31 * t.m13
    let a23 = t.m11 * t.m32 - t.m31 * t.m12
    
    let a31 = t.m12 * t.m23 - t.m22 * t.m13
    let a32 = t.m11 * t.m23 - t.m21 * t.m13
    let a33 = t.m11 * t.m22 - t.m21 * t.m12
    
    //adjugate (adjoint) matrix: apply + - + ... pattern
    let adj = Matrix3x3(
      m11: a11, m12: -a12, m13: a13,
      m21: -a21, m22: a22, m23: -a23,
      m31: a31, m32: -a32, m33: a33)
    return adj / d
  }
  
  func determinant() -> CGFloat {
    m11 * (m22 * m33 - m32 * m23) - m12 * (m21 * m33 - m31 * m23) + m13 * (m21 * m32 - m31 * m22)
  }
  
  func transpose() -> Matrix3x3 {
    Matrix3x3(m11: m11, m12: m21, m13: m31, m21: m12, m22: m22, m23: m32, m31: m13, m32: m23, m33: m33)
  }
  
  /// Converts the 3x3 matrix to a CGAffineTransform. Assumes that unused terms are zero.
  func toCGAffineTransform() -> CGAffineTransform {
    let CGM = self.transpose()
    return CGAffineTransform(a: CGM.m11, b: CGM.m12, c: CGM.m21, d: CGM.m22, tx: CGM.m31, ty: CGM.m32)
  }
  
  /// |  a11  a12  a13 |      |  b11  b12  b13 |
  /// |  a21  a22  a23 | *   |  b21  b22  b23 |
  /// |  a31  a32  a33 |      |  b31  b32  b33 |
  static func * (_ a: Matrix3x3, _ b: Matrix3x3) -> Matrix3x3 {
    return Matrix3x3(
      m11: a.m11 * b.m11 + a.m12 * b.m21 + a.m13 * b.m31,
      m12: a.m11 * b.m12 + a.m12 * b.m22 + a.m13 * b.m32,
      m13: a.m11 * b.m13 + a.m12 * b.m23 + a.m13 * b.m33,
      
      m21: a.m21 * b.m11 + a.m22 * b.m21 + a.m23 * b.m31,
      m22: a.m21 * b.m12 + a.m22 * b.m22 + a.m23 * b.m32,
      m23: a.m21 * b.m13 + a.m22 * b.m23 + a.m23 * b.m33,
      
      m31: a.m31 * b.m11 + a.m32 * b.m21 + a.m33 * b.m31,
      m32: a.m31 * b.m12 + a.m32 * b.m22 + a.m33 * b.m32,
      m33: a.m31 * b.m13 + a.m32 * b.m23 + a.m33 * b.m33)
  }
  
  static func / (_ m: Matrix3x3, _ s: CGFloat) -> Matrix3x3 {
    Matrix3x3(
      m11: m.m11/s, m12: m.m12/s, m13: m.m13/s,
      m21: m.m21/s, m22: m.m22/s, m23: m.m23/s,
      m31: m.m31/s, m32: m.m32/s, m33: m.m33/s)
  }
}

/// Three points nominally defining a triangle, but possibly colinear.
/// Used as an argument to the function SAM.transform(t1:t2:)
struct Triangle {
  var point1: CGPoint
  var point2: CGPoint
  var point3: CGPoint
  
  var x1: CGFloat { point1.x }
  var y1: CGFloat { point1.y }
  var x2: CGFloat { point2.x }
  var y2: CGFloat { point2.y }
  var x3: CGFloat { point3.x }
  var y3: CGFloat { point3.y }
  
  /// Point1 as a 2D vector
  var vector1: CGVector { CGVector(point1) }
  
  /// Point2 as a 2D vector
  var vector2: CGVector { CGVector(point2) }
  
  /// Point3 as a 2D vector
  var vector3: CGVector { CGVector(point3) }
  
  /// Return a Triangle after applying an affine transform to self.
  func applying(_ t: CGAffineTransform) -> Triangle {
    Triangle(
      point1: self.point1.applying(t),
      point2: self.point2.applying(t),
      point3: self.point3.applying(t)
    )
  }
  
  init(point1: CGPoint, point2: CGPoint, point3: CGPoint) {
    self.point1 = point1
    self.point2 = point2
    self.point3 = point3
  }
  
  init(x1: CGFloat, y1: CGFloat, x2: CGFloat, y2: CGFloat, x3: CGFloat, y3: CGFloat) {
    point1 = CGPoint(x: x1, y: y1)
    point2 = CGPoint(x: x2, y: y2)
    point3 = CGPoint(x: x3, y: y3)
  }
  
  /// Returns a (Bool, CGFloat) tuple indicating whether the points in the Triangle are colinear, and the angle between vectors tested.
  func colinear(degreesTolerance: CGFloat = 0.5) -> Bool {
    let v1 = vector2 - vector1
    let v2 = vector3 - vector2
    let radians = CGVector.angleBetweenVectors(v1: v1, v2: v2)
    
    if radians.isNaN {
      return true
    }
    
    var degrees = radiansToDegrees(radians)
    
    if degrees > 90 {
      degrees = 180 - degrees
    }
    
    return degrees < degreesTolerance
  }
  
  /// | p1.x    p2.x      p3.x |
  /// | p1.y    p2.y      p3.y |
  /// |   1       1            1    |
  func toMatrix() -> Matrix3x3 {
    Matrix3x3(
      m11: point1.x, m12: point2.x, m13: point3.x,
      m21: point1.y, m22: point2.y, m23: point3.y,
      m31: 1, m32: 1, m33: 1)
  }
}

func affineTransform(from: Triangle, to: Triangle) -> Matrix3x3? {
  // following example from https://stackoverflow.com/questions/18844000/transfer-coordinates-from-one-triangle-to-another-triangle
  // M * A = B
  // M = B * Inv(A)
  let A = from.toMatrix()
  
  guard let invA = A.inverted() else {
    return nil
  }
  
  let B = to.toMatrix()
  let M = B * invA
  
  return M
}

func cgAffineTransform(from: Triangle, to: Triangle) -> CGAffineTransform? {
  guard let M = affineTransform(from: from, to: to) else {
    return nil
  }
  return M.toCGAffineTransform()
}

// warpAffine
// https://stackoverflow.com/questions/49281334/replicate-cvwarpaffine-in-swift-with-core-image
