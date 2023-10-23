// Perspective transform from quadrilateral to quadrilateral in Swift using SIMD for matrix operations
// Gary Bartos Jan 6, 2021
// https://rethunk.medium.com/perspective-transform-from-quadrilateral-to-quadrilateral-in-swift-using-simd-for-matrix-operations-15dc3f090860

import CoreGraphics
import Foundation
import simd

/// Finds the affine transform (translation, rotation, scale, ...) from one triangle to another.
/// Used in perspectiveTransform( ).
func affineTransform(from: Triangle, to: Triangle) -> float3x3? {
  // following example from https://stackoverflow.com/questions/18844000/transfer-coordinates-from-one-triangle-to-another-triangle
  // M * A = B
  // M = B * Inv(A)
  let A = from.toMatrix()
  let invA = A.inverse
  
  if invA.determinant.isNaN {
    return nil
  }
  
  let B = to.toMatrix()
  let M = B * invA
  
  return M
}

// Conversions to/from CGPoint for use with CGImage and SIMD matrix operations.
extension CGPoint {
  /// A 1x2 vector of the point: (x, y)
  var vector2: simd_float2 {
    simd_float2(Float(self.x), Float(self.y))
  }
  
  /// A 1x3 vector of the point (x, y, 1)
  var vector3: simd_float3 {
    simd_float3(Float(self.x), Float(self.y), Float(1))
  }
  
  /// Returns a point (v.x, v.y)
  static func fromVector2(_ v: simd_float2) -> CGPoint {
    CGPoint(x: CGFloat(v.x), y: CGFloat(v.y))
  }
  
  /// Returns a point (x, y) = (v.x / v.z, v.y / v.z)
  /// Returns {x +∞, y +∞} if v.z == 0
  static func fromVector3(_ v: simd_float3) -> CGPoint {
    CGPoint(x: CGFloat(v.x / v.z), y: CGFloat(v.y / v.z))
  }
}

// Conversions between 2D points and 1x3 homogeneous coordinates.
extension simd_float2 {
  /// Returns (inf, inf) if v.z == 0
  static func fromVector3(_ v: simd_float3) -> simd_float2 {
    simd_float2(v.x / v.z, v.y / v.z)
  }
  
  /// Returns (x, y, 1)
  func toVector3() -> simd_float3 {
    simd_float3(self.x, self.y, 1)
  }
}

// Conversions between 1x3 homogeneous coordinates and 2D points.
extension simd_float3 {
  /// Returns (x,y,1)
  static func fromVector2(_ v: simd_float2) -> simd_float3 {
    simd_float3(v.x, v.y, 1)
  }
  
  /// Returns (inf,inf) if v.z == 0
  func toVector2() -> simd_float2 {
    simd_float2(self.x / self.z, self.y / self.z)
  }
}

/// Three points nominally defining a triangle, but possibly colinear.
/// Used as an argument to the function SAM.transform(t1:t2:)
struct Triangle {
  var point1: simd_float2
  var point2: simd_float2
  var point3: simd_float2
  
  init(_ point1: simd_float2, _ point2: simd_float2, _ point3: simd_float2) {
    self.point1 = point1
    self.point2 = point2
    self.point3 = point3
  }
  
  init(_ vector1: simd_float3, _ vector2: simd_float3, _ vector3: simd_float3) {
    point1 = vector1.toVector2()
    point2 = vector2.toVector2()
    point3 = vector3.toVector2()
  }
  
  /// Three points are colinear if their determinant is zero. We assume close to colinear might as well be colinear.
  ///    | x1  x2  x3 |
  /// det | y1  y2  y3 |  = 0     -->    abs( det(M) )  < tolerance ?
  ///    |1 1  1 |
  func colinear(tolerance: Float = 0.01) -> Bool {
    let m = toMatrix()
    return abs(m.determinant) < tolerance
  }
  
  /// | p1.x    p2.x      p3.x |
  /// | p1.y    p2.y      p3.y |
  /// |   1       1            1    |
  func toMatrix() -> float3x3 {
    float3x3(point1.toVector3(), point2.toVector3(), point3.toVector3())
  }
}

func cgAffineTransform(from: Triangle, to: Triangle) -> CGAffineTransform? {
  guard let M = affineTransform(from: from, to: to) else {
    return nil
  }
  return CGAffineTransform(a: CGFloat(M[0][0]), b: CGFloat(M[0][1]),
                           c: CGFloat(M[1][0]), d: CGFloat(M[1][1]),
                           tx: CGFloat(M[2][0]), ty: CGFloat(M[2][1]))
}
