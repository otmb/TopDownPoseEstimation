import SwiftUI
import Vision
import Accelerate

class KeyPointProcess {
  var modelWidth: Int
  var modelHeight: Int
  var heatmapHeight: Int
  var heatmapWidth: Int
  var keypointsNumber: Int
  let pixelStd = 200.0
  
  init(modelWidth: Int, modelHeight: Int, keypointsNumber: Int = 17){
    self.modelWidth = modelWidth
    self.modelHeight = modelHeight
    self.heatmapWidth = Int(modelWidth / 4)
    self.heatmapHeight = Int(modelHeight / 4)
    self.keypointsNumber = keypointsNumber
  }
  
  func preExecute(image: UIImage, box: CGRect) -> UIImage? {
    let (center, scale) = box2cs(box: box)
    let _scale = CGPoint(x: scale.x * pixelStd, y: scale.y * pixelStd)
    let outputSize = CGSize(width: modelWidth, height: modelHeight)
    if let trans = getAffineTransform(center: center, scale: _scale,
                                      rot: 0, outputSize: outputSize, inv: 0) {
      return image.transformed(by: trans, size: outputSize)
    }
    return nil
  }
  
  func postExecute(heatmap: [Double], box: CGRect) -> HumanPose? {
    let (center, scale) = box2cs(box: box)
    let heatmapSize = CGSize(width: heatmapWidth, height: heatmapHeight)
    
    let maxCoords = getMaxCoords(heatmap: heatmap)
    let maxvals = maxCoords.map { $0.maxval }
    let coords = maxCoords.enumerated().map { (j, maxCoord) in
      var coord = maxCoord.coord
      let index = j * heatmapHeight * heatmapWidth
      let px = Int(coord.x + 0.5)
      let py = Int(coord.y + 0.5)
      
      if (px > 0 && px < heatmapWidth - 1) {
        coord.x += sign(heatmap[index + py * heatmapWidth + px + 1] -
                        heatmap[index + py * heatmapWidth + px - 1]) * 0.25
      }
      if (py > 0 && py < heatmapHeight - 1) {
        coord.y += sign(heatmap[index + (py + 1) * heatmapWidth + px] -
                        heatmap[index + (py - 1) * heatmapWidth + px]) * 0.25
      }
      return coord
    }
    
    let _scale = CGPoint(x: scale.x * pixelStd, y: scale.y * pixelStd)
    guard let preds = transformPreds(coords: coords,
                                     center: center, scale: _scale,
                                     outputSize: heatmapSize) else { return nil }
    
    return HumanPose(keypoints: preds, scores: maxvals, box: box)
  }
  
  func get3rdPoint(_ a: simd_double3,_ b: simd_double3) -> simd_double3 {
    let direct = simd_double2(a.x - b.x, a.y - b.y)
    return simd_double3(a.x - direct.y, a.y + direct.x, 1)
  }
  
  func getAffineTransform(center: CGPoint, scale: CGPoint, rot: Double,
                          outputSize: CGSize, inv: Int) -> CGAffineTransform? {
    
    let src_w = scale.x
    let dst_w = Double(outputSize.width)
    let dst_h = Double(outputSize.height)
    let rot_rad = rot * .pi / 180
    
    let src_dir = [-0.5 * src_w, 0.0, rot_rad]
    let dst_dir = [-0.5 * dst_w, 0.0]
    
    let src1 = simd_double3(center.x, center.y, 1)
    let src2 = simd_double3(center.x + src_dir[0], center.y + src_dir[1], 1)
    let src = simd_double3x3(src1, src2, get3rdPoint(src1, src2))
    
    let dst1 = simd_double3(dst_w * 0.5, dst_h * 0.5, 1)
    let dst2 = simd_double3(dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1], 1)
    let dst = simd_double3x3(dst1, dst2, get3rdPoint(dst1, dst2))
    
    if (inv == 0) {
      return cgAffineTransform(from: src, to: dst)
    } else {
      return cgAffineTransform(from: dst, to: src)
    }
  }
  
  func box2cs(box: CGRect) -> (center: CGPoint, scale: CGPoint) {
    
    let x = box.minX
    let y = box.minY
    var w = box.width
    var h = box.height
    
    let center = CGPoint(x: x + w * 0.5, y: y + h * 0.5)
    let aspectRatio: Double = Double(modelWidth) / Double(modelHeight)
    
    if (w > aspectRatio * h) {
      h = w * 1.0 / aspectRatio
    } else if (w < aspectRatio * h) {
      w = h * aspectRatio
    }
    
    var scale = CGPoint(x: w * 1.0 / pixelStd, y: h * 1.0 / pixelStd)
    if (center.x != -1) {
      scale.x *= 1.25
      scale.y *= 1.25
    }
    return (center, scale)
  }
  
  struct MaxCoord {
    var coord = CGPoint()
    var maxval = 0.0
  }
  
  func getMaxCoords(heatmap: [Double]) -> [MaxCoord] {
    let width = Double(heatmapWidth)
    
    return (0..<keypointsNumber).map { j in
      let idx = j * heatmapHeight * heatmapWidth
      let end = idx + heatmapHeight * heatmapWidth
      let (maxIdx, maxValue) = vDSP.indexOfMaximum(heatmap[idx..<end])
      let coord = CGPoint(
        x: Double(maxIdx).truncatingRemainder(dividingBy: width),
        y: Double(maxIdx) / width)
      return MaxCoord(coord: coord, maxval: maxValue)
    }
  }
  
  func transformPreds(coords: [CGPoint], center: CGPoint,
                      scale: CGPoint, outputSize: CGSize) -> [CGPoint]? {
    
    guard let t = getAffineTransform(center: center, scale: scale,
                                     rot: 0, outputSize: outputSize, inv: 1) else {
      return nil
    }
    
    let trans = double2x3(
      simd_double3(t.a, t.b, t.tx),
      simd_double3(t.c, t.d, t.ty)
    )
    return coords.map {
      affineTransform(point: $0, trans: trans)
    }
  }
  
  func affineTransform(point: CGPoint, trans: double2x3) -> CGPoint {
    let w = simd_double3(point.x, point.y, 1) * trans
    return CGPoint(x: w.x, y: w.y)
  }
  
  // https://rethunk.medium.com/perspective-transform-from-quadrilateral-to-quadrilateral-in-swift-using-simd-for-matrix-operations-15dc3f090860
  // Gary Bartos
  func affineTransform(from: double3x3, to: double3x3) -> double3x3? {
    let invA = from.inverse
    if invA.determinant.isNaN {
      return nil
    }
    return to * invA
  }
  
  func cgAffineTransform(from: double3x3, to: double3x3) -> CGAffineTransform? {
    guard let M = affineTransform(from: from, to: to) else {
      return nil
    }
    return M.toCGAffineTransform
  }
}

// warpAffine
// https://stackoverflow.com/questions/49281334/replicate-cvwarpaffine-in-swift-with-core-image

extension UIImage {
  func transformed(by transform: CGAffineTransform, size: CGSize) -> UIImage? {
    UIGraphicsBeginImageContext(size)
    let context = UIGraphicsGetCurrentContext()
    let orig = CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height)
    context?.concatenate(transform)
    self.draw(in: orig)
    let result = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return result
  }
}

extension double3x3 {
  var toCGAffineTransform: CGAffineTransform {
    let (m1, m2, m3) = self.columns
    return CGAffineTransform(a: m1.x, b: m1.y, c: m2.x, d: m2.y, tx: m3.x, ty: m3.y)
  }
}
