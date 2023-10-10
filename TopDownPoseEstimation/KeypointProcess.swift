import SwiftUI
import Vision

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
        let diff_x = heatmap[index + py * heatmapWidth + px + 1] -
        heatmap[index + py * heatmapWidth + px - 1]
        coord.x += sign(diff_x) * 0.25
      }
      if (py > 0 && py < heatmapHeight - 1) {
        let diff_y = heatmap[index + (py + 1) * heatmapWidth + px] -
        heatmap[index + (py - 1) * heatmapWidth + px]
        coord.y += sign(diff_y) * 0.25
      }
      return coord
    }
    
    let _scale = CGPoint(x: scale.x * pixelStd, y: scale.y * pixelStd)
    guard let preds = transformPreds(coords: coords,
                                     center: center, scale: _scale,
                                     outputSize: heatmapSize) else { return nil }
    
    return HumanPose(keypoints: preds, scores: maxvals, box: box)
  }
  
  func get3rdPoint(_ a: CGPoint,_ b: CGPoint) -> CGPoint {
    let direct = CGPoint(x: a.x - b.x, y: a.y - b.y)
    return CGPoint(x: a.x - direct.y, y: a.y + direct.x)
  }
  
  func getAffineTransform(center: CGPoint, scale: CGPoint, rot: Double,
                          outputSize: CGSize, inv: Int) -> CGAffineTransform? {
    
    let src_w = scale.x
    let dst_w = Double(outputSize.width)
    let dst_h = Double(outputSize.height)
    let rot_rad = rot * .pi / 180
    
    let src_dir = [-0.5 * src_w, 0.0, rot_rad]
    let dst_dir = [-0.5 * dst_w, 0.0]
    
    var srcPoint = Array(repeating: CGPoint(), count : 3)
    srcPoint[0] = center
    srcPoint[1] = CGPoint(x: center.x + src_dir[0], y: center.y + src_dir[1])
    srcPoint[2] = get3rdPoint(srcPoint[0], srcPoint[1])
    let src = Triangle(point1: srcPoint[0], point2: srcPoint[1], point3: srcPoint[2])
    
    var dstPoint = Array(repeating: CGPoint(), count : 3)
    dstPoint[0] = CGPoint(x: dst_w * 0.5, y: dst_h * 0.5)
    dstPoint[1] = CGPoint(x: dst_w * 0.5 + dst_dir[0], y: dst_h * 0.5 + dst_dir[1])
    dstPoint[2] = get3rdPoint(dstPoint[0], dstPoint[1])
    let dst = Triangle(point1: dstPoint[0], point2: dstPoint[1], point3: dstPoint[2])
    
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
      var slice = heatmap[idx..<end]
      let pointer = slice.withUnsafeMutableBufferPointer{ $0 }
      if let maxValue = pointer.max() {
        if let maxId = pointer.firstIndex(of: maxValue) {
          let coord = CGPoint(
                        x: Double(maxId).truncatingRemainder(dividingBy: width),
                        y: Double(maxId) / width)
          return MaxCoord(coord: coord, maxval: maxValue)
        }
      }
      return MaxCoord()
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
    let pt = simd_double3(point.x, point.y, 1.0)
    let w:simd_double2 = simd_mul(pt, trans)
    return CGPoint(x: w.x, y: w.y)
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
