import SwiftUI
import Vision
import Accelerate

class KeyPointProcess {
  var modelWidth: Int
  var modelHeight: Int
  var keypointsNumber: Int
  let pixelStd = 200.0
  
  init(modelWidth: Int, modelHeight: Int, keypointsNumber: Int = 17){
    self.modelWidth = modelWidth
    self.modelHeight = modelHeight
    self.keypointsNumber = keypointsNumber
  }
  
  func preExecute(image: UIImage, box: CGRect) -> UIImage? {
    let (center, scale) = box2cs(box: box)
    let _scale = CGPoint(x: scale.x * pixelStd, y: scale.y * pixelStd)
    let outputSize = CGSize(width: modelWidth, height: modelHeight)
    let trans = getAffineTransform(center: center, scale: _scale,
                                   rot: 0, outputSize: outputSize, inv: 0)
    if let trans = trans {
      return image.transformed(by: trans, size: outputSize)
    }
    return nil
  }
  
  func postExecute(heatmap: [Double], box: CGRect) -> HumanPose {
    let (center, scale) = box2cs(box: box)
    let heatmapHeight = modelHeight / 4
    let heatmapWidth = modelWidth / 4
    var dim: [Int] = [ 1, keypointsNumber, heatmapHeight, heatmapWidth ]
    let imgSize = CGSize(width: heatmapWidth, height: heatmapHeight)
    var coords: [Double] = Array(repeating: 0.0, count: keypointsNumber * 2)
    var maxvals: [Double] = Array(repeating: 0.0, count: keypointsNumber)
    var preds: [Double] = Array(repeating: 0.0, count: keypointsNumber * 3)
    
    getMaxCoords(heatmap: heatmap, dim: &dim, coords: &coords,
                 maxvals: &maxvals)
    
    for j in 0..<dim[1] {
      let index = j * dim[2] * dim[3]
      let px = Int(coords[j * 2] + 0.5)
      let py = Int(coords[j * 2 + 1] + 0.5)
      
      if (px > 0 && px < heatmapWidth - 1) {
        let diff_x = heatmap[index + py * dim[3] + px + 1] -
        heatmap[index + py * dim[3] + px - 1]
        coords[j * 2] += sign(diff_x) * 0.25
      }
      if (py > 0 && py < heatmapHeight - 1) {
        let diff_y = heatmap[index + (py + 1) * dim[3] + px] -
        heatmap[index + (py - 1) * dim[3] + px]
        coords[j * 2+1] += sign(diff_y) * 0.25
      }
    }
    
    let _scale = CGPoint(x: scale.x * pixelStd, y: scale.y * pixelStd)
    
    transformPreds(coords: coords, center: center, scale: _scale,
                   outputSize: imgSize, dim: dim, targetCoords: &preds)
    
    var pose = HumanPose(keypointsNumber: keypointsNumber)
    for j in 0..<keypointsNumber {
      pose.keypoints[j] = CGPoint(x: preds[j * 3 + 1], y: preds[j * 3 + 2])
      pose.scores[j] = maxvals[j]
    }
    pose.score = vDSP.mean(pose.scores)
    return pose
  }
  
  func get3rdPoint(_ a: CGPoint,_ b: CGPoint) -> CGPoint {
    let direct = CGPoint(x: a.x - b.x, y: a.y - b.y)
    return CGPoint(x: a.x - direct.y, y: a.y + direct.x)
  }
  
  func getAffineTransform(center: CGPoint, scale: CGPoint, rot: Double, outputSize: CGSize, inv: Int) -> CGAffineTransform? {
    
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
  
  func getMaxCoords(heatmap: [Double], dim: inout [Int], coords: inout [Double],
                    maxvals: inout [Double]) {
    let width = Double(dim[3])
    
    for j in 0..<dim[1] {
      let idx = j * dim[2] * dim[3]
      let end = idx + dim[2] * dim[3]
      var slice = heatmap[idx..<end]
      let pointer = slice.withUnsafeMutableBufferPointer{ $0 }
      if let maxValue = pointer.max() {
        maxvals[j] = maxValue
        if let maxId = pointer.firstIndex(of: maxValue) {
          coords[j * 2] = Double(maxId).truncatingRemainder(dividingBy: width)
          coords[j * 2 + 1] = Double(maxId) / width
        }
      }
    }
  }
  
  func transformPreds(coords: [Double], center: CGPoint,
                      scale: CGPoint, outputSize: CGSize,
                      dim: [Int], targetCoords: inout [Double]){
    
    let trans = getAffineTransform(center: center, scale: scale,
                                   rot: 0, outputSize: outputSize, inv: 1)
    if let t = trans {
      let trans = double2x3(
        simd_double3(t.a, t.b, t.tx),
        simd_double3(t.c, t.d, t.ty)
      )
      for p in 0..<dim[1] {
        affineTransform(point: CGPoint(x: coords[p * 2], y: coords[p * 2 + 1]),
                        trans: trans, preds: &targetCoords, p: p)
      }
    }
  }
  
  func affineTransform(point: CGPoint, trans: double2x3, preds: inout [Double], p: Int){
    let pt = simd_double3(point.x, point.y, 1.0)
    let w:simd_double2 = simd_mul(pt, trans)
    preds[p * 3 + 1] = w.x
    preds[p * 3 + 2] = w.y
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
