import SwiftUI
import Vision
import PoseRender
import Accelerate

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

class PoseEstimation: ObservableObject {
  
  @Published var uiImage: UIImage?
  var keypoints = [Float32]()
  var center = CGPoint()
  var scale = CGPoint()
  let pixelStd = 200.0
  let modelWidth = 192
  let modelHeight = 256
  let keypointsNumber = 17
  private var requests = [VNRequest]()
  
  init(){
    if let error = setupVision(){
      print(error.localizedDescription)
    }
  }
  
  @discardableResult
  func setupVision() -> NSError? {
    let error: NSError! = nil
    do {
      guard let modelURL = Bundle.main.url(forResource: "vitpose-b256x192_fp16", withExtension: "mlmodelc") else {
        return NSError(domain: "TopDownPoseEstimation", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
      }
      let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
      requests = [VNCoreMLRequest(model: visionModel, completionHandler: handleClassification)]
    } catch {
      return NSError(domain: "TopDownPoseEstimation", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
    }
    return error
  }
  
  func run(sourceImage: UIImage, boxes: inout [Float32]) -> UIImage? {
    keypoints = [Float32]()
    let peopleNum = boxes.count / 4
    for num in 0..<peopleNum {
      let box = boxes[num*4..<num*4+4].map{ Double($0) }
      (center, scale) = box2cs(box: CGRect(x: box[0], y: box[1], width: box[2], height: box[3]))
      let uiImage = preExecute(image: sourceImage, box: box)
      if let uiImage = uiImage {
        runCoreML(uiImage: uiImage)
      }
    }
    let render = PoseRender()
    return render.renderHumanPose(sourceImage, keypoints: &keypoints, peopleNum: Int32(peopleNum), boxes: &boxes)
  }
  
  func preExecute(image: UIImage, box: [Double]) -> UIImage? {
    let _scale = CGPoint(x: scale.x * 200, y: scale.y * 200)
    let outputSize = CGSize(width: modelWidth, height: modelHeight)
    let trans = getAffineTransform(center: center, scale: _scale,
                                   rot: 0, outputSize: outputSize, inv: 0)
    if let trans = trans {
      return image.transformed(by: trans, size: outputSize)
    }
    return nil
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
  
  func box2cs(box: CGRect) -> (CGPoint, CGPoint) {
    
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
  
  func runCoreML(uiImage: UIImage) {
    let cgiImage = uiImage.cgImage!
    let classifierRequestHandler = VNImageRequestHandler(cgImage: cgiImage, options: [:])
    do {
      try classifierRequestHandler.perform(requests)
    } catch {
      print(error.localizedDescription)
    }
  }
  
  func handleClassification(request: VNRequest, error: Error?){
    guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else { fatalError() }
    let mlarray = observations[0].featureValue.multiArrayValue!
    
    let length = mlarray.count
    let floatPtr =  mlarray.dataPointer.bindMemory(to: Float32.self, capacity: length)
    let floatBuffer = UnsafeBufferPointer(start: floatPtr, count: length)
    
    let results = postExecute(heatmap: Array(floatBuffer).map { Double($0) } )
    let preds: [Float32] = results.map { Float32($0) }
    
    keypoints.append(contentsOf: preds)
  }
  
  func postExecute(heatmap: [Double]) -> [Double]{
    let heatmapHeight = modelHeight / 4
    let heatmapWidth = modelWidth / 4
    var dim: [Int] = [ 1, keypointsNumber, heatmapHeight, heatmapWidth ]
    let imgSize = CGSize(width: heatmapWidth, height: heatmapHeight)
    var coords: [Double] = Array(repeating: 0.0, count: keypointsNumber * 2)
    var maxvals: [Double] = Array(repeating: 0.0, count: keypointsNumber)
    var preds: [Double] = Array(repeating: 0.0, count: keypointsNumber * 3)
    
    getMaxCoords(heatmap: heatmap, dim: &dim, coords: &coords,
                 maxvals: &maxvals, batchid: 0)
    
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
    
    let _scale = CGPoint(x: scale.x * 200, y: scale.y * 200)
    
    transformPreds(coords: coords, center: center, scale: _scale,
                   outputSize: imgSize, dim: dim, targetCoords: &preds)
    
    var results = Array(repeating: 0.0, count: keypointsNumber * 3)
    
    for j in 0..<dim[1] {
      results[j * 3] = preds[j * 3 + 1]
      results[j * 3 + 1] = preds[j * 3 + 2]
      results[j * 3 + 2] = maxvals[j]
    }
    return results
  }
  
  func getMaxCoords(heatmap: [Double], dim: inout [Int], coords: inout [Double],
                    maxvals: inout [Double], batchid: Int) {
    let numJoints = dim[1]
    let width = Double(dim[3])
    
    for j in 0..<dim[1] {
      let idx = batchid * numJoints * dim[2] * dim[3] + j * dim[2] * dim[3]
      let end = idx + dim[2] * dim[3]
      var heat = heatmap[idx..<end]
      let pointer = heat.withUnsafeMutableBufferPointer { $0.baseAddress! }
      if let _maxDis = UnsafeMutableBufferPointer(start: pointer, count: heat.count).max() {
        let maxDis = heat.firstIndex(of: _maxDis) ?? 0
        let maxId = Double(idx.distance(to: maxDis))
        
        maxvals[j] = _maxDis
        if (maxDis > 0) {
          coords[j * 2] = maxId.truncatingRemainder(dividingBy: width)
          coords[j * 2 + 1] = maxId / width
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
