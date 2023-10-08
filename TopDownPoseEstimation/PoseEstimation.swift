import SwiftUI
import Vision
import PoseRender
import Accelerate

class PoseEstimation: ObservableObject {
  
  @Published var uiImage: UIImage?
  var keypoints = [Float32]()
  let modelWidth = 192
  let modelHeight = 256
  let keypointsNumber = 17
  var box = CGRect()
  private var requests = [VNRequest]()
  let keypointProcess: KeyPointProcess
  
  init(){
    keypointProcess = KeyPointProcess(
      modelWidth: modelWidth, modelHeight: modelHeight, keypointsNumber: keypointsNumber)
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
  
  func run(sourceImage: UIImage, boxes: [Double]) -> UIImage? {
    keypoints = [Float32]()
    let peopleNum = boxes.count / 4
    for num in 0..<peopleNum {
      box = CGRect(x: boxes[num*4], y: boxes[num*4+1], width: boxes[num*4+2], height: boxes[num*4+3])
      let uiImage = keypointProcess.preExecute(image: sourceImage, box: box)
      if let uiImage = uiImage {
        runCoreML(uiImage: uiImage)
      }
    }
    let render = PoseRender()
    var boxes = boxes.map { Float32($0) }
    return render.renderHumanPose(sourceImage, keypoints: &keypoints, peopleNum: Int32(peopleNum), boxes: &boxes)
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
    
    let results = keypointProcess.postExecute(heatmap: floatBuffer.map{ Double($0) }, box: box )
    let preds: [Float32] = results.map { Float32($0) }
    
    keypoints.append(contentsOf: preds)
  }
}
