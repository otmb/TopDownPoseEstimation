import SwiftUI
import Vision

class PoseEstimation: ObservableObject {
  let modelName = "vitpose-b256x192_fp16"
  @Published var uiImage: UIImage?
  var poses = [HumanPose]()
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
      guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
        return NSError(domain: "TopDownPoseEstimation", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
      }
      let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
      requests = [VNCoreMLRequest(model: visionModel, completionHandler: visionPoseEstimationResults)]
    } catch {
      return NSError(domain: "TopDownPoseEstimation", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
    }
    return error
  }
  
  func prediction(uiImage: UIImage, boxes: [Double]) throws -> UIImage? {
    poses = [HumanPose]()
    let personNum = boxes.count / 4
    for num in 0..<personNum {
      box = CGRect(x: boxes[num*4], y: boxes[num*4+1], width: boxes[num*4+2], height: boxes[num*4+3])
      if let uiImage = keypointProcess.preExecute(image: uiImage, box: box){
        try runCoreML(uiImage: uiImage)
      }
    }
    let render = PoseRender(uiImage, poses: poses)
    return render.render()
  }
  
  func runCoreML(uiImage: UIImage) throws {
    let cgiImage = uiImage.cgImage!
    let classifierRequestHandler = VNImageRequestHandler(cgImage: cgiImage, options: [:])
    try classifierRequestHandler.perform(requests)
  }
  
  func visionPoseEstimationResults(request: VNRequest, error: Error?){
    guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else { fatalError() }
    let mlarray = observations[0].featureValue.multiArrayValue!
    
    let length = mlarray.count
    let floatPtr =  mlarray.dataPointer.bindMemory(to: Float32.self, capacity: length)
    let floatBuffer = UnsafeBufferPointer(start: floatPtr, count: length)
    
    if let pose = keypointProcess.postExecute(heatmap: floatBuffer.map{ Double($0) }, box: box ){
      poses.append(pose)
    }
  }
}
