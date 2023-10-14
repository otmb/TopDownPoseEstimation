import SwiftUI
import Vision

class ObjectDetaction: ObservableObject {
  let modelName = "yolov7-tiny_fp16"
  private var requests = [VNRequest]()
  var originalImage: UIImage? = nil
  var bboxes = [Double]()
  
  init(){
    if let error = setupVision(){
      print(error.localizedDescription)
    }
  }
  
  @discardableResult
  func setupVision() -> NSError? {
    // Setup Vision parts
    let error: NSError! = nil
    guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
      return NSError(domain: "TopDownPoseEstimation", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
    }
    do {
      let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
      let request = VNCoreMLRequest(model: visionModel, completionHandler: visionObjectDetectionResults)
      request.imageCropAndScaleOption = .scaleFit
      requests = [request]
    } catch let error as NSError {
      print("Model loading went wrong: \(error)")
    }
    
    return error
  }
  
  func runCoreML(uiImage: UIImage, orientation: CGImagePropertyOrientation) throws {
    let cgiImage = uiImage.cgImage!
    let classifierRequestHandler = VNImageRequestHandler(cgImage: cgiImage, 
                                                         orientation: orientation, options: [:])
    try classifierRequestHandler.perform(requests)
  }
  
  func visionObjectDetectionResults(request: VNRequest, error: Error?) {
    bboxes = []
    guard let observations = request.results as? [VNRecognizedObjectObservation] else { fatalError() }
    let label = "person"
    guard let uiImage = originalImage else { return }
    
    for observation in observations {
      let width = uiImage.size.width
      let height = uiImage.size.height
      
      let topLabelObservation = observation.labels[0]
      let bufferSize = CGSize(width: width, height: height)
      let objectBounds = VNImageRectForNormalizedRect(observation.boundingBox, Int(bufferSize.width), Int(bufferSize.height))
      
      if topLabelObservation.identifier == label {
        let minY:CGFloat = height - objectBounds.minY // 画像の下側の値を返すので反転
        let bbox:[Double] = [
          objectBounds.minX,
          minY - objectBounds.height,
          objectBounds.width,
          objectBounds.height
        ]
        bboxes.append(contentsOf: bbox)
      }
    }
  }
  
  func prediction(uiImage: UIImage, orientation: CGImagePropertyOrientation = .up) throws -> [Double] {
    self.originalImage = uiImage
    try runCoreML(uiImage: uiImage, orientation: orientation)
    return bboxes
  }
}
