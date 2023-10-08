import SwiftUI
import Vision

class ObjectDetaction: ObservableObject {
  @Published var uiImage: UIImage?
  private var requests = [VNRequest]()
  var originalImage: UIImage? = nil
  var poseEstimation = PoseEstimation()
  
  init(){
    if let error = setupVision(){
      print(error.localizedDescription)
    }
  }
  
  @discardableResult
  func setupVision() -> NSError? {
    // Setup Vision parts
    let error: NSError! = nil
    guard let modelURL = Bundle.main.url(forResource: "yolov7-tiny_fp16", withExtension: "mlmodelc") else {
      return NSError(domain: "TopDownPoseEstimation", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
    }
    do {
      let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
      let objectRecognition = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
        Task {
          if let results = request.results {
            await self.drawVisionRequestResults(results)
          }
        }
      })
      objectRecognition.imageCropAndScaleOption = .scaleFit
      self.requests = [objectRecognition]
    } catch let error as NSError {
      print("Model loading went wrong: \(error)")
    }
    
    return error
  }
  
  func drawVisionRequestResults(_ results: [Any]) async {
    var bboxes: [Double] = []
    let label = "person"
    for observation in results where observation is VNRecognizedObjectObservation {
      guard let objectObservation = observation as? VNRecognizedObjectObservation else {
        continue
      }
      
      let width = self.originalImage!.size.width
      let height = self.originalImage!.size.height
      
      // Select only the label with the highest confidence.
      let topLabelObservation = objectObservation.labels[0]
      let bufferSize = CGSize(width: width, height: height)
      let objectBounds = VNImageRectForNormalizedRect(objectObservation.boundingBox, Int(bufferSize.width), Int(bufferSize.height))
      
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
    let uiImage = poseEstimation.run(sourceImage: self.originalImage!, boxes: bboxes)
    Task {
      await MainActor.run { [weak self] in
        self!.uiImage = uiImage
      }
    }
  }
  
  func prediction(imageBuffer: UIImage) async {
    self.originalImage = imageBuffer
    let exifOrientation: CGImagePropertyOrientation = .up
    let imageRequestHandler = VNImageRequestHandler(cgImage: imageBuffer.cgImage!, orientation: exifOrientation, options: [:])
    do {
      try imageRequestHandler.perform(self.requests)
    } catch {
      print(error)
    }
  }
}
