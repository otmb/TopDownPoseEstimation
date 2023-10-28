import SwiftUI

struct ValTestView: View {
  @ObservedObject var detection = ObjectDetaction()
  @ObservedObject var poseEstimation = PoseEstimation()
  @State var image: UIImage?
  @State var cocoKeyPointList = [CocoKeyPoint]()
  @State private var progress = 0.0
  @State private var progressText = ""
  
  var body: some View {
    VStack {
      if let image = image {
        Image(uiImage: image).resizable()
          .aspectRatio(contentMode: .fit)
      }
      ProgressView(progressText,value: progress)
    }
    .task {
      await predict()
      do {
        try saveJson()
      } catch {
        print(error.localizedDescription)
      }
    }
    .padding()
  }
  
  struct CocoKeyPoint: Encodable {
    var image_id: Int
    var category_id: Int
    var keypoints: [Double]
    var score: Double
  }
  
  func saveJson() throws {
    let fileName = poseEstimation.modelName + "_results.json"
    
    let json = try JSONEncoder().encode(cocoKeyPointList)
    guard let url = try? FileManager.default.url(for: .documentDirectory, 
                                                 in: .userDomainMask,
                                                 appropriateFor: nil, 
                                                 create: true)
      .appendingPathComponent(fileName) else { return }
    
    try json.write(to: url)
  }
  
  func predict() async {
    guard let files = getFileList(Bundle.main.bundlePath) else { return }
    
    let regex = /^(\d{12})\.jpg$/
    let fileList = files.filter { $0.contains(regex) }
    let fileCount = fileList.count
    for (idx, file) in fileList.enumerated() {
      await MainActor.run {
        progress = Double(idx) / Double(fileCount)
        progressText = "\(idx) / \(fileCount)"
      }
      guard let match = file.firstMatch(of: regex) else { continue }
      guard let imageId = Int(match.1) else { continue }
      guard let uiImage = UIImage(named: file) else { continue }
      guard let resultImage = try? _predict(imageId: imageId, uiImage: uiImage) else { continue }
      await MainActor.run {
        image = resultImage
      }
    }
  }
  
  func _predict(imageId: Int, uiImage: UIImage) throws -> UIImage? {
    let boxes = try detection.prediction(uiImage: uiImage)
    if boxes.count > 0 {
      do {
        let poses = try poseEstimation.prediction(uiImage: uiImage, boxes: boxes)
        addCocoData(imageId: imageId, poses: poses)
        let render = PoseRender(uiImage, poses: poses)
        return render.render()
      } catch {
        print(imageId, error.localizedDescription)
      }
    } else {
      print(imageId, "no boxes")
    }
    return nil
  }
  
  func addCocoData(imageId: Int, poses: [HumanPose]){
    for pose in poses {
      let keypoints = Array(zip(pose.keypoints, pose.scores).map { [$0.x , $0.y, $1] }.joined())
      let cocoKeyPoint = CocoKeyPoint(image_id: imageId, category_id: 1, keypoints: keypoints, score: pose.score)
      
      cocoKeyPointList.append(cocoKeyPoint)
    }
  }
  
  func getFileList(_ dirName: String) -> [String]? {
    let fileManager = FileManager.default
    let files: [String]
    do {
      files = try fileManager.contentsOfDirectory(atPath: dirName)
    } catch {
      return nil
    }
    return files
  }
}

