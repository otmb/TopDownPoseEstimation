import SwiftUI

struct ContentView: View {
  @ObservedObject var detection = ObjectDetaction()
  @ObservedObject var poseEstimation = PoseEstimation()
  var uiImage = UIImage(named: "test.jpg")
  @State var image: UIImage?
  var body: some View {
    VStack {
      if let image = image {
        Image(uiImage: image).resizable()
          .aspectRatio(contentMode: .fit)
      }
    }
    .onAppear {
      predict()
    }
    .padding()
  }
  
  func predict(){
    do {
      if let uiImage = uiImage {
        let boxes = try detection.prediction(uiImage: uiImage)
        if boxes.count > 0 {
          image = try poseEstimation.prediction(uiImage: uiImage, boxes: boxes)
        }
      }
    } catch {
      print(error)
    }
  }
}
