import SwiftUI

struct ContentView: View {
  @ObservedObject var detection = ObjectDetaction()
  var uiImage = UIImage(named: "test.jpg")!
  @State var image: UIImage?
  var body: some View {
    VStack {
      if let image = detection.uiImage {
        Image(uiImage: image).resizable()
          .aspectRatio(contentMode: .fit)
      }
    }
    .onAppear {
      detection.prediction(imageBuffer: uiImage)
    }
    .padding()
  }
}
