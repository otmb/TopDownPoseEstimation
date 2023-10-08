import CoreGraphics
import UIKit
import Accelerate

struct HumanPose {
  var keypoints = [CGPoint]()
  var scores = [Double]()
  var score: Double = 0.0
}

class PoseRender {
  
  let joints: [(Int,Int)] = [
    (0, 1), // nose , l_eye
    (0, 2), // nose , r_eye
    (1, 3),
    (2, 4),
    (2, 4),
    (5, 7), // l shoulder l elbow
    (7, 9), // l elbow l wrist
    (6, 8), // r shoulder r elbow
    (8, 10),// r elbow r wrist
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6), // l shoulder r shoulder
    (11, 12), //
    (5, 11),
    (6, 12),
  ]
  
  var sourceImage: UIImage
  var poses = [HumanPose]()
  
  init(_ sourceImage: UIImage, keypoints: [Double], boxes: [Double], keypointsNumber: Int){
    self.sourceImage = sourceImage
    let peopleNum = Int(boxes.count / 4)
    for i in 0..<peopleNum {
      var pose = HumanPose()
      pose.keypoints = Array(repeating: CGPoint(), count: keypointsNumber)
      pose.scores = Array(repeating: 0.0, count: keypointsNumber)
      for j in 0..<keypointsNumber {
        let n = i * keypointsNumber * 3 + j * 3
        pose.keypoints[j] = CGPoint(x: keypoints[n], y: keypoints[n + 1])
        pose.scores[j] = keypoints[n + 2]
      }
      pose.score = vDSP.mean(pose.scores)
      poses.append(pose)
    }
  }
  
  func render() -> UIImage {
    
    let dstImageSize = sourceImage.size
    let dstImageFormat = UIGraphicsImageRendererFormat()
    
    dstImageFormat.scale = 1
    let renderer = UIGraphicsImageRenderer(size: dstImageSize,
                                           format: dstImageFormat)
    let dstImage = renderer.image { rendererContext in
      draw(image: sourceImage.cgImage!, in: rendererContext.cgContext)
      
      for pose in poses {
        for joint in joints {
          drawLine(from: pose.keypoints[joint.0],
                   to: pose.keypoints[joint.1],
                   in: rendererContext.cgContext)
          
        }
        for keypoint in pose.keypoints {
          draw(circle: keypoint, in: rendererContext.cgContext)
        }
      }
    }
    return dstImage
  }
  
  // Detecting human body poses in an image
  // https://developer.apple.com/documentation/coreml/model_integration_samples/detecting_human_body_poses_in_an_image
  func draw(image: CGImage, in cgContext: CGContext) {
    cgContext.saveGState()
    // The given image is assumed to be upside down; therefore, the context
    // is flipped before rendering the image.
    cgContext.scaleBy(x: 1.0, y: -1.0)
    // Render the image, adjusting for the scale transformation performed above.
    let drawingRect = CGRect(x: 0, y: -image.height, width: image.width, height: image.height)
    cgContext.draw(image, in: drawingRect)
    cgContext.restoreGState()
  }
  
  let segmentColor: UIColor = UIColor.systemTeal
  var segmentLineWidth: CGFloat = 2
  var jointColor: UIColor = UIColor.systemPink
  var jointRadius: CGFloat = 4
  
  func drawLine(from parentJoint: CGPoint,
                to childJoint: CGPoint,
                in cgContext: CGContext) {
    cgContext.setStrokeColor(segmentColor.cgColor)
    cgContext.setLineWidth(segmentLineWidth)
    
    cgContext.move(to: parentJoint)
    cgContext.addLine(to: childJoint)
    cgContext.strokePath()
  }
  
  private func draw(circle joint: CGPoint, in cgContext: CGContext) {
    cgContext.setFillColor(jointColor.cgColor)
    
    let rectangle = CGRect(x: joint.x - jointRadius, y: joint.y - jointRadius,
                           width: jointRadius * 2, height: jointRadius * 2)
    cgContext.addEllipse(in: rectangle)
    cgContext.drawPath(using: .fill)
  }
}