#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#include <iostream>
#include <numeric>
#import "PoseRender.h"

const size_t keypointsNumber = 17;
@implementation PoseRender

- (UIImage*) renderHumanPose: (UIImage*) uiImage
                   keypoints: (float*) keypoints
                   peopleNum: (int) peopleNum
                       boxes: (float*) boxes
{
  std::vector<HumanPose> poses;
  for (int i = 0; i < peopleNum; ++i) {
    HumanPose pose{
      std::vector<cv::Point2f>(keypointsNumber, cv::Point2f(-1.0f, -1.0f)),
      std::vector<float>(keypointsNumber, 0.0),
      1.0};
    for (int j = 0; j < keypointsNumber; ++j) {
      int n = i * keypointsNumber * 3 + j * 3;
      pose.keypoints[j].x = keypoints[n];
      pose.keypoints[j].y = keypoints[n + 1];
      pose.scores[j] = keypoints[n + 2];
    }
    pose.score = std::accumulate(pose.scores.begin(), pose.scores.end(), 0.0) / pose.scores.size();
    poses.push_back(pose);
    DEBUG_MSG("score: " << pose.score);
  }
  
  cv::Mat outputImg;
  UIImageToMat(uiImage, outputImg);
  cv::cvtColor(outputImg, outputImg, cv::COLOR_RGB2RGBA);
  
  static const cv::Scalar colors[keypointsNumber] =
  {
    cv::Scalar(255, 0, 0),
    cv::Scalar(255, 85, 0),
    cv::Scalar(255, 170, 0),
    cv::Scalar(255, 255, 0),
    cv::Scalar(170, 255, 0),
    cv::Scalar(85, 255, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 85),
    cv::Scalar(0, 255, 170),
    cv::Scalar(0, 255, 255),
    cv::Scalar(0, 170, 255),
    cv::Scalar(0, 85, 255),
    cv::Scalar(0, 0, 255),
    cv::Scalar(85, 0, 255),
    cv::Scalar(170, 0, 255),
    cv::Scalar(255, 0, 255),
    cv::Scalar(255, 0, 170),
  };
  /*
   0: nose        1: l eye      2: r eye    3: l ear   4: r ear
   5: l shoulder  6: r shoulder 7: l elbow  8: r elbow
   9: l wrist    10: r wrist    11: l hip   12: r hip  13: l knee
   14: r knee    15: l ankle    16: r ankle
   */
  static const std::pair<int, int> keypointsOP[] = {
    {0, 1}, // nose , l_eye
    {0, 2}, // nose , r_eye
    {1, 3},
    {2, 4},
    {2, 4},
    {5, 7}, // l shoulder l elbow
    {7, 9}, // l elbow l wrist
    {6, 8}, // r shoulder r elbow
    {8, 10},// r elbow r wrist
    {11, 13},
    {13, 15},
    {12, 14},
    {14, 16},
    {5, 6}, // l shoulder r shoulder
    {11, 12}, //
    {5, 11},
    {6, 12},
  };
  
  const int stickWidth = 2;
  const cv::Point2f absentKeypoint(-1.0f, -1.0f);
  for (auto& pose : poses) {
    for (size_t keypointIdx = 0; keypointIdx < pose.keypoints.size(); keypointIdx++) {
      if (pose.keypoints[keypointIdx] != absentKeypoint) {
        cv::circle(outputImg, pose.keypoints[keypointIdx], 2, colors[keypointIdx], -1);
      }
    }
  }
  
  std::vector<std::pair<int, int>> limbKeypointsIds;
  if (!poses.empty()) {
    limbKeypointsIds.insert(limbKeypointsIds.begin(), std::begin(keypointsOP), std::end(keypointsOP));
  }
  
  cv::Mat pane = outputImg.clone();
  for (auto pose : poses) {
    for (const auto& limbKeypointsId : limbKeypointsIds) {
      std::pair<cv::Point2f, cv::Point2f> limbKeypoints(pose.keypoints[limbKeypointsId.first],
                                                        pose.keypoints[limbKeypointsId.second]);
      if (limbKeypoints.first == absentKeypoint || limbKeypoints.second == absentKeypoint) {
        continue;
      }
      
      float meanX = (limbKeypoints.first.x + limbKeypoints.second.x) / 2;
      float meanY = (limbKeypoints.first.y + limbKeypoints.second.y) / 2;
      cv::Point difference = limbKeypoints.first - limbKeypoints.second;
      double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
      int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
      std::vector<cv::Point> polygon;
      cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, stickWidth), angle, 0, 360, 1, polygon);
      cv::fillConvexPoly(pane, polygon, colors[limbKeypointsId.second]);
    }
  }
  cv::addWeighted(outputImg, 0.4, pane, 0.6, 0, outputImg);
  
  std::vector<float> _boxes(&boxes[0], boxes + peopleNum * 4);
  for (int j = 0; j < peopleNum; ++j) {
    std::vector<float> box = { _boxes[j*4], _boxes[j*4+1], _boxes[j*4+2], _boxes[j*4+3] };
    cv::rectangle(outputImg, cv::Point(box[0], box[1]), cv::Point(box[2] + box[0], box[3] + box[1]), cv::Scalar(255,0,0), 2);
  }
  
  UIImage *preview = MatToUIImage(outputImg);
  outputImg.release();
  return preview;
}

struct HumanPose {
  std::vector<cv::Point2f> keypoints;
  std::vector<float> scores;
  float score;
};

@end
