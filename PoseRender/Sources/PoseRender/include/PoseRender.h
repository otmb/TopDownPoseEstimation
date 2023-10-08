#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

#ifndef PoseRender_h
#define PoseRender_h

#ifdef OBJC_DEBUG
#define DEBUG_MSG(str) do { std::cout << str << std::endl; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif

@interface PoseRender : NSObject

- (UIImage*) renderHumanPose: (UIImage*) uiImage
                   keypoints: (float*) keypoints
                   peopleNum: (int) peopleNum
                       boxes: (float*) boxes;

@end

#endif /* PoseRender_h */
