#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

int main(int, char **)
{
    int keyboard;
    int numFrames = 0;
    Scalar color = Scalar(255, 0, 0);

    VideoCapture cap(0); // open the default camera

    if (!cap.isOpened()) // check if we succeeded
        return -1;

    Mat frame, output;

    cap >> frame;

    for (;;)
    {
        cap >> frame; // get a new frame from camera
        if (frame.empty())
            break;

        // Detect
        Ptr<Feature2D> f2d = xfeatures2d::SiftFeatureDetector::create();
        vector<KeyPoint> keypoints;
        //The SIFT feature extractor and descriptor
        xfeatures2d::SiftDescriptorExtractor descriptor; 
        //Detect feature points
        f2d->detect(frame, keypoints);
        //compute the descriptors for each keypoint
        f2d->compute(frame, keypoints, descriptor);  

        // Add results to image and save.
        Mat output;
        drawKeypoints(frame, keypoints, output);

        imshow("output", output);

        //get the input from the keyboard
        keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    return 0;
}