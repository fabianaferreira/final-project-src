// #include "functions.h"

// using namespace std;
// using namespace cv;

// void runningAverage(Mat &bg, Mat image, double weight)
// {
//     if (bg.empty())
//     {
//         image.copyTo(bg);
//     }
//     accumulateWeighted(image, bg, weight);
// }

// vector<Point> getMaximumContour(vector<vector<Point>> cnts, vector<Vec4i> hierarchy)
// {
//     double max = 0;
//     int currentParent = -1;
//     int maxParent = -1;
//     unsigned counter = 0;
//     vector<Point> maxCnt;

//     for (auto cnt : cnts)
//     {
//         currentParent = hierarchy[counter][3];
//         double currentArea = contourArea(cnt);

//         if (currentArea > max && currentParent >= maxParent)
//         {
//             maxCnt = cnt;
//             max = currentArea;
//             maxParent = currentParent;
//         }
//         counter += 1;
//     }

//     return maxCnt;
// }

// void segmentImage(Mat currentFrame, Mat bg, Mat &thresholded, vector<Point> &segmented)
// {
//     Mat diff, hierarchy, bg8b, result;
//     vector<vector<Point>> contours;

//     bg.convertTo(bg8b, CV_8UC1);

//     absdiff(bg8b, currentFrame, diff);

//     adaptiveThreshold(diff, thresholded, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 2);

//     // threshold(diff, thresholded, 0, 255, THRESH_BINARY + THRESH_OTSU);

//     //need to erode and dilate

//     findContours(thresholded, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

//     if (!contours.empty())
//     {
//         segmented = getMaximumContour(contours, hierarchy);
//     }
// }