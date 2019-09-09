#include "functions.h"

using namespace std;
using namespace cv;

int main(int, char **)
{
    int keyboard;
    int numFrames = 0;
    double weight = 0.05;
    Scalar color = Scalar(255, 0, 0);

    VideoCapture cap(0); // open the default camera

    if (!cap.isOpened()) // check if we succeeded
        return -1;

    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
    Mat frame, gray, max_drawing, thresholded, bg, gaussian;
    vector<Point> maxCountour;

    cap >> frame;

    /*Initializing background*/
    bg = Mat::zeros(frame.size(), CV_32FC1);

    for (;;)
    {
        /*Initializing maximum contour*/
        max_drawing = Mat::zeros(frame.size(), CV_8UC3);

        cap >> frame; // get a new frame from camera
        if (frame.empty())
            break;

        cvtColor(frame, gray, CV_BGR2GRAY);
        GaussianBlur(gray, gaussian, Size(7, 7), 0);

        if (numFrames < 30)
        {
            /*Estimates background based on a moving average*/
            runningAverage(bg, gaussian, weight);
        }
        else
        {
            segmentImage(gaussian, bg, thresholded, maxCountour);

            if (!thresholded.empty())
            {
                drawContours(max_drawing, vector<vector<Point>>(1, maxCountour), -1, color, 1, 8);

                //Detect Hull in current contour
                vector<vector<Point>> hulls(1);
                vector<vector<int>> hullsI(1);
                convexHull(Mat(maxCountour), hulls[0], false);
                convexHull(Mat(maxCountour), hullsI[0], false);
                drawContours(max_drawing, hulls, -1, cv::Scalar(0, 255, 0), 2);

                imshow("gray", gaussian);
                // imshow("thresholded", thresholded);
                imshow("max_drawing", max_drawing);
            }
        }

        numFrames += 1;

        //get the input from the keyboard
        keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    return 0;
}