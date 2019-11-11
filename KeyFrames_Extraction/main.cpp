#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/video.hpp>

#include "KeyFramesAnalysis.h"

using namespace std;
using namespace cv;

#define DIRECTORY_PATH "/home/fabiana/Desktop/projeto-final-src/HandGestureRecognition/datasets/HandGesture"

int main(int, char **)
{
    Mat frame, hist;
    const string path = "/home/fabiana/Desktop/projeto-final-src/HandGestureRecognition/datasets/HandGesture";
    vector<string> *files = new vector<string>();
    vector<float> *totalVideoEntropy = new vector<float>();

    int histSize = 256;
    int hist_w = 512;
    int hist_h = 400;

    //Getting all video filenames
    getFilesList(path, files, false);
    unsigned filesQnt = files->size();

    for (unsigned i = 0; i < filesQnt; i++)
    {
        unsigned frameNumber = 0;

        cout << "Video " << i << "/" << filesQnt << endl;
        // Create a VideoCapture object and open the input file
        // If the input is the web camera, pass 0 instead of the video file name
        VideoCapture cap(files->at(i));

        unsigned frameQnt = cap.get(CAP_PROP_FRAME_COUNT);

        // Check if camera opened successfully
        if (!cap.isOpened())
        {
            cout << "Error opening video stream or file" << endl;
            return -1;
        }

        while (frameNumber < frameQnt)
        {
            Mat frame, gray;

            // Capture frame-by-frame
            cap >> frame;

            //Converting to gray scale so as to calculate entropy
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            hist = myEntropy(gray, histSize);
            totalVideoEntropy->push_back(calcEntropy(hist, gray.size(), histSize));

            frameNumber++;
        }

        
        cout << getPeaksPositions(totalVideoEntropy).size() << endl;
        cap.release();
        totalVideoEntropy->clear();

        break;
    }

    delete files;
    delete totalVideoEntropy;
    return 0;
}