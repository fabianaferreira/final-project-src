#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/video.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/core/persistence.hpp>
#include <opencv4/opencv2/core/types.hpp>

#include "functions.h"

using namespace std;
using namespace cv;

#define DIRECTORY_PATH "/home/fabiana/Desktop/projeto-final-src/HandGestureRecognition/datasets/HandGesture"

int main(int, char **)
{
    Mat frame, output;
    const string path = "/home/fabiana/Desktop/projeto-final-src/HandGestureRecognition/datasets/HandGesture";
    vector<string> *files = new vector<string>();

    vector<KeyPoint> keypoints;
    Mat descriptor;
    xfeatures2d::SiftDescriptorExtractor detector;

    //the number of bags
    int dictionarySize = 16;
    //define Term Criteria
    TermCriteria tc(TermCriteria::MAX_ITER, 100, 0.001);
    //retries number
    int retries = 1;
    //necessary flags
    int flags = KMEANS_PP_CENTERS;

    //Getting all video filenames
    getFilesList(path, files, false);
    unsigned filesQnt = files->size();

    for (unsigned i = 0; i < filesQnt; i++)
    {
        Mat featuresUnclustered, dictionary;
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
            Mat frame;

            // Capture frame-by-frame
            cap >> frame;

            Ptr<Feature2D> f2d = xfeatures2d::SURF::create(dictionarySize);
            f2d->detectAndCompute(frame, noArray(), keypoints, descriptor);

            //Put the all frame features descriptors in a single Mat object
            featuresUnclustered.push_back(descriptor);

            //Create the BoW (or BoF) trainer
            BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
            //cluster the feature vectors
            dictionary = bowTrainer.cluster(featuresUnclustered);

            cout << featuresUnclustered.size() << endl;

            frameNumber++;
        }

        /*-----BOF-----*/
        //store the vocabulary
        FileStorage fs("dictionary.yml", FileStorage::APPEND);
        fs << "vocabulary" << dictionary;
        fs.release();
        featuresUnclustered.release();
        cap.release();

        break;
    }

    delete files;
    return 0;
}