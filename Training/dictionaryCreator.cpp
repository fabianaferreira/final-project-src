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
#include<opencv4/opencv2/core/persistence.hpp>
#include<opencv4/opencv2/core/types.hpp>

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
    Mat featuresUnclustered;
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
        cout << "Video " << i << "/" << filesQnt << endl;
        // Create a VideoCapture object and open the input file
        // If the input is the web camera, pass 0 instead of the video file name
        VideoCapture cap(files->at(i));

        // Check if camera opened successfully
        if (!cap.isOpened())
        {
            cout << "Error opening video stream or file" << endl;
            return -1;
        }

        do
        {
            Mat frame;
            // Capture frame-by-frame
            cap >> frame;

            /*----- SIFT -----*/
            // Detect
            Ptr<Feature2D> f2d = xfeatures2d::SiftFeatureDetector::create();

            //Detect feature points
            f2d->detect(frame, keypoints);

            //Compute the descriptors for each keypoint
            f2d->compute(frame, keypoints, descriptor);

            //Put the all frame features descriptors in a single Mat object
            featuresUnclustered.push_back(descriptor);

            /*-----BOF-----*/
            // Construct BOWKMeansTrainer

            //Create the BoW (or BoF) trainer
            BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
            //cluster the feature vectors
            Mat dictionary = bowTrainer.cluster(featuresUnclustered);
            //store the vocabulary
            FileStorage fs("dictionary.yml", FileStorage::APPEND);
            fs << "vocabulary" << dictionary;
            fs.release();

        } while (!frame.empty());

        cap.release();
    }

    cout << featuresUnclustered.size() << endl;

    delete files;
    return 0;
}
// //Construct BOWKMeansTrainer
// //the number of bags
// int dictionarySize=200;
// //define Term Criteria
// TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
// //retries number
// int retries=1;
// //necessary flags
// int flags=KMEANS_PP_CENTERS;
// //Create the BoW (or BoF) trainer
// BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
// //cluster the feature vectors
// Mat dictionary=bowTrainer.cluster(featuresUnclustered);
// //store the vocabulary
// FileStorage fs("dictionary.yml", FileStorage::WRITE);
// fs << "vocabulary" << dictionary;
// fs.release();
