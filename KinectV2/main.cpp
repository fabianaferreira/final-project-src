#include <iostream>
#include <sstream>
#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>

#define USE_GESTURE
#include "NtKinect.h"

using namespace std;
using namespace cv;

void putText(cv::Mat& img, string s, cv::Point p) {
	cv::putText(img, s, p, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2.0);
}

void doJob() {
	NtKinect kinect;
	//kinect.setGestureFile(L"Gostar.gba");
	//kinect.setGestureFile(L"Seated.gbd");
	//auto start = chrono::steady_clock::now();

	vector<KeyPoint> keypoints;
	Mat descriptors;
	Mat greyMat;
	Mat output;


	unsigned frameCounter = 0;
	while (1) {
		kinect.setRGB();
		//kinect.setSkeleton();
		/*for (auto person : kinect.skeleton) {
			for (auto joint : person) {
				if (joint.TrackingState == TrackingState_NotTracked) continue;
				ColorSpacePoint cp;
				kinect.coordinateMapper->MapCameraPointToColorSpace(joint.Position, &cp);
				cv::rectangle(kinect.rgbImage, cv::Rect((int)cp.X - 5, (int)cp.Y - 5, 10, 10), cv::Scalar(0, 0, 255), 2);
			}
		}
		kinect.setGesture();*/
		/*for (int i = 0; i<kinect.discreteGesture.size(); i++) {
			auto g = kinect.discreteGesture[i];
			putText(kinect.rgbImage, kinect.gesture2string(g.first) + " " + to_string(g.second), cv::Point(50, 30 + 30 * i));
		}
		for (int i = 0; i<kinect.continuousGesture.size(); i++) {
			auto g = kinect.continuousGesture[i];
			putText(kinect.rgbImage, kinect.gesture2string(g.first) + " " + to_string(g.second), cv::Point(500, 30 + 30 * i));
		}*/

		//frameCounter++;
		//auto end = chrono::steady_clock::now();
		//int seconds = chrono::duration_cast<chrono::seconds>(end - start).count();
		//if (seconds != 0)
		//	putText(kinect.rgbImage, "FPS: " + to_string(frameCounter/seconds), cv::Point(50, 200));

			
		cvtColor(kinect.rgbImage, greyMat, cv::COLOR_BGR2GRAY);

		Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();

		detector->detect(greyMat, keypoints);

		drawKeypoints(greyMat, keypoints, output);

		imshow("fastfeature", output);
		auto key = cv::waitKey(1);
		if (key == 'q') break;
	}
	cv::destroyAllWindows();
}

int main(int argc, char** argv) {
	try {
		doJob();
	}
	catch (exception &ex) {
		cout << ex.what() << endl;
		string s;
		cin >> s;
	}
	return 0;
}
