
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>

#include "CatFinder.h"

//#define USE_ESP32CAM
#ifdef USE_ESP32CAM
#define ESP32CAM_VIDEO_URL "http://10.42.0.69/stream.mjpeg"
#endif

using namespace cv;
using namespace dnn;

int main() {
    CatFinder* catFinder = new CatFinder();

    if (!catFinder->initialised()) {
        return -1;
    }

    // open a window
    static const std::string windowName = "window";
    namedWindow(windowName, WINDOW_NORMAL);

    // open a video device
    VideoCapture cap;
#ifdef USE_ESP32CAM
    if (!cap.open(ESP32CAM_VIDEO_URL, CAP_FFMPEG)) {
        std::cout << "failed to open camera stream" << std::endl;
        return -1;
    }
#else
    if (!cap.open(0)) {
        std::cout << "failed to open camera" << std::endl;
        return -1;
    }

    /*
    if(!cap.open("../cat.mp4")){
        std::cout << "failed to open video" << std::endl;
        return -1;
    }
    */
#endif

    // keep running until a key is pressed
    // call to waitKey() is needed for the window to show up
    // and change it's contents
    while (waitKey(1) < 0) {
        // get a frame
        Mat frame;
        cap >> frame;

        bool overlay = true;
        catFinder->processFrame(&frame, overlay);

        if (!frame.empty()) {
            std::vector<CatBox>* cats = catFinder->getCats();
            for (int i = 0; i < cats->size(); i++) {
                CatBox cat = cats->at(i);
                /*
                printf("found: %.2f %i %i %i %i\n", cat.confidence, cat.x,
                cat.y, cat.width, cat.height);
                */
            }

            // show the frame
            imshow(windowName, frame);
        }
    }
}
