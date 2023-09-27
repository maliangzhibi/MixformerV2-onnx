#include <iostream>
#include <cstdlib>
#include <string>

#include "ostrack_onnx.h"

void track(OSTrack *tracker, const char *video_path)
{
    // Read video.
    cv::VideoCapture capture;
    bool ret;
    if (strlen(video_path)==1)
        ret = capture.open(atoi(video_path));
    else
        ret = capture.open(video_path);

    // Exit if video not opened.
    if (!ret)
        std::cout << "Open cap failed!" << std::endl;

    // Read first frame.
    cv::Mat frame;

    bool ok = capture.read(frame);
    if (!ok)
    {
        std::cout<< "Cannot read video file" << std::endl;
        return;
    }

    // Select a rect.
    cv::namedWindow("demo");
    cv::Rect trackWindow = cv::selectROI("demo", frame);
    // cv::Rect trackWindow(744, 417, 42, 95);
    

    // Initialize tracker with first frame and rect.
    std::cout << "Start track init ..." << std::endl;
    std::cout << "==========================" << std::endl;
    DrOBB bbox;
    bbox.box.x0 = trackWindow.x;
    bbox.box.x1 = trackWindow.x+trackWindow.width;
    bbox.box.y0 = trackWindow.y;
    bbox.box.y1 = trackWindow.y+trackWindow.height;
    tracker->init(frame, bbox);
    std::cout << "==========================" << std::endl;
    std::cout << "Init done!" << std::endl;
    std::cout << std::endl;
    for (;;)
    {
        // Read a new frame.
        capture >> frame;
        if (frame.empty())
            break;

        // Start timer
        double t = (double)cv::getTickCount();

        // Update tracker.
        DrOBB bbox = tracker->track(frame);

        // Calculate Frames per second (FPS)
        double fps = cv::getTickFrequency() / ((double)cv::getTickCount() - t);

        // Result to rect.
        cv::Rect rect;
        rect.x = bbox.box.x0;
        rect.y = bbox.box.y0;
        rect.width = int(bbox.box.x1 - bbox.box.x0);
        rect.height = int(bbox.box.y1 - bbox.box.y0);

        std::cout << "[x0, y0, w, h]: [" << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << "]" << std::endl;
        std::cout << "score: " << bbox.score << std::endl;

        // Boundary judgment.
        cv::Mat track_window;
        if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= frame.cols && 0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= frame.rows)
        {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0));
        }

        // Display FPS 
        std::cout << "FPS: " << fps << std::endl;
        std::cout << "==========================" << std::endl;
        std::cout << std::endl;


        // Display result.
        cv::imshow("demo", frame);
        cv::waitKey(1);

        // Exit if 'q' pressed.
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }
    cv::destroyWindow("demo");
    capture.release();
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [modelpath] [videopath(file or camera)]\n", argv[0]);
        return -1;
    }

    // Get model path.
    const char* model_path = argv[1]; // OSTrack-mnn/model/ostrack-bit8.mnn";

    // Get video path.
    const char* video_path = argv[2]; 

    // Build tracker.
    OSTrack *ostracker;
    ostracker = new OSTrack(model_path, 8);
    track(ostracker, video_path);

    return 0;
}