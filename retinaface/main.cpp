#include <iostream>
#include <RetinaFace.h>

using namespace std;

int main()
{
    string path = "../model";
    RetinaFace rf(path, "net3");

    cv::Mat img = cv::imread("/home/ubuntu/Pictures/1.jpg");
    
    vector<Mat> imgs;
    for(int i = 0; i < 8; i++) {
        string prefix = "/home/ubuntu/Project/faceengine/faceengine/test/FaceEngineTest/images/gakki/";
        string imgname = prefix + std::to_string(2005 + i) + ".jpg";
        cv::Mat src = cv::imread(imgname);
        imgs.push_back(img.clone());
    }

    //rf.detect(img, 0.9);
    //rf.detectBatchImages(imgs, 0.9);

    int c = 0;
    double t1 = (double)getTickCount();
    while(1) {
        //rf.detect(img, 0.9);
        rf.detectBatchImages(imgs, 0.9);
        c++;
        if(c >= 1000) {
            break;
        }
    }
    t1 = (double)getTickCount() - t1;
    std::cout << "all compute time :" << t1*1000.0 / cv::getTickFrequency() << " ms \n";

    return 0;
}

