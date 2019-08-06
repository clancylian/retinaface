#include <iostream>
#include <RetinaFace.h>
#include "timer.h"

using namespace std;

int main()
{
//    std::unique_ptr<caffe::Caffe> caffe_context_;
//    cudaSetDevice(0);
//    caffe_context_.reset(new Caffe);
//    Caffe::Set(caffe_context_.get());

    string path = "../model";
    RetinaFace *rf = new RetinaFace(path, "net3");

    //cv::VideoCapture cap(0);
    cv::Mat img = cv::imread("/home/ubuntu/Pictures/t1.jpg");
    
    vector<Mat> imgs;
    for(int i = 0; i < 64; i++) {
        string prefix = "/home/ubuntu/Project/faceengine/faceengine/test/FaceEngineTest/images/gakki/";
        string imgname = prefix + std::to_string(2005 + i) + ".jpg";
        cv::Mat src = cv::imread(imgname);
        imgs.push_back(img.clone());
    }

    //rf.detect(img, 0.9);
    //rf.detectBatchImages(imgs, 0.9);

//    int c = 0;
//    float time = 0;
//    int count = 0;

    //注：使用OPENCV计时和timer类计时有点偏差
    float time = 0;
    int count = 0;
    RK::Timer ti;

    while(/*cap.read(img)*/1) {
        ti.reset();
        //double t1 = (double)getTickCount();
        rf->detect(img, 0.9);
        //rf->detectBatchImages(imgs, 0.9);

        //t1 = (double)getTickCount() - t1;
        //time += t1 * 1000 / cv::getTickFrequency();
        time += ti.elapsedMilliSeconds();
        count ++;
        if(count % 1000 == 0) {
            printf("face detection average time = %f.\n", time / count);
        }
    }
    //t1 = (double)getTickCount() - t1;
    //std::cout << "all compute time :" << t1*1000.0 / cv::getTickFrequency() << " ms \n";

    return 0;
}

