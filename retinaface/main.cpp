#include <iostream>
#include <RetinaFace.h>
#include <caffe/caffe.hpp>
#include "timer.h"

using namespace caffe;
using namespace std;

int main()
{
//    std::unique_ptr<caffe::Caffe> caffe_context_;
//    cudaSetDevice(0);
//    caffe_context_.reset(new Caffe);
//    Caffe::Set(caffe_context_.get());

    int gpuid = 0;
    string path = "../model";
    RetinaFace rf(path, gpuid, "net3");

    cv::Mat img = cv::imread("/home/ubuntu/Project/retinaFaceReImp/data/img.jpg");

    int count = 0;
    float time = 0;

    RK::Timer ti;
    ti.reset();
    while(1){
        rf.detect(img, 0.9);
        count++;
        if(count == 100) {
            break;
        }
    }

    cout << "time cost: " << ti.elapsedMilliSeconds() << endl;

    rf.detect(img, 0.9);

    cout << "Hello World!" << endl;
    return 0;
}

