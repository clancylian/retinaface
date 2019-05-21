#ifndef RETINAFACE_H
#define RETINAFACE_H

#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "tensorrt/trtretinafacenet.h"

using namespace cv;
using namespace std;
using namespace caffe;

struct anchor_win
{
    float x_ctr;
    float y_ctr;
    float w;
    float h;
};

struct anchor_box
{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct FacePts
{
    float x[5];
    float y[5];
};

struct FaceDetectInfo
{
    float score;
    anchor_box rect;
    FacePts pts;
};

struct anchor_cfg
{
public:
    //int STRIDE;
    vector<int> SCALES;
    int BASE_SIZE;
    vector<float> RATIOS;
    int ALLOWED_BORDER;

    anchor_cfg()
    {
        //STRIDE = 0;
        SCALES.clear();
        BASE_SIZE = 0;
        RATIOS.clear();
        ALLOWED_BORDER = 0;
    }
};

class RetinaFace
{
public:
    RetinaFace(string &model, int ctx_id = 0, string network = "net3",
               float nms = 0.4, bool nocrop = false, float decay4 = 0.5, bool vote = false);
    ~RetinaFace();

    void detectBatchImages(vector<cv::Mat> imgs, float threshold=0.5);
    void detect(cv::Mat img, float threshold=0.5, float scales=1.0);
private:
    vector<anchor_box> bbox_pred(vector<anchor_box> anchors, vector<cv::Vec4f> regress);
    vector<FacePts> landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts);
    static bool CompareBBox(const FaceDetectInfo &a, const FaceDetectInfo &b);
    std::vector<FaceDetectInfo> nms(std::vector<FaceDetectInfo> &bboxes, float threshold);
private:
    boost::shared_ptr<Net<float> > Net_;
    
    TrtRetinaFaceNet *trtNet;
    int width;
    int height;
    int channels;
    int maxbatchsize;
    float *cpuBuffers;

    float pixel_means[3] = {0.0, 0.0, 0.0};
    float pixel_stds[3] = {1.0, 1.0, 1.0};
    float pixel_scale = 1.0;

    int ctx_id;
    string network;
    float decay4;
    float nms_threshold;
    bool vote;
    bool nocrop;
    bool debug;

    bool preprocess;
    vector<float> _ratio;

    vector<int> _feat_stride_fpn;
    map<int, anchor_cfg> cfg;

    map<string, vector<anchor_box>> _anchors_fpn;
    map<string, int> _num_anchors;

    bool use_landmarks;
};

#endif // RETINAFACE_H
