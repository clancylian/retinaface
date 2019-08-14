## compile
```
$ mkdir build
$ cd build
$ cmake ../
$ make
```

### I have write a base class to do calibrate, you can derived from it.

```c++
class CalibrationTableRetinaFace : public CalibrationTableBase
{
public:
    CalibrationTableRetinaFace();
    virtual ~CalibrationTableRetinaFace();
private:
    virtual cv::Mat preprocess(cv::Mat img) override;
};

CalibrationTableRetinaFace::CalibrationTableRetinaFace()
{
    //设置参数，也可以调用接口设置
    input_c = 3;
    input_h = 448;
    input_w = 448;

    num_per_batch = 1;

    input_blob_name = "data";
    output_blob_name = {"face_rpn_cls_prob_reshape_stride32",
                        "face_rpn_bbox_pred_stride32",
                        "face_rpn_landmark_pred_stride32",
                        "face_rpn_cls_prob_reshape_stride16",
                        "face_rpn_bbox_pred_stride16",
                        "face_rpn_landmark_pred_stride16",
                        "face_rpn_cls_prob_reshape_stride8",
                        "face_rpn_bbox_pred_stride8",
                        "face_rpn_landmark_pred_stride8"};
}

CalibrationTableRetinaFace::~CalibrationTableRetinaFace()
{

}
//impl
cv::Mat CalibrationTableRetinaFace::preprocess(cv::Mat img)
{
    cv::Mat image_rgb, sample_float;
    cv::cvtColor(img, image_rgb, CV_BGR2RGB);
    image_rgb.convertTo(sample_float, CV_32FC3);
    return sample_float;
}
```



###  you need to set model path and input size and output and so on.

```c++
int main(int argc, char** argv)
{
    CalibrationTableRetinaFace *calibra = new CalibrationTableRetinaFace();
    //设置参数
    //模型是否加密
    calibra->setEncryption(false);
    //数据路径
    calibra->setInputDir("../INT8-Calibration-Tool/dataSet");
    //预处理后输出路径
    calibra->setOutputDir("../batches");
    //模型路径
    calibra->setModelPath("../model/mnet-deconv-0517.prototxt",
                          "../model/mnet-deconv-0517.caffemodel");

    //校准
    calibra->doCalibration();

    delete calibra;

    return 0;
}
```
