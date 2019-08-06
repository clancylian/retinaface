#ifndef CALIBRATIONTABLEIMPL_H
#define CALIBRATIONTABLEIMPL_H
#include "calibrationtable.h"
#include <opencv2/opencv.hpp>
using namespace cv;

class CalibrationTableRetinaFace : public CalibrationTableBase
{
public:
    CalibrationTableRetinaFace();
    virtual ~CalibrationTableRetinaFace();
private:
    virtual cv::Mat preprocess(cv::Mat img) override;
};

#endif // CALIBRATIONTABLEIMPL_H
