#include "CalibrationTableImpl.h"

CalibrationTableRetinaFace::CalibrationTableRetinaFace()
{
    input_c = 3;
    input_h = 320;
    input_w = 320;

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

cv::Mat CalibrationTableRetinaFace::preprocess(cv::Mat img)
{
    cv::Mat image_rgb, sample_float;
    cv::cvtColor(img, image_rgb, CV_BGR2RGB);
    image_rgb.convertTo(sample_float, CV_32FC3);
    return sample_float;
}
