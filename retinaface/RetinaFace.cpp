#include "RetinaFace.h"

#define TRT

//processing
anchor_win  _whctrs(anchor_box anchor)
{
    //Return width, height, x center, and y center for an anchor (window).
    anchor_win win;
    win.w = anchor.x2 - anchor.x1 + 1;
    win.h = anchor.y2 - anchor.y1 + 1;
    win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
    win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

    return win;
}

anchor_box _mkanchors(anchor_win win)
{
    //Given a vector of widths (ws) and heights (hs) around a center
    //(x_ctr, y_ctr), output a set of anchors (windows).
    anchor_box anchor;
    anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
    anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
    anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
    anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

    return anchor;
}

vector<anchor_box> _ratio_enum(anchor_box anchor, vector<float> ratios)
{
    //Enumerate a set of anchors for each aspect ratio wrt an anchor.
    vector<anchor_box> anchors;
    for(size_t i = 0; i < ratios.size(); i++) {
        anchor_win win = _whctrs(anchor);
        float size = win.w * win.h;
        float scale = size / ratios[i];

        win.w = std::round(sqrt(scale));
        win.h = std::round(win.w * ratios[i]);

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

vector<anchor_box> _scale_enum(anchor_box anchor, vector<int> scales)
{
    //Enumerate a set of anchors for each scale wrt an anchor.
    vector<anchor_box> anchors;
    for(size_t i = 0; i < scales.size(); i++) {
        anchor_win win = _whctrs(anchor);

        win.w = win.w * scales[i];
        win.h = win.h * scales[i];

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

vector<anchor_box> generate_anchors(int base_size = 16, vector<float> ratios = {0.5, 1, 2},
                      vector<int> scales = {8, 64}, int stride = 16, bool dense_anchor = false)
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    anchor_box base_anchor;
    base_anchor.x1 = 0;
    base_anchor.y1 = 0;
    base_anchor.x2 = base_size - 1;
    base_anchor.y2 = base_size - 1;

    vector<anchor_box> ratio_anchors;

    ratio_anchors = _ratio_enum(base_anchor, ratios);

    vector<anchor_box> anchors;
    for(size_t i = 0; i < ratio_anchors.size(); i++) {
        vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }

    if(dense_anchor) {
        assert(stride % 2 == 0);
        vector<anchor_box> anchors2 = anchors;
        for(size_t i = 0; i < anchors2.size(); i++) {
            anchors2[i].x1 += stride / 2;
            anchors2[i].y1 += stride / 2;
            anchors2[i].x2 += stride / 2;
            anchors2[i].y2 += stride / 2;
        }
        anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
    }

    return anchors;
}

vector<vector<anchor_box>> generate_anchors_fpn(bool dense_anchor = false, map<int, anchor_cfg> cfg = {})
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    vector<vector<anchor_box>> anchors;
    map<int, anchor_cfg>::iterator cfg_iter = cfg.begin();
    for(;cfg_iter != cfg.end(); cfg_iter++) {
        //stride从小到大[8 16 32]
        anchor_cfg tmp = cfg_iter->second;
        int bs = tmp.BASE_SIZE;
        vector<float> ratios = tmp.RATIOS;
        vector<int> scales = tmp.SCALES;
        int stride = cfg_iter->first;

        vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
        anchors.push_back(r);
    }

    return anchors;
}

vector<anchor_box> anchors_plane(int height, int width, int stride, vector<anchor_box> base_anchors)
{
    /*
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: a base set of anchors
    */

    vector<anchor_box> all_anchors;
    for(size_t k = 0; k < base_anchors.size(); k++) {
        for(int ih = 0; ih < height; ih++) {
            int sh = ih * stride;
            for(int iw = 0; iw < width; iw++) {
                int sw = iw * stride;

                anchor_box tmp;
                tmp.x1 = base_anchors[k].x1 + sw;
                tmp.y1 = base_anchors[k].y1 + sh;
                tmp.x2 = base_anchors[k].x2 + sw;
                tmp.y2 = base_anchors[k].y2 + sh;
                all_anchors.push_back(tmp);
            }
        }
    }

    return all_anchors;
}

void clip_boxes(vector<anchor_box> &boxes, int width, int height)
{
    //Clip boxes to image boundaries.
    for(size_t i = 0; i < boxes.size(); i++) {
        if(boxes[i].x1 < 0) {
            boxes[i].x1 = 0;
        }
        if(boxes[i].y1 < 0) {
            boxes[i].y1 = 0;
        }
        if(boxes[i].x2 > width - 1) {
            boxes[i].x2 = width - 1;
        }
        if(boxes[i].y2 > height - 1) {
            boxes[i].y2 = height -1;
        }
//        boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
//        boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
//        boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
//        boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);
    }
}

void clip_boxes(anchor_box &box, int width, int height)
{
    //Clip boxes to image boundaries.
    if(box.x1 < 0) {
        box.x1 = 0;
    }
    if(box.y1 < 0) {
        box.y1 = 0;
    }
    if(box.x2 > width - 1) {
        box.x2 = width - 1;
    }
    if(box.y2 > height - 1) {
        box.y2 = height -1;
    }
//    boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
//    boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
//    boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
//    boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);

}

//######################################################################
//retinaface
//######################################################################

RetinaFace::RetinaFace(string &model, int ctx_id, string network,
                       float nms, bool nocrop, float decay4, bool vote)
    : ctx_id(ctx_id), network(network), decay4(decay4), nms_threshold(nms),
      vote(vote), nocrop(nocrop)
{
    preprocess = false;
    use_landmarks = true;
    //主干网络选择
    //_ratio = (1.,)
    int fmc = 3;

    if (network=="ssh" || network=="vgg") {
        pixel_means[0] = 103.939;
        pixel_means[1] = 116.779;
        pixel_means[2] = 123.68;
        preprocess = true;
    }
    else if(network == "net3") {
        _ratio = {1.0};
    }
    else if(network == "net3a") {
        _ratio = {1.0, 1.5};
    }
    else if(network == "net6") { //like pyramidbox or s3fd
        fmc = 6;
    }
    else if(network == "net5") { //retinaface
        fmc = 5;
    }
    else if(network == "net5a") {
        fmc = 5;
        _ratio = {1.0, 1.5};
    }

    else if(network == "net4") {
        fmc = 4;
    }
    else if(network == "net5a") {
        fmc = 4;
        _ratio = {1.0, 1.5};
    }
    else {
        std::cout << "network setting error" << network << std::endl;
    }

    //anchor配置
    if(fmc == 3) {
        _feat_stride_fpn = {32, 16, 8};

        anchor_cfg tmp;
        tmp.SCALES = {32, 16};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        cfg[32] = tmp;

        tmp.SCALES = {8, 4};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        cfg[16] = tmp;

        tmp.SCALES = {2, 1};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        cfg[8] = tmp;
    }
    else {
        std::cout << "please reconfig anchor_cfg" << network << std::endl;
    }

    //加载网络
#ifdef TRT
    trtNet = new TrtRetinaFaceNet("retina");
    trtNet->buildTrtContext(model + "/mnet-deconv-0517.prototxt", model+"/mnet-deconv-0517.caffemodel");

    int maxbatchsize = trtNet->getMaxBatchSize();
    int channels = trtNet->getChannel();
    int inputW = trtNet->getNetWidth();
    int inputH = trtNet->getNetHeight();
    //
    int inputsize = maxbatchsize * channels * inputW * inputH * sizeof(float);
    cpuBuffers = (float*)malloc(inputsize);
    memset(cpuBuffers, 0, inputsize);

    vector<int> outputW = trtNet->getOutputWidth();
    vector<int> outputH = trtNet->getOutputHeight();

    bool dense_anchor = false;
    vector<vector<anchor_box>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
    int sz = _feat_stride_fpn.size();
    for(size_t i = 0; i < anchors_fpn.size(); i++) {
        int stride = _feat_stride_fpn[sz-i-1];
        string key = "stride" + std::to_string(_feat_stride_fpn[sz-i-1]);
        _anchors_fpn[key] = anchors_fpn[i];
        _num_anchors[key] = anchors_fpn[i].size();
        //有三组不同输出宽高
        _anchors[key] = anchors_plane(outputH[sz-i-1], outputW[sz-i-1], stride, _anchors_fpn[key]);
    }
#else

#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    Net_.reset(new Net<float>((model + "/mnet-deconv-0517.prototxt"), TEST));
    Net_->CopyTrainedLayersFrom(model+"/mnet-deconv-0517.caffemodel");

    bool dense_anchor = false;
    vector<vector<anchor_box>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
    int sz = _feat_stride_fpn.size();
    for(size_t i = 0; i < anchors_fpn.size(); i++) {
        string key = "stride" + std::to_string(_feat_stride_fpn[sz-i-1]);
        _anchors_fpn[key] = anchors_fpn[i];
        _num_anchors[key] = anchors_fpn[i].size();
    }
 #endif
}

RetinaFace::~RetinaFace()
{
#ifdef TRT
    delete trtNet;

    free(cpuBuffers);
#endif
}

vector<anchor_box> RetinaFace::bbox_pred(vector<anchor_box> anchors, vector<cv::Vec4f> regress)
{
    //"""
    //  Transform the set of class-agnostic boxes into class-specific boxes
    //  by applying the predicted offsets (box_deltas)
    //  :param boxes: !important [N 4]
    //  :param box_deltas: [N, 4 * num_classes]
    //  :return: [N 4 * num_classes]
    //  """

    vector<anchor_box> rects(anchors.size());
    for(size_t i = 0; i < anchors.size(); i++) {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        float pred_ctr_x = regress[i][0] * width + ctr_x;
        float pred_ctr_y = regress[i][1] * height + ctr_y;
        float pred_w = exp(regress[i][2]) * width;
        float pred_h = exp(regress[i][3]) * height;

        rects[i].x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
        rects[i].y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
        rects[i].x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
        rects[i].y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
    }

    return rects;
}

anchor_box RetinaFace::bbox_pred(anchor_box anchor, cv::Vec4f regress)
{
    anchor_box rect;

    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = exp(regress[2]) * width;
    float pred_h = exp(regress[3]) * height;

    rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    return rect;
}

vector<FacePts> RetinaFace::landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts)
{
    vector<FacePts> pts(anchors.size());
    for(size_t i = 0; i < anchors.size(); i++) {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        for(size_t j = 0; j < 5; j ++) {
            pts[i].x[j] = facePts[i].x[j] * width + ctr_x;
            pts[i].y[j] = facePts[i].y[j] * height + ctr_y;
        }
    }

    return pts;
}

FacePts RetinaFace::landmark_pred(anchor_box anchor, FacePts facePt)
{
    FacePts pt;
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    for(size_t j = 0; j < 5; j ++) {
        pt.x[j] = facePt.x[j] * width + ctr_x;
        pt.y[j] = facePt.y[j] * height + ctr_y;
    }

    return pt;
}

bool RetinaFace::CompareBBox(const FaceDetectInfo & a, const FaceDetectInfo & b)
{
    return a.score > b.score;
}

std::vector<FaceDetectInfo> RetinaFace::nms(std::vector<FaceDetectInfo>& bboxes, float threshold)
{
    std::vector<FaceDetectInfo> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        //如果全部执行完则返回
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        anchor_box select_bbox = bboxes[select_idx].rect;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            anchor_box& bbox_i = bboxes[i].rect;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;   //<- float 型不加1
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;

   
            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
                mask_merged[i] = 1;
            }
        }
    }

    return bboxes_nms;
}

#ifdef TRT
vector<FaceDetectInfo> RetinaFace::postProcess(int inputW, int inputH, float threshold)
{
    string name_bbox = "face_rpn_bbox_pred_";
    string name_score ="face_rpn_cls_prob_reshape_";
    string name_landmark ="face_rpn_landmark_pred_";

    vector<FaceDetectInfo> faceInfo;
    for(size_t i = 0; i < _feat_stride_fpn.size(); i++) {
///////////////////////////////////////////////
        double s1 = (double)getTickCount();
///////////////////////////////////////////////
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        int stride = _feat_stride_fpn[i];

        string str = name_score + key;
        TrtBlob* score_blob = trtNet->blob_by_name(str);
        std::vector<float> score = score_blob->result[0];
        std::vector<float>::iterator begin = score.begin() + score.size() / 2;
        std::vector<float>::iterator end = score.end();
        score = std::vector<float>(begin, end);

        str = name_bbox + key;
        TrtBlob* bbox_blob = trtNet->blob_by_name(str);
        std::vector<float> bbox_delta = bbox_blob->result[0];

        str = name_landmark + key;
        TrtBlob* landmark_blob = trtNet->blob_by_name(str);
        std::vector<float> landmark_delta = landmark_blob->result[0];

        int width = score_blob->outputDims.w();
        int height = score_blob->outputDims.h();
        size_t count = width * height;
        size_t num_anchor = _num_anchors[key];

///////////////////////////////////////////////
        s1 = (double)getTickCount() - s1;
        std::cout << "s1 compute time :" << s1*1000.0 / cv::getTickFrequency() << " ms \n";
///////////////////////////////////////////////

        for(size_t num = 0; num < num_anchor; num++) {
            for(size_t j = 0; j < count; j++) {
                //置信度小于阈值跳过
                float conf = score[j + count * num];
                if(conf <= threshold) {
                    continue;
                }

                cv::Vec4f regress;
                float dx = bbox_delta[j + count * (0 + num * 4)];
                float dy = bbox_delta[j + count * (1 + num * 4)];
                float dw = bbox_delta[j + count * (2 + num * 4)];
                float dh = bbox_delta[j + count * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);

                //回归人脸框
                anchor_box rect = bbox_pred(_anchors[key][j + count * num], regress);
                //越界处理
                clip_boxes(rect, inputW, inputH);

                FacePts pts;
                for(size_t k = 0; k < 5; k++) {
                    pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                    pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                }
                //回归人脸关键点
                FacePts landmarks = landmark_pred(_anchors[key][j + count * num], pts);

                FaceDetectInfo tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                faceInfo.push_back(tmp);
            }
        }
    }

    //排序nms
    faceInfo = nms(faceInfo, 0.4);

    return faceInfo;
}

void RetinaFace::detect(Mat img, float threshold, float scales)
{
    if(img.empty()) {
        return;
    }

    double pre = (double)getTickCount();

    int inputW = trtNet->getNetWidth();
    int inputH = trtNet->getNetHeight();

    float sw = 1.0 * img.cols / inputW;
    float sh = 1.0 * img.rows / inputH;

    if(sw > 1.0 || sh > 1.0) {
        if(sw > sh) {
            cv::resize(img, img, cv::Size(), 1 /sw, 1 / sw);
            cv::copyMakeBorder(img, img, 0, inputH - img.rows, 0, 0, cv::BORDER_CONSTANT,cv::Scalar(0));
        }
        else {
            cv::resize(img, img, cv::Size(), 1 /sh, 1 / sh);
            cv::copyMakeBorder(img, img, 0, 0, 0, inputW - img.cols, cv::BORDER_CONSTANT,cv::Scalar(0));
        }
    }
    else {
        //直接补边到目标大小
        cv::copyMakeBorder(img, img, 0, inputH - img.rows, 0, inputW - img.cols, cv::BORDER_CONSTANT,cv::Scalar(0));
    }

    cv::Mat src = img.clone();

    //to float
    img.convertTo(img, CV_32FC3);

    //rgb
    cvtColor(img, img, CV_BGR2RGB);

    vector<Mat> input_channels;
    float* input_data = (float *)cpuBuffers;

    for (int i = 0; i < trtNet->getChannel(); ++i) {
        Mat channel(inputH, inputW, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += inputW * inputH;
    }

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the Mat
    * objects in input_channels. */
    split(img, input_channels);

    float *inputData = (float*)trtNet->getBuffer(0);
    cudaMemcpy(inputData, cpuBuffers, inputW * inputH * 3 * sizeof(float), cudaMemcpyHostToDevice);

    pre = (double)getTickCount() - pre;
    std::cout << "pre compute time :" << pre*1000.0 / cv::getTickFrequency() << " ms \n";

    //LOG(INFO) << "Start net_->Forward()";
    double t1 = (double)getTickCount();
    trtNet->doInference(1);
    t1 = (double)getTickCount() - t1;
    std::cout << "doInference compute time :" << t1*1000.0 / cv::getTickFrequency() << " ms \n";
    //LOG(INFO) << "Done net_->Forward()";

    double post = (double)getTickCount();
    string name_bbox = "face_rpn_bbox_pred_";
    string name_score ="face_rpn_cls_prob_reshape_";
    string name_landmark ="face_rpn_landmark_pred_";

    vector<FaceDetectInfo> faceInfo;
    for(size_t i = 0; i < _feat_stride_fpn.size(); i++) {
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);

        string str = name_score + key;
        TrtBlob* score_blob = trtNet->blob_by_name(str);
        std::vector<float> score = score_blob->result[0];
        std::vector<float>::iterator begin = score.begin() + score.size() / 2;
        std::vector<float>::iterator end = score.end();
        score = std::vector<float>(begin, end);

        str = name_bbox + key;
        TrtBlob* bbox_blob = trtNet->blob_by_name(str);
        std::vector<float> bbox_delta = bbox_blob->result[0];

        str = name_landmark + key;
        TrtBlob* landmark_blob = trtNet->blob_by_name(str);
        std::vector<float> landmark_delta = landmark_blob->result[0];

        int width = score_blob->outputDims.w();
        int height = score_blob->outputDims.h();
        size_t count = width * height;
        size_t num_anchor = _num_anchors[key];

        for(size_t num = 0; num < num_anchor; num++) {
            for(size_t j = 0; j < count; j++) {
                //置信度小于阈值跳过
                float conf = score[j + count * num];
                if(conf <= threshold) {
                    continue;
                }

                cv::Vec4f regress;
                float dx = bbox_delta[j + count * (0 + num * 4)];
                float dy = bbox_delta[j + count * (1 + num * 4)];
                float dw = bbox_delta[j + count * (2 + num * 4)];
                float dh = bbox_delta[j + count * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);

                //回归人脸框
                anchor_box rect = bbox_pred(_anchors[key][j + count * num], regress);
                //越界处理
                clip_boxes(rect, inputW, inputH);

                FacePts pts;
                for(size_t k = 0; k < 5; k++) {
                    pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                    pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                }
                //回归人脸关键点
                FacePts landmarks = landmark_pred(_anchors[key][j + count * num], pts);

                FaceDetectInfo tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                faceInfo.push_back(tmp);
            }
        }
    }
    //排序nms
    faceInfo = nms(faceInfo, 0.4);

    post = (double)getTickCount() - post;
    std::cout << "post compute time :" << post*1000.0 / cv::getTickFrequency() << " ms \n";

    for(size_t i = 0; i < faceInfo.size(); i++) {
        cv::Rect rect = cv::Rect(cv::Point2f(faceInfo[i].rect.x1, faceInfo[i].rect.y1),
                                 cv::Point2f(faceInfo[i].rect.x2, faceInfo[i].rect.y2));
        cv::rectangle(src, rect, Scalar(0, 0, 255), 2);

        for(size_t j = 0; j < 5; j++) {
            cv::Point2f pt = cv::Point2f(faceInfo[i].pts.x[j], faceInfo[i].pts.y[j]);
            cv::circle(src, pt, 1, Scalar(0, 255, 0), 2);
        }
    }

    imshow("dst", src);
    imwrite("trt_result.jpg", src);
    waitKey(0);
}

void RetinaFace::detectBatchImages(vector<cv::Mat> imgs, float threshold)
{
    //预处理
    int inputW = trtNet->getNetWidth();
    int inputH = trtNet->getNetHeight();
    double t2 = (double)getTickCount();
    vector<cv::Mat> src;
    for(size_t i = 0; i < imgs.size(); i++) {
        float sw = 1.0 * imgs[i].cols / inputW;
        float sh = 1.0 * imgs[i].rows / inputH;

        if(sw > 1.0 || sh > 1.0) {
            if(sw > sh) {
                cv::resize(imgs[i], imgs[i], cv::Size(), 1 /sw, 1 / sw);
                cv::copyMakeBorder(imgs[i], imgs[i], 0, inputH - imgs[i].rows, 0, 0, cv::BORDER_CONSTANT,cv::Scalar(0));
            }
            else {
                cv::resize(imgs[i], imgs[i], cv::Size(), 1 /sh, 1 / sh);
                cv::copyMakeBorder(imgs[i], imgs[i], 0, 0, 0, inputW - imgs[i].cols, cv::BORDER_CONSTANT,cv::Scalar(0));
            }
        }
        else {
            //直接补边到目标大小
            cv::copyMakeBorder(imgs[i], imgs[i], 0, inputH - imgs[i].rows, 0, inputW - imgs[i].cols, cv::BORDER_CONSTANT,cv::Scalar(0));
        }

        src.push_back(imgs[i]);
        //to float
        imgs[i].convertTo(imgs[i], CV_32FC3);

        //rgb
        cvtColor(imgs[i], imgs[i], CV_BGR2RGB);
    }
    t2 = (double)getTickCount() - t2;
    std::cout << "pre process compute time :" << t2*1000.0 / cv::getTickFrequency() << " ms \n";
    
    //填充数据
    vector<vector<Mat>> input_channels;
    float* input_data = (float *)cpuBuffers;
    for(size_t j = 0; j < imgs.size(); j++) {
        vector<Mat> input_chans;
        for (int i = 0; i < trtNet->getChannel(); ++i) {
            Mat channel(inputH, inputW, CV_32FC1, input_data);
            input_chans.push_back(channel);
            input_data += inputW * inputH;
        }
        input_channels.push_back(input_chans);
    }

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the Mat
    * objects in input_channels. */
    for(size_t j = 0; j < imgs.size(); j++) {
        split(imgs[j], input_channels[j]);
    }
    
    float *inputData = (float*)trtNet->getBuffer(0);
    cudaMemcpy(inputData, cpuBuffers, imgs.size() * inputW * inputH * 3 * sizeof(float), cudaMemcpyHostToDevice);

    //LOG(INFO) << "Start net_->Forward()";
    double t1 = (double)getTickCount();
    trtNet->doInference(imgs.size());
    t1 = (double)getTickCount() - t1;
    std::cout << "doInference compute time :" << t1*1000.0 / cv::getTickFrequency() << " ms \n";
    //LOG(INFO) << "Done net_->Forward()";

    double post = (double)getTickCount();
    string name_bbox = "face_rpn_bbox_pred_";
    string name_score ="face_rpn_cls_prob_reshape_";
    string name_landmark ="face_rpn_landmark_pred_";

    vector<vector<FaceDetectInfo>> faceInfos;
    for(size_t batch = 0; batch < imgs.size(); batch++) {
        vector<FaceDetectInfo> faceInfo;
        for(size_t i = 0; i < _feat_stride_fpn.size(); i++) {
            string key = "stride" + std::to_string(_feat_stride_fpn[i]);
            string str = name_score + key;
            TrtBlob* score_blob = trtNet->blob_by_name(str);
            std::vector<float> score = score_blob->result[batch];
            std::vector<float>::iterator begin = score.begin() + score.size() / 2;
            std::vector<float>::iterator end = score.end();
            score = std::vector<float>(begin, end);

            str = name_bbox + key;
            TrtBlob* bbox_blob = trtNet->blob_by_name(str);
            std::vector<float> bbox_delta = bbox_blob->result[batch];

            str = name_landmark + key;
            TrtBlob* landmark_blob = trtNet->blob_by_name(str);
            std::vector<float> landmark_delta = landmark_blob->result[batch];

            int width = score_blob->outputDims.w();
            int height = score_blob->outputDims.h();
            size_t count = width * height;
            size_t num_anchor = _num_anchors[key];

            for(size_t num = 0; num < num_anchor; num++) {
                for(size_t j = 0; j < count; j++) {
                    //置信度小于阈值跳过
                    float conf = score[j + count * num];
                    if(conf <= threshold) {
                        continue;
                    }

                    cv::Vec4f regress;
                    float dx = bbox_delta[j + count * (0 + num * 4)];
                    float dy = bbox_delta[j + count * (1 + num * 4)];
                    float dw = bbox_delta[j + count * (2 + num * 4)];
                    float dh = bbox_delta[j + count * (3 + num * 4)];
                    regress = cv::Vec4f(dx, dy, dw, dh);

                    //回归人脸框
                    anchor_box rect = bbox_pred(_anchors[key][j + count * num], regress);
                    //越界处理
                    clip_boxes(rect, inputW, inputH);

                    FacePts pts;
                    for(size_t k = 0; k < 5; k++) {
                        pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                        pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                    }
                    //回归人脸关键点
                    FacePts landmarks = landmark_pred(_anchors[key][j + count * num], pts);

                    FaceDetectInfo tmp;
                    tmp.score = conf;
                    tmp.rect = rect;
                    tmp.pts = landmarks;
                    faceInfo.push_back(tmp);
                }
            }  
        }

        faceInfos.push_back(faceInfo);
    }
    //排序nms
    for(size_t batch = 0; batch < imgs.size(); batch++){
        faceInfos[batch] = nms(faceInfos[batch], 0.4);
    }

    post = (double)getTickCount() - post;
    std::cout << "post compute time :" << post*1000.0 / cv::getTickFrequency() << " ms \n";

//    for(size_t batch = 0; batch < imgs.size(); batch++){
//        for(size_t i = 0; i < faceInfos[batch].size(); i++) {
//            cv::Rect rect = cv::Rect(cv::Point2f(faceInfos[batch][i].rect.x1, faceInfos[batch][i].rect.y1),
//                                     cv::Point2f(faceInfos[batch][i].rect.x2, faceInfos[batch][i].rect.y2));
//            cv::rectangle(src[batch], rect, Scalar(0, 0, 255), 2);

//            for(size_t j = 0; j < 5; j++) {
//                cv::Point2f pt = cv::Point2f(faceInfos[batch][i].pts.x[j], faceInfos[batch][i].pts.y[j]);
//                cv::circle(src[batch], pt, 1, Scalar(0, 255, 0), 2);
//            }
//        }

//        string win = "dst" + std::to_string(batch);
//        imshow(win, src[batch]);
//    }

//    //imwrite("trt_result.jpg", src);
//    waitKey(0);
}

#else
void RetinaFace::detect(Mat img, float threshold, float scales)
{
    if(img.empty()) {
        return;
    }

    double pre = (double)getTickCount();
    int ws = (img.cols + 31) / 32 * 32;
    int hs = (img.rows + 31) / 32 * 32;

    cv::copyMakeBorder(img, img, 0, hs - img.rows, 0, ws - img.cols, cv::BORDER_CONSTANT,cv::Scalar(0));

    cv::Mat src = img.clone();

    //to float
    img.convertTo(img, CV_32FC3);

    //rgb
    cvtColor(img, img, CV_BGR2RGB);

    Blob<float>* input_layer = Net_->input_blobs()[0];

    input_layer->Reshape(1, 3, img.rows, img.cols);
    Net_->Reshape();

    vector<Mat> input_channels;
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the Mat
    * objects in input_channels. */
    split(img, input_channels);

    pre = (double)getTickCount() - pre;
    std::cout << "pre compute time :" << pre*1000.0 / cv::getTickFrequency() << " ms \n";

    //LOG(INFO) << "Start net_->Forward()";
    double t1 = (double)getTickCount();
    Net_->Forward();
    t1 = (double)getTickCount() - t1;
    std::cout << "infer compute time :" << t1*1000.0 / cv::getTickFrequency() << " ms \n";
    //LOG(INFO) << "Done net_->Forward()";

    double post = (double)getTickCount();
    string name_bbox = "face_rpn_bbox_pred_";
    string name_score ="face_rpn_cls_prob_reshape_";
    string name_landmark ="face_rpn_landmark_pred_";

    vector<FaceDetectInfo> faceInfo;
    for(size_t i = 0; i < _feat_stride_fpn.size(); i++) {
///////////////////////////////////////////////
        double s1 = (double)getTickCount();
///////////////////////////////////////////////
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        int stride = _feat_stride_fpn[i];

        string str = name_score + key;
        const boost::shared_ptr<Blob<float>> score_blob = Net_->blob_by_name(str);
        const float* scoreB = score_blob->cpu_data() + score_blob->count() / 2;
        const float* scoreE = scoreB + score_blob->count() / 2;
        std::vector<float> score = std::vector<float>(scoreB, scoreE);

        str = name_bbox + key;
        const boost::shared_ptr<Blob<float>> bbox_blob = Net_->blob_by_name(str);
        const float* bboxB = bbox_blob->cpu_data();
        const float* bboxE = bboxB + bbox_blob->count();
        std::vector<float> bbox_delta = std::vector<float>(bboxB, bboxE);

        str = name_landmark + key;
        const boost::shared_ptr<Blob<float>> landmark_blob = Net_->blob_by_name(str);
        const float* landmarkB = landmark_blob->cpu_data();
        const float* landmarkE = landmarkB + landmark_blob->count();
        std::vector<float> landmark_delta = std::vector<float>(landmarkB, landmarkE);

        int width = score_blob->width();
        int height = score_blob->height();
        size_t count = width * height;
        size_t num_anchor = _num_anchors[key];

///////////////////////////////////////////////
        s1 = (double)getTickCount() - s1;
        std::cout << "s1 compute time :" << s1*1000.0 / cv::getTickFrequency() << " ms \n";
///////////////////////////////////////////////

        for(size_t num = 0; num < num_anchor; num++) {
            for(size_t j = 0; j < count; j++) {
                //置信度小于阈值跳过
                float conf = score[j + count * num];
                if(conf <= threshold) {
                    continue;
                }

                cv::Vec4f regress;
                float dx = bbox_delta[j + count * (0 + num * 4)];
                float dy = bbox_delta[j + count * (1 + num * 4)];
                float dw = bbox_delta[j + count * (2 + num * 4)];
                float dh = bbox_delta[j + count * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);

                //存储顺序 h * w * num_anchor
                vector<anchor_box> anchors = anchors_plane(height, width, stride, _anchors_fpn[key]);
                //回归人脸框
                anchor_box rect = bbox_pred(anchors[j + count * num], regress);
                //越界处理
                clip_boxes(rect, ws, hs);

                FacePts pts;
                for(size_t k = 0; k < 5; k++) {
                    pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                    pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                }
                //回归人脸关键点
                FacePts landmarks = landmark_pred(anchors[j + count * num], pts);

                FaceDetectInfo tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                faceInfo.push_back(tmp);
            }
        }
    }

    //排序nms
    faceInfo = nms(faceInfo, 0.4);

    post = (double)getTickCount() - post;
    std::cout << "post compute time :" << post*1000.0 / cv::getTickFrequency() << " ms \n";


    for(size_t i = 0; i < faceInfo.size(); i++) {
        cv::Rect rect = cv::Rect(cv::Point2f(faceInfo[i].rect.x1, faceInfo[i].rect.y1), cv::Point2f(faceInfo[i].rect.x2, faceInfo[i].rect.y2));
        cv::rectangle(src, rect, Scalar(0, 0, 255), 2);

        for(size_t j = 0; j < 5; j++) {
            cv::Point2f pt = cv::Point2f(faceInfo[i].pts.x[j], faceInfo[i].pts.y[j]);
            cv::circle(src, pt, 1, Scalar(0, 255, 0), 2);
        }
    }

    imshow("dst", src);
    waitKey(0);
}
#endif
