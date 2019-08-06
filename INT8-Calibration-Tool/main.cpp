#include "CalibrationTableImpl.h"
#include <string>

using namespace std;

int main(int argc, char** argv)
{
    CalibrationTableRetinaFace *calibra = new CalibrationTableRetinaFace();
    calibra->setEncryption(false);
    calibra->setInputDir("../INT8-Calibration-Tool/dataSet");
    calibra->setOutputDir("../batches");
    calibra->setModelPath("../model/mnet-deconv-0517.prototxt",
                          "../model/mnet-deconv-0517.caffemodel");

    calibra->doCalibration();

    delete calibra;

    return 0;
}
