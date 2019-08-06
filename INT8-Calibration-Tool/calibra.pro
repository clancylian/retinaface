TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
QT += core
QT -= gui

HEADERS += \
    calibrationtable.h \
    common/aescrypto.h \
    common/helpercuda.h \
    common/logger.h \
    common/logging.h \
    CalibrationTableImpl.h

SOURCES += \
    calibrationtable.cpp \
    common/aescrypto.cpp \
    common/logger.cpp \
    CalibrationTableImpl.cpp \
    main.cpp

CUDA_SOURCES += \
    common/calcprelu.cu

INCLUDEPATH += ./common/

LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs

INCLUDEPATH += /home/ubuntu/caffe-office/caffe/include

LIBS += -lprotobuf -L/home/ubuntu/caffe-office/caffe/build/lib -lcaffe

# 3rd party
LIBS += -lglog -lboost_system -lssl -lcrypto

INCLUDEPATH += /usr/local/TensorRT/include
LIBS += -L/usr/local/TensorRT/lib -lnvinfer -lnvcaffe_parser\

unix {
    CUDA_DIR = /usr/local/cuda
    SYSTEM_TYPE = 64            #操作系统位数 '32' or '64',
    CUDA_ARCH = sm_61         # cuda架构, for example 'compute_10', 'compute_11', 'sm_10'
    NVCC_OPTIONS = -lineinfo -Xcompiler -fPIC

    INCLUDEPATH += $$CUDA_DIR/include
    LIBS += -L$$CUDA_DIR/lib64
    LIBS += -lcuda -lcudart -lcublas -lcudnn -lcurand
    # npp
    LIBS += -lnppig -lnppicc -lnppc -lnppidei -lnppist

    CUDA_OBJECTS_DIR = ./

    # The following makes sure all path names (which often include spaces) are put between quotation marks
    CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

    CONFIG(debug, debug|release): {
        cuda_d.input = CUDA_SOURCES
        cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE \
            -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda_d.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda_d
    } else:CONFIG(release, debug|release): {
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE \
            -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda
    }
}
