/**
 * @file       dbg.h
 * @brief      调试信息打印管理 API
 * @details    调试信息打印管理 API
 * @author     susongjian@gmail.com
 * @date       2017.08.10
 * @version    V0.2
 * @par Copyright (C):
 *			   罗普特(厦门)科技集团有限公司
 * @par History:
 *  - V0.2     xiaochengliu.prc@foxmail.com     2017.08.10 \n
 *             功能添加 \n
 *  - V0.1     susongjian@gmail.com		        2017.07.01 \n
 *             原型开发
 */

#ifndef __DBG_H__
#define __DBG_H__

#if WIN32
#define snprintf _snprintf
#endif

#include <stdio.h>  
#include <stdarg.h>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#define checkCudaErrors( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);


enum DbgLevel {
    DbgLevelVerbose,
    DbgLevelInfo,
    DbgLevelWarn,
    DbgLevelError,
    DbgLevelNone
};

void dbg(DbgLevel level, const char *file, int line, const char *fmt, ...);

#define dbgVerbose(fmt, ...)    dbg(DbgLevelVerbose, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define dbgInfo(fmt, ...)       dbg(DbgLevelInfo, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define dbgWarn(fmt, ...)       dbg(DbgLevelWarn, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define dbgError(fmt, ...)      dbg(DbgLevelError, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

void imdbgShow(std::string name, cv::Mat img);
void imdbgWrite(std::string name, cv::Mat img);

void drawPoints(cv::Mat &image, std::vector<cv::Point2f> points, cv::Scalar color = cv::Scalar::all(255));
void drawBox(cv::Mat &image, cv::Rect box, cv::Scalar color = cv::Scalar::all(255), int thick = 1);

#endif // DBG_H
