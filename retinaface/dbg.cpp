/**
 * @file       dbg.cpp
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

#include <chrono>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "dbg.h"

using namespace std::chrono;

static const int LogBufSize = 256;

DbgLevel gDbgLevel = DbgLevelInfo;

bool gImageDbgShowOn = false;
bool gImageDbgWriteOn = false;
bool gDrawDbgOn = false;
bool gRecordLog = true;

static char gLogBuf[LogBufSize];
static std::mutex gLogMutex;
static std::ofstream gRecordFile;

void dbg(DbgLevel level, const char *file, int line, const char *fmt, ...)
{
    if (level >= gDbgLevel) {
        gLogMutex.lock();

        int len = snprintf(gLogBuf, LogBufSize, "<%s, %d>: ", file, line);
        va_list ap;
        va_start(ap, fmt);
        len += vsnprintf(gLogBuf + len, LogBufSize - len, fmt, ap);
        va_end(ap);

        printf("%s", gLogBuf);
        if (gRecordLog) {
            if (!gRecordFile.is_open()) {
                time_t time = system_clock::to_time_t(system_clock::now());
                std::tm *now = std::gmtime(&time);
                std::stringstream ss;
                ss << "AlgorithmLog_" << now->tm_year + 1900
                   << std::setw(2) << std::setfill('0') << now->tm_mon + 1
                   << std::setw(2) << std::setfill('0') << now->tm_mday
                   << std::setw(2) << std::setfill('0') << (now->tm_hour + 8)%24
                   << std::setw(2) << std::setfill('0') << now->tm_min
                   << std::setw(2) << std::setfill('0') << now->tm_sec << ".txt";
                gRecordFile.open(ss.str());
            }

            gRecordFile.write(gLogBuf, len);
            gRecordFile.flush();
        }

        gLogMutex.unlock();
    }
}

void imdbgShow(std::string name, cv::Mat img)
{
    if (gImageDbgShowOn) {
		cv::imshow(name, img);
	}
}

void imdbgWrite(std::string name, cv::Mat img)
{
	if (gImageDbgWriteOn) {
		cv::imwrite(name, img);
	}
}

void drawPoints(cv::Mat &image, std::vector<cv::Point2f> points, cv::Scalar color)
{
	if (gDrawDbgOn) {
		for (std::vector<cv::Point2f>::const_iterator i = points.begin(), ie = points.end(); i != ie; ++i) {
			circle(image, *i, 1, color, 1);
		}
	}
}

void drawBox(cv::Mat &image, cv::Rect box, cv::Scalar color, int thick)
{
	if (gDrawDbgOn) {
		rectangle(image, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), color, thick);
	}
}
