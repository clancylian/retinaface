/**
 * @file       timer.h
 * @brief      基于 C++11 实现的计时器
 * @details    基于 C++11 实现的计时器
 * @author     susongjian@gmail.com
 * @date       2017.08.23
 * @version    V0.1
 * @par Copyright (C):
 *			   罗普特(厦门)科技集团有限公司
 * @par History:
 *  - V0.1     susongjian@gmail.com         2017.08.23 \n
 *             原型开发 \n
 */

#ifndef TIMER_H
#define TIMER_H

#include <chrono>

namespace RK {

using namespace std;
using namespace std::chrono;

class Timer
{
public:
    Timer() : begin(high_resolution_clock::now()) {}

    void reset()
    {
        begin = high_resolution_clock::now();
    }


    // 默认输出秒
    double elapsed() const
    {
        return duration_cast<duration<double>>(high_resolution_clock::now() - begin).count();
    }

    // 毫秒
    int64_t elapsedMilliSeconds() const
    {
        return duration_cast<chrono::milliseconds>(high_resolution_clock::now() - begin).count();
    }

    // 微秒
    int64_t elapsedMicroSeconds() const
    {
        return duration_cast<chrono::microseconds>(high_resolution_clock::now() - begin).count();
    }

    // 纳秒
    int64_t elapsedNanoSeconds() const
    {
        return duration_cast<chrono::nanoseconds>(high_resolution_clock::now() - begin).count();
    }

    // 秒
    int64_t elapsedSeconds() const
    {
        return duration_cast<chrono::seconds>(high_resolution_clock::now() - begin).count();
    }

    // 分钟
    int64_t elapsedMinutes() const
    {
        return duration_cast<chrono::minutes>(high_resolution_clock::now() - begin).count();
    }

    // 小时
    int64_t elapsedHours() const
    {
        return duration_cast<chrono::hours>(high_resolution_clock::now() - begin).count();
    }

private:
    time_point<high_resolution_clock> begin;
};

} //namespace RK

#endif // TIMER_H
