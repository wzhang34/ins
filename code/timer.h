#ifndef TIMER_H_INCLUDED
#define TIMER_H_INCLUDED

#include <sys/time.h>

struct Timer
{
    struct timeval _start;
    struct timeval _end;

    Timer(){}

    ~Timer(){}

    void start()
    {
        gettimeofday(&_start, NULL);
    }

    double elapsed()
    {
        gettimeofday(&_end, NULL);
        long seconds  = _end.tv_sec  - _start.tv_sec;
        long useconds = _end.tv_usec - _start.tv_usec;
        long d = ((seconds) * 1000 + useconds/1000.0) + 0.5;

        printf(" %.2f sec. ", d/1000.0);
        return d;
    }
};

#endif // TIMER_H_INCLUDED
