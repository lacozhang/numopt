#ifndef __UTIL_H__
#define __UTIL_H__
#include <chrono>

class timeutil {
public:
	timeutil();
	void tic();
	double toc();

private:
	std::chrono::time_point<std::chrono::system_clock> counter_;
};

#endif
