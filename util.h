#ifndef __UTIL_H__
#define __UTIL_H__
#include <ctime>

class timeutil {
public:
	timeutil();
	void tic();
	double toc();

private:
	clock_t t_;
};

#endif
