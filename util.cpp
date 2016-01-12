#include "util.h"

timeutil::timeutil(){
}

void timeutil::tic(){
	t_ = clock();
}

double timeutil::toc(){
	clock_t endTime = clock();
	return ((double)(endTime - t_))/CLOCKS_PER_SEC;
}
