#include "util.h"

timeutil::timeutil(){
}

void timeutil::tic(){
	t_ = clock();
}

double timeutil::toc(){
	return ((double)t_)/CLOCKS_PER_SEC;
}
