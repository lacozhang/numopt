#include "util.h"

timeutil::timeutil(){
}

void timeutil::tic(){
	counter_ = std::chrono::system_clock::now();
}

double timeutil::toc(){
	std::chrono::duration<double> t = std::chrono::system_clock::now() - counter_;
	return t.count();
}
