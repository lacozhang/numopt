#pragma once

#ifndef __UTIL_H__
#define __UTIL_H__
#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
#include <chrono>

class timeutil {
public:
	timeutil();
	void tic();
	double toc();

private:
	std::chrono::time_point<std::chrono::system_clock> counter_;
};

boost::program_options::variables_map ParseArgs(int argc, const char* argv[],
	boost::program_options::options_description optionsdesc, bool allowunreg);
#endif
