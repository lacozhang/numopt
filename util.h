#pragma once

#ifndef __UTIL_H__
#define __UTIL_H__
#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
#include <chrono>
#include <fstream>

class timeutil {
public:
	timeutil();
	void tic();
	double toc();

private:
	std::chrono::time_point<std::chrono::system_clock> counter_;
};

class BinaryFileHandler {
public:
	BinaryFileHandler(std::fstream& sink) : sink_(sink) {}
	~BinaryFileHandler() {
		if (sink_.is_open()) {
			sink_.close();
		}
	}

	bool WriteInt(int val) {
		return WriteSimpleType<int>(sink_, val);
	}

	bool ReadInt(int& val) {
		return ReadSimpleType<int>(sink_, val);
	}

	bool WriteSizeT(size_t val) {
		return WriteSimpleType<size_t>(sink_, val);
	}

	bool ReadSizeT(size_t& val) {
		return ReadSimpleType<size_t>(sink_, val);
	}


private:

	template <class Type>
	bool WriteSimpleType(std::fstream& sink, Type val) {
		if (sink.good()) {
			sink.write((const char*)&val, sizeof(Type));
		}
		return sink.good();
	}

	template <class Type>
	bool ReadSimpleType(std::fstream& src, Type& val) {
		if (src.good()) {
			src.read((char*)&val, sizeof(Type));
		}
		return src.good();
	}

	std::fstream& sink_;
};

boost::program_options::variables_map ParseArgs(int argc, const char* argv[],
	boost::program_options::options_description optionsdesc, bool allowunreg);
#endif
