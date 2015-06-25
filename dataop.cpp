#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include "dataop.h"
#include "util.h"

void splitstring(std::string& line, const char* delim, std::vector<std::string>& strs){
	char* ptr = new char[line.length() + 1];
	memset(ptr, 0, sizeof(char)*(line.length() + 1));
	line.copy(ptr, line.length());

	char* tok = std::strtok(ptr, delim);
	while (tok != NULL){
		strs.push_back(tok);
		tok = std::strtok(NULL, delim);
	}
	delete[] ptr;
}


void matrix_size_estimation(std::string featfile, 
			    Eigen::VectorXi& datsize,
			    int& row, int& col){
	std::ifstream featsrc(featfile.c_str());
	std::vector<int> rowsize;
	row = 0;
	col = 0;
	
	if( ! featsrc.is_open() ){
		std::cerr << "open file " << featfile << " failed" << std::endl;
		std::abort();
	}

	timeutil t;
	t.tic();
	std::string line;
	std::getline(featsrc, line);

	while( featsrc.good() ){
		++row;

		std::vector<std::string> strs;
		splitstring(line, "\t ", strs);
		// get the larest valu index
		std::vector<std::string> featval;
		splitstring(strs[strs.size() - 1], ":", featval);
		if (featval.size() != 2){
			std::cerr << "Error, feat value pair not correct " << strs.at(strs.size()-1)
				<< std::endl;
		}
		else {
			int featidx = std::atoi(featval.at(0).c_str());
			if (featidx > col){
				col = featidx;
			}
		}
		// active feature for sample row
		rowsize.push_back(strs.size()+1);

		// get next line from file
		std::getline(featsrc, line);
	}
	std::cout << "data size estimation costs " << t.toc() << std::endl;
	datsize.resize(row);
	for(int i=0; i < rowsize.size(); ++i){
		datsize(i) = rowsize[i];
	}
	col += 1;
}

void load_libsvm_data(std::string featfile, 
	       boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor> >& Samples,
	       boost::shared_ptr<Eigen::VectorXi>& labels){

	// estimate the data size for loading
	Eigen::VectorXi datasize;
	int rowsize, colsize;
	matrix_size_estimation(featfile, datasize, rowsize, colsize);

	std::cout << "finish the size estimation" << std::endl;

	timeutil t;
	t.tic();
	// set data size and space
	Samples.reset( new Eigen::SparseMatrix<double, Eigen::RowMajor>(rowsize, colsize) );
	if( Samples.get() == NULL ){
		std::cerr << "Error, new operator for samples error" << std::endl;
		std::exit(-1);
	}
	Samples->reserve(datasize);

	labels.reset( new Eigen::VectorXi(rowsize) );

	std::string line;
	std::ifstream ifs(featfile.c_str());
	if(! ifs.is_open() ){
		std::cerr << "open file " << featfile << " failed" << std::endl;
		std::abort();
	}

	std::getline(ifs, line);
	int nrow = 0;
	while( ifs.good() ){
		
		char* buf = new char[line.length() + 1];
		buf[line.length()] = 0;
		line.copy(buf, line.length());

		char* ptr = strtok(buf, "\t ");
		if (!ptr){
			std::cerr << "Error, string abnormal" << ptr << std::endl;
			std::exit(-1);
		}
		labels->coeffRef(nrow) = std::atoi(ptr);

		std::vector<std::string> feats;
		ptr = strtok(NULL, "\t ");
		while (ptr != NULL){

			feats.push_back(ptr);
			ptr = strtok(NULL, "\t ");
		}
		delete[] buf;

		for (int i = 0; i < feats.size(); ++i){

			char* buf2 = new char[feats[i].length() + 1];
			memset(buf2, 0, sizeof(char)*(feats[i].length() + 1));
			int idx = 0;
			double val = 0;

			feats[i].copy(buf2, feats[i].length());

			char* ptr2 = strtok(buf2, ":");
			if (!ptr2){
				std::cerr << "line format error" << line << std::endl;
				std::exit(-1);
			}
			else {
				idx = std::atoi(ptr2);
			}

			ptr2 = strtok(NULL, ":");
			if (!ptr2){
				std::cerr << "line format error " << line << std::endl;
				std::exit(-1);
			}
			else {
				val = std::atof(ptr2);
			}

			delete[] buf2;
			Samples->insert(nrow, idx) = val;
		}
		++nrow;
		std::getline(ifs, line);
	}
	Samples->makeCompressed();
	std::cout << "loading data costs " << t.toc() << " seconds " << std::endl;
}
