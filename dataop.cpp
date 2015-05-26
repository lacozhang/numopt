#include <iostream>
#include <vector>
#include <fstream>
#include <boost/regex.hpp>
#include "dataop.h"
#include "util.h"

boost::shared_ptr<boost::regex> tokenizer( new boost::regex("\\s+"));
boost::shared_ptr<boost::regex> featvalpair(new boost::regex("^(\\d+):([\\d.+-]+)$"));

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
		int cnt = 0;

		boost::sregex_token_iterator iter(line.begin(), line.end(),
			*tokenizer, -1);
		boost::sregex_token_iterator end;
		while( iter != end ){
			++cnt;
			
			std::string strfeatpair = iter->str();
			// the fist part is label part
			if( cnt > 1 ){
				boost::smatch m;
				if( boost::regex_match(strfeatpair, m, *featvalpair) ){
					int featIdx = std::atoi(m[1].str().c_str());
					double featVal = std::atof(m[2].str().c_str());
					if( col < featIdx ){
						col = featIdx;
					}
				} else {
					std::cerr << "feat value format error in row " << row 
						<< std::endl;
				}
			}

			++iter;
		}

		// active feature for sample row
		rowsize.push_back(cnt+1);

		// get next line from file
		std::getline(featsrc, line);
	}

	std::cout << "data size estimation costs " << t.toc() << std::endl;
	datsize.resize(row);
	for(int i=0; i < rowsize.size(); ++i){
		datsize(i) = rowsize[i];
	}
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
	std::cout << "sample reserve success" << std::endl;

	labels.reset( new Eigen::VectorXi(rowsize) );
	std::cout << "label reserve success" << std::endl;

	std::string line;
	std::ifstream ifs(featfile.c_str());
	if(! ifs.is_open() ){
		std::cerr << "open file " << featfile << " failed" << std::endl;
		std::abort();
	}

	std::getline(ifs, line);
	int nrow = 0;
	while( ifs.good() ){
		
		int cnt = 0;
		boost::sregex_token_iterator iter(line.begin(), line.end(),
			*tokenizer, -1);
		boost::sregex_token_iterator end;

		while( iter != end ){
			++cnt;
			std::string featval = iter->str();

			if( cnt == 1 ){
				labels->coeffRef(nrow) = std::atoi(featval.c_str());
			} else {
				boost::smatch m;
				if( boost::regex_match(featval, m, *featvalpair) ){
					int featIdx = std::atoi(m[1].str().c_str())-1;
					double featVal = std::atof(m[2].str().c_str());
					Samples->insert(nrow, featIdx) = featVal;
				} else {
					std::cerr << "error, can not match feat-value pair " << rowsize << featval 
						<< std::endl;
				}
			}
			iter++;
		}

		++nrow;
		std::getline(ifs, line);
	}
	Samples->makeCompressed();
	std::cout << "loading data costs " << t.toc() << " seconds " << std::endl;
}
