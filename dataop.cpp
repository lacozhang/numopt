#pragma warning(disable : 4996)

#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "dataop.h"
#include "util.h"

namespace {

	static const char *libsvmseps = "\t ";
	static char TempLineBuffer[UINT16_MAX] = { '\0' };

	void matrix_size_estimation_from_text(std::ifstream& featsrc, std::vector<size_t>& rowsize,
		int &row, int &col){

		row = col = 0;

		std::vector<std::pair<size_t, double>> feats;
		int label = 0;

		featsrc.getline(TempLineBuffer, sizeof(TempLineBuffer));
		while (featsrc.good()) {
			++row;
			parselibsvmline(TempLineBuffer, feats, label, true);
			// active feature for sample row
			rowsize.push_back(feats.size() + 1);

			for (std::pair<size_t, double> &item : feats) {
				if (col < item.first) {
					col = item.first;
				}
			}

			// get next line from file
			featsrc.getline(TempLineBuffer, sizeof(TempLineBuffer));
		}
	}


	void matrix_size_estimation_from_bin(std::ifstream& featsrc, std::vector<size_t>& rowsize,
		int &row, int &col){
		row = col = 0;

		int label = 0, n = 0;
		featsrc.read((char*)&label, sizeof(int));

		while (featsrc.good())
		{
			featsrc.read((char*)&n, sizeof(int));
			rowsize.push_back(n+1);

			while (n > 0)
			{
				size_t index = 0;
				double val = 0;

				featsrc.read((char*)&index, sizeof(size_t));
				featsrc.read((char*)&val, sizeof(double));

				if (col < index){
					col = index;
				}

				--n;
			}
			++row;

			featsrc.read((char*)&label, sizeof(int));
		}
	}

	void load_libsvm_data_text(std::ifstream& ifs, boost::shared_ptr<DataSamples> samples, 
		boost::shared_ptr<ClsLabelVector> labels, int featsize){
		
		std::vector<std::pair<size_t, double>> featline;

		ifs.getline(TempLineBuffer, sizeof(TempLineBuffer));
		int nrow = 0;
		while (ifs.good()) {

			int label;
			parselibsvmline(TempLineBuffer, featline, label, true);
			labels->coeffRef(nrow) = label;

			for (std::pair<size_t, double> &item : featline) {
				if (item.first < featsize) {
					samples->insert(nrow, item.first) = item.second;
				}
			}
			++nrow;
			ifs.getline(TempLineBuffer, sizeof(TempLineBuffer));
		}
	}

	void load_libsvm_data_bin(std::ifstream& ifs, boost::shared_ptr<DataSamples>& samples, 
		boost::shared_ptr<ClsLabelVector>& labels, int featsize) {

		int rowindex = 0;

		int label = 0, n = std::numeric_limits<int>::infinity();
		ifs.read(reinterpret_cast<char*>(&label), sizeof(int));

		while (ifs.good())
		{
			ifs.read(reinterpret_cast<char*>(&n), sizeof(int));

			while (n>0)
			{

				size_t index = 0;
				double val = 0;

				ifs.read(reinterpret_cast<char*>(&index), sizeof(size_t));
				ifs.read(reinterpret_cast<char*>(&val), sizeof(double));

				if (index < featsize) {
					samples->coeffRef(rowindex, index) = val;
				}
				--n;
			}

			++rowindex;
			ifs.read(reinterpret_cast<char*>(&label), sizeof(int));
		}
		std::cout << "Load Total " << rowindex << " samples" << std::endl;
	}

}

void parselibsvmline(char *line, std::vector<std::pair<size_t, double>> &feats,
	int &label, bool parse) {
	feats.clear();

	char *ptr = strtok(line, libsvmseps);
	if (!ptr) {
		std::cerr << "Error, string abnormal" << line << std::endl;
		std::exit(-1);
	}
	label = std::atoi(ptr);

	ptr = strtok(NULL, ": \t");
	while (ptr != nullptr) {
		size_t index = std::atoi(ptr);
		double val = 0;

		ptr = strtok(NULL, ": \t");
		if (ptr != nullptr) {
			val = std::atof(ptr);
			ptr = strtok(NULL, ": \t");
			feats.push_back(std::pair<size_t, double>(index, val));
		}
		else {
			std::cerr << "error, data format error" << std::endl;
			std::exit(-1);
		}
	}
}


void matrix_size_estimation(std::string featfile, Eigen::VectorXi &datsize,
                            int &row, int &col) {

	bool filebinary = false;
	if (boost::algorithm::ends_with(featfile, ".bin")){
		filebinary = true;
	}

	std::ifstream featsrc;
	if (!filebinary){
		featsrc.open(featfile.c_str(), std::ios_base::in);
	}
	else {
		featsrc.open(featfile.c_str(), std::ios_base::in | std::ios_base::binary);
	}

  if (!featsrc.is_open()) {
    std::cerr << "open file " << featfile << " failed" << std::endl;
    std::abort();
  }

  timeutil t;
  t.tic();

  std::vector<size_t> rowsize;
  if (!filebinary){
	  matrix_size_estimation_from_text(featsrc, rowsize, row, col);
  }
  else {
	  matrix_size_estimation_from_bin(featsrc, rowsize, row, col);
  }

  col += 1;

  std::cout << "data size estimation costs " << t.toc() << std::endl;
  std::cout << "row size " << row << " colsize " << col << std::endl;

  datsize.resize(row);
  for (int i = 0; i < rowsize.size(); ++i) {
    datsize(i) = rowsize[i];
  }

}

void load_libsvm_data(
	std::string featfile,
	boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor>> &Samples,
	boost::shared_ptr<Eigen::VectorXi> &labels, bool estimate, int colsize) {

	// estimate the data size for loading
	Eigen::VectorXi datasize;
	std::vector<std::pair<size_t, double>> featline;

	bool binary = boost::algorithm::ends_with(featfile, ".bin");

	int estrowsize, estcolsize;
	matrix_size_estimation(featfile, datasize, estrowsize, estcolsize);

	std::cout << "finish the size estimation" << std::endl;

	timeutil t;
	t.tic();

	if (estimate)
		colsize = estcolsize;

	std::cout << "rowsize " << estrowsize << " colsize " << colsize << std::endl;
	Samples.reset(new DataSamples(estrowsize, colsize));

	if (Samples.get() == NULL) {
		std::cerr << "Error, new operator for samples error" << std::endl;
		std::exit(-1);
	}

	Samples->reserve(datasize);

	labels.reset(new Eigen::VectorXi(estrowsize));

	std::ifstream featsrc;
	if (binary) {
		featsrc.open(featfile, std::ios_base::in | std::ios_base::binary);
	}
	else {
		featsrc.open(featfile, std::ios_base::in);
	}

	if (!featsrc.is_open()) {
		std::cerr << "open file " << featfile << " failed" << std::endl;
		return;
	}

	if (binary) {		
		load_libsvm_data_bin(featsrc, Samples, labels, colsize);
	}
	else {
		load_libsvm_data_text(featsrc, Samples, labels, colsize);
	}

	Samples->makeCompressed();
	std::cout << "Loading data costs " << t.toc() << " seconds " << std::endl;
}