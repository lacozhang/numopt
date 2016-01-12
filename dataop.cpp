#pragma warning(disable : 4996)

#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include "dataop.h"
#include "util.h"

namespace {

static const char *libsvmseps = "\t ";
static char TempLineBuffer[UINT16_MAX] = {'\0'};

void parselibsvmline(char *line, std::vector<std::pair<size_t, size_t>> &feats,
                     int &label, bool parse = true) {
  std::vector<std::string> featstrs;

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
    size_t val = 0;

    ptr = strtok(NULL, ": \t");
    if (ptr != nullptr) {
      val = std::atoi(ptr);
      ptr = strtok(NULL, ": \t");
      feats.push_back(std::pair<size_t, size_t>(index, val));
    } else {
      std::cerr << "error, data format error" << std::endl;
      std::exit(-1);
    }
  }
}
}

void matrix_size_estimation(std::string featfile, Eigen::VectorXi &datsize,
                            int &row, int &col) {
  std::ifstream featsrc(featfile.c_str());
  std::vector<std::pair<size_t, size_t>> feats;
  std::vector<size_t> rowsize;
  int label;
  row = 0;
  col = 0;

  if (!featsrc.is_open()) {
    std::cerr << "open file " << featfile << " failed" << std::endl;
    std::abort();
  }

  timeutil t;
  t.tic();

  featsrc.getline(TempLineBuffer, sizeof(TempLineBuffer));

  while (featsrc.good()) {
    ++row;
    parselibsvmline(TempLineBuffer, feats, label, true);
    // active feature for sample row
    rowsize.push_back(feats.size() + 1);

    for (std::pair<size_t, size_t> &item : feats) {
      if (col < item.first) {
        col = item.first;
      }
    }

    // get next line from file
    featsrc.getline(TempLineBuffer, sizeof(TempLineBuffer));
  }
  std::cout << "data size estimation costs " << t.toc() << std::endl;
  datsize.resize(row);
  for (int i = 0; i < rowsize.size(); ++i) {
    datsize(i) = rowsize[i];
  }
  col += 1;
}

void load_libsvm_data(
    std::string featfile,
    boost::shared_ptr<Eigen::SparseMatrix<double, Eigen::RowMajor>> &Samples,
    boost::shared_ptr<Eigen::VectorXi> &labels, bool estimate, int colsize) {

  // estimate the data size for loading
  Eigen::VectorXi datasize;
  std::vector<std::pair<size_t, size_t>> featline;

  int estrowsize, estcolsize;
  matrix_size_estimation(featfile, datasize, estrowsize, estcolsize);

  std::cout << "finish the size estimation" << std::endl;

  timeutil t;
  t.tic();

  if (estimate)
    colsize = estcolsize;

  Samples.reset(new DataSamples(estrowsize, colsize));

  if (Samples.get() == NULL) {
    std::cerr << "Error, new operator for samples error" << std::endl;
    std::exit(-1);
  }

  Samples->reserve(datasize);

  labels.reset(new Eigen::VectorXi(estrowsize));

  std::ifstream ifs(featfile.c_str());
  if (!ifs.is_open()) {
    std::cerr << "open file " << featfile << " failed" << std::endl;
    std::abort();
  }

  ifs.getline(TempLineBuffer, sizeof(TempLineBuffer));
  int nrow = 0;
  while (ifs.good()) {

    int label;
    parselibsvmline(TempLineBuffer, featline, label, true);
    labels->coeffRef(nrow) = label;

    for (std::pair<size_t, size_t> &item : featline) {
      if (item.first <= colsize) {
        Samples->insert(nrow, item.first) = item.second;
      } else {
        std::cerr << "warning line " << item.first
                  << " has unsupported features" << std::endl;
      }
    }
    ++nrow;
    ifs.getline(TempLineBuffer, sizeof(TempLineBuffer));
  }
  Samples->makeCompressed();
  std::cout << "loading data costs " << t.toc() << " seconds " << std::endl;
}

void estimate_binary_datasize(std::string featfile, int& row, int& col){
  row = col = 0;

  std::ifstream src(featfile.c_str());
}



void load_binary_data(std::string featfile,
  boost::shared_ptr<DataSamples> &samples,
  boost::shared_ptr<ClsLabelVector> &labels, bool estimate,
  int colsize){

  std::ifstream src(featfile.c_str());
  if (!src.is_open()){
    std::cerr << "open file " << featfile << " failed" << std::endl;
    std::exit(1);
  }


}