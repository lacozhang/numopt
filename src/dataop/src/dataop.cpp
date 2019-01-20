#pragma warning(disable : 4996)

#include "dataop/dataop.h"
#include "util/producer_consumer.h"
#include "util/util.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>
#include <vector>

namespace {

static const char *kBinMagicString = "binv2";
static const char *libsvmseps = "\t ";
static char TempLineBuffer[UINT16_MAX] = {'\0'};
} // namespace

void matrix_size_estimation_from_text(std::ifstream &featsrc,
                                      std::vector<size_t> &rowsize, int &row,
                                      int &col) {

  row = col = 0;

  std::vector<std::pair<size_t, float>> feats;
  int label = 0;

  featsrc.getline(TempLineBuffer, sizeof(TempLineBuffer));
  while (featsrc.good()) {
    ++row;
    parselibsvmline(TempLineBuffer, feats, label, true);
    // active feature for sample row
    rowsize.push_back(feats.size() + 1);

    for (std::pair<size_t, float> &item : feats) {
      if (col < item.first) {
        col = item.first;
      }
    }

    // get next line from file
    featsrc.getline(TempLineBuffer, sizeof(TempLineBuffer));
  }
}

bool matrix_size_estimation_from_bin(std::ifstream &ifs,
                                     std::vector<size_t> &rowsize, int &row,
                                     int &col) {
  row = col = 0;

  bool success = true;
  size_t featindex = 0, samplecnt = 0, featcnt = 0, elementcnt = 0;
  float featval = 0;
  size_t temp = 0;
  signed char label = 0;

  // read magic string
  std::memset(TempLineBuffer, 0, sizeof(TempLineBuffer));
  ifs.read(TempLineBuffer, std::strlen(kBinMagicString));
  success = success && ifs.good();
  if (!success) {
    LOG(ERROR) << "read magic string failed";
    return false;
  }
  if (std::strncmp(TempLineBuffer, kBinMagicString,
                   std::strlen(kBinMagicString)) != 0) {
    LOG(ERROR) << "magic string do not equal";
    return false;
  }

  // read sample count
  ifs.read((char *)&samplecnt, sizeof(size_t));
  success = success && ifs.good();
  if (!success) {
    LOG(ERROR) << "read sample count failed";
    return false;
  }
  row = samplecnt;
  rowsize.resize(row);

  // read maximum feature id
  ifs.read((char *)&featcnt, sizeof(size_t));
  success = success && ifs.good();
  if (!success) {
    LOG(ERROR) << "read maximum feature id failed";
    return false;
  }
  col = featcnt;

  // read non-zero elements in data file
  ifs.read((char *)&elementcnt, sizeof(size_t));
  success = success && ifs.good();
  if (!success) {
    LOG(ERROR) << "read number of non-zero elements failed";
    return false;
  }

  // read number of non-zero elements in each row.
  for (int rowindex = 0; rowindex < row; ++rowindex) {
    ifs.read((char *)&temp, sizeof(size_t));
    success = success && ifs.good();
    if (!success) {
      LOG(ERROR) << "read non-zero elements for each row failed";
      return false;
    }
    rowsize[rowindex] = temp;
  }
  return true;
}

bool load_libsvm_data_text(std::ifstream &ifs,
                           boost::shared_ptr<DataSamples> samples,
                           boost::shared_ptr<LabelVector> labels,
                           int featsize) {

  std::vector<std::pair<size_t, float>> featline;

  ifs.getline(TempLineBuffer, sizeof(TempLineBuffer));
  int nrow = 0;
  while (ifs.good()) {

    int label;
    parselibsvmline(TempLineBuffer, featline, label, true);
    labels->coeffRef(nrow) = label;

    for (std::pair<size_t, float> &item : featline) {
      if (item.first < featsize) {
        samples->insert(nrow, item.first) = item.second;
      }
    }
    ++nrow;
    ifs.getline(TempLineBuffer, sizeof(TempLineBuffer));
  }
  return true;
}

bool load_libsvm_data_bin(std::ifstream &ifs,
                          boost::shared_ptr<DataSamples> &samples,
                          boost::shared_ptr<LabelVector> &labels,
                          int featsize) {

  bool success = true;
  int rowindex = 0;
  size_t featindex = 0, samplecnt = 0, featcnt = 0, elementcnt = 0;
  float featval = 0;
  size_t temp = 0;
  signed char label = 0;

  // read magic string
  std::memset(TempLineBuffer, 0, sizeof(TempLineBuffer));
  ifs.read(TempLineBuffer, std::strlen(kBinMagicString));
  success = success && ifs.good();
  if (!success) {
    LOG(ERROR) << "read magic string failed";
    return false;
  }
  if (std::strncmp(TempLineBuffer, kBinMagicString,
                   std::strlen(kBinMagicString)) != 0) {
    LOG(ERROR) << "magic string do not equal";
    return false;
  }

  // read sample count
  ifs.read((char *)&samplecnt, sizeof(size_t));
  success = success && ifs.good();
  if (!success) {
    LOG(ERROR) << "read sample count failed";
    return false;
  }

  // read maximum feature id
  ifs.read((char *)&featcnt, sizeof(size_t));
  success = success && ifs.good();
  if (!success) {
    LOG(ERROR) << "read maximum feature id failed";
    return false;
  }

  if (featcnt > samples->cols()) {
    LOG(INFO)
        << "maximum feature id of file: " << featcnt
        << "maximum feature id of data matrix: " << samples->cols();
  }

  // read non-zero elements in data file
  ifs.read((char *)&elementcnt, sizeof(size_t));
  success = success && ifs.good();
  if (!success) {
    LOG(ERROR) << "read number of non-zero elements failed";
    return false;
  }

  // read number of non-zero elements in each row.
  for (int rowindex = 0; rowindex < samplecnt; ++rowindex) {
    ifs.read((char *)&temp, sizeof(size_t));
    success = success && ifs.good();
    if (!success) {
      LOG(ERROR) << "read non-zero elements for each row failed";
      return false;
    }
  }

  // read real data
  for (rowindex = 0; rowindex < samplecnt; ++rowindex) {
    ifs.read((char *)&temp, sizeof(size_t));
    success = success && ifs.good();
    if (!success) {
      LOG(ERROR) << "read non-zero elements of line failed";
      return false;
    }

    ifs.read((char *)&label, sizeof(signed char));
    success = success && ifs.good();
    if (!success) {
      LOG(ERROR) << "read label of line failed";
      return false;
    }

    labels->coeffRef(rowindex) = label;

    for (int i = 0; i < temp; ++i) {

      ifs.read((char *)&featindex, sizeof(size_t));
      success = success && ifs.good();
      if (!success) {
        LOG(ERROR) << "read index of feat failed";
        return false;
      }

      ifs.read((char *)&featval, sizeof(float));
      success = success && ifs.good();
      if (!success) {
        LOG(ERROR) << "read value of feat failed";
        return false;
      }

      if (featindex <= samples->cols()) {
        samples->coeffRef(rowindex, featindex) = featval;
      }
    }
  }
  std::cout << "Load Total " << rowindex << " samples" << std::endl;
  return true;
}

bool save_libsvm_data_bin(std::string filepath,
                          boost::shared_ptr<DataSamples> &samples,
                          boost::shared_ptr<LabelVector> &labels) {

  if (!boost::algorithm::ends_with(filepath, ".bin")) {
    LOG(INFO) << "output file do not ends with .bin";
  }

  std::ofstream ofs(filepath, std::ios_base::out | std::ios_base::trunc |
                                  std::ios_base::binary);
  if (!ofs.is_open()) {
    LOG(FATAL) << "create file " << filepath << " failed";
    return false;
  }

  size_t samplecnt = samples->rows(), featurecnt = samples->cols() - 1,
         elementcnt = samples->nonZeros();
  bool success = true;

  // write magic string
  ofs.write(kBinMagicString, std::strlen(kBinMagicString));
  success = success && ofs.good();
  if (!success) {
    LOG(ERROR) << "write magic number failed";
    return false;
  }

  // write number of samples in this set
  ofs.write((char *)&samplecnt, sizeof(size_t));
  success = success && ofs.good();
  if (!success) {
    LOG(ERROR) << "write sample count failed";
    return false;
  }

  // write maximum feature id in this set
  ofs.write((const char *)&featurecnt, sizeof(size_t));
  success = success && ofs.good();
  if (!success) {
    LOG(ERROR) << "write maximum feature id failed";
    return false;
  }

  // write number of non-zero elements in this set
  ofs.write((const char *)&elementcnt, sizeof(size_t));
  success = success && ofs.good();

  // write number of non-zero elements in each row
  for (int rowindex = 0; rowindex < samples->outerSize(); ++rowindex) {
    size_t cnt = 0;
    for (DataSamples::InnerIterator it(*samples, rowindex); it; ++it, ++cnt)
      ;

    ofs.write((char *)&cnt, sizeof(size_t));
    success = success && ofs.good();
    if (!success) {
      LOG(ERROR) << "write nonzero elements of one row failed";
      return false;
    }
  }

  // write data labels into file
  LOG(INFO) << "write data to file";
  signed char label = 0;
  size_t index = 0;
  float val = 0;

  for (int rowindex = 0; rowindex < samples->outerSize(); ++rowindex) {

    size_t cnt = 0;
    for (DataSamples::InnerIterator it(*samples, rowindex); it; ++it, ++cnt)
      ;

    ofs.write((char *)&cnt, sizeof(size_t));
    success = success && ofs.good();
    if (!success) {
      LOG(ERROR) << "write element count failed";
      return false;
    }

    label = labels->coeff(rowindex);
    ofs.write((char *)&label, sizeof(signed char));
    success = success && ofs.good();
    if (!success) {
      LOG(ERROR) << "write line " << rowindex << " label failed";
      return false;
    }

    if (!success) {
      LOG(ERROR) << "write label failed";
      return false;
    }

    for (DataSamples::InnerIterator it(*samples, rowindex); it; ++it) {
      index = it.col();
      val = it.value();

      ofs.write((char *)&index, sizeof(size_t));
      success = success && ofs.good();

      if (!success) {
        LOG(FATAL)
            << "write index for line " << rowindex << " failed";
        return false;
      }

      ofs.write((char *)&val, sizeof(float));
      success = success && ofs.good();
      if (!success) {
        LOG(FATAL)
            << "write value for index line " << rowindex << " failed";
        return false;
      }
    }
  }
  ofs.close();
  LOG(INFO) << "sample count  : " << samplecnt;
  LOG(INFO) << "max feature id: " << featurecnt;
  return true;
}

void parselibsvmline(char *line, std::vector<std::pair<size_t, float>> &feats,
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
    float val = 0;

    ptr = strtok(NULL, ": \t");
    if (ptr != nullptr) {
      val = std::atof(ptr);
      ptr = strtok(NULL, ": \t");
      feats.push_back(std::pair<size_t, float>(index, val));
    } else {
      LOG(ERROR) << "error, data format error" << std::endl;
      std::abort();
    }
  }
}

bool matrix_size_estimation(std::string featfile, Eigen::VectorXi &datsize,
                            int &row, int &col) {

  bool filebinary = false;
  if (boost::algorithm::ends_with(featfile, ".bin")) {
    filebinary = true;
  }

  std::ifstream featsrc;
  if (!filebinary) {
    featsrc.open(featfile.c_str(), std::ios_base::in);
  } else {
    featsrc.open(featfile.c_str(), std::ios_base::in | std::ios_base::binary);
  }

  if (!featsrc.is_open()) {
    LOG(INFO)
        << "open file " << featfile << " failed" << std::endl;
    std::abort();
  }

  timeutil t;
  t.tic();

  std::vector<size_t> rowsize;
  if (!filebinary) {
    matrix_size_estimation_from_text(featsrc, rowsize, row, col);
  } else {
    if (!matrix_size_estimation_from_bin(featsrc, rowsize, row, col)) {
      LOG(FATAL) << "estimate matrix size failed " << featfile;
      return false;
    }
  }

  featsrc.close();
  LOG(INFO) << "estimate costs " << t.toc() << " seconds";

  col += 1;

  datsize.resize(row);
  for (int i = 0; i < rowsize.size(); ++i) {
    datsize(i) = rowsize[i];
  }
  return true;
}

bool load_libsvm_data(std::string featfile,
                      boost::shared_ptr<DataSamples> &Samples,
                      boost::shared_ptr<LabelVector> &labels, bool estimate,
                      int colsize) {

  // estimate the data size for loading
  Eigen::VectorXi datasize;
  std::vector<std::pair<size_t, double>> featline;

  bool binary = boost::algorithm::ends_with(featfile, ".bin");
  if (binary) {
    LOG(INFO) << "Loading data from binary file";
  } else {
    LOG(INFO) << "Loading data from text file";
  }

  int estrowsize, estcolsize;
  if (!matrix_size_estimation(featfile, datasize, estrowsize, estcolsize)) {
    LOG(FATAL) << "load data file " << featfile << " failed";
    return false;
  }

  LOG(INFO) << "Finish data size estimation";

  timeutil t;

  if (estimate)
    colsize = estcolsize;

  LOG(INFO) << "Sample count  :" << estrowsize;
  LOG(INFO) << "Max feature id:" << (colsize - 1);
  Samples.reset(new DataSamples(estrowsize, colsize));

  if (Samples.get() == NULL) {
    LOG(ERROR)
        << "Error, new operator for samples error" << std::endl;
    std::exit(-1);
  }

  Samples->reserve(datasize);

  labels.reset(new LabelVector(estrowsize));

  std::ifstream featsrc;
  if (binary) {
    featsrc.open(featfile, std::ios_base::in | std::ios_base::binary);
  } else {
    featsrc.open(featfile, std::ios_base::in);
  }

  if (!featsrc.is_open()) {
    LOG(ERROR) << "open file " << featfile << " failed";
    return false;
  }

  bool ret = true;
  t.tic();
  if (binary) {
    ret = load_libsvm_data_bin(featsrc, Samples, labels, colsize);
  } else {
    ret = load_libsvm_data_text(featsrc, Samples, labels, colsize);
  }

  Samples->makeCompressed();
  LOG(INFO) << "Loading data costs " << t.toc() << " seconds "
                          << std::endl;
  LOG(INFO) << "data samples : " << Samples->rows() << std::endl;
  LOG(INFO) << "max feat id  : " << (Samples->cols() - 1)
                          << std::endl;
  return ret;
}

bool load_lccrf_data_bin(std::string srcfilepath,
                         boost::shared_ptr<LccrfSamples> &samples,
                         boost::shared_ptr<LccrfLabels> &labels, bool estimate,
                         size_t maxunifeatid, size_t maxbifeatid,
                         size_t maxlabelid) {

  bool binary = false;
  binary = boost::algorithm::ends_with(srcfilepath, ".bin");

  if (!binary) {
    LOG(ERROR) << "For lccrf, binary file only";
    return false;
  }

  std::vector<std::vector<int>> currunifeats, currbifeats;
  std::vector<int> currlabels, currunifeatcnts, currbifeatcnts;

  std::fstream src(srcfilepath, std::ios_base::in | std::ios_base::binary);
  if (!src.is_open()) {
    LOG(ERROR) << "Failed to open file " << srcfilepath;
    return false;
  }

  BinaryFileHandler reader(src);
  int numsamples = 0;
  size_t wordcount;
  int tmp;

  if (!reader.ReadInt(tmp)) {
    LOG(ERROR) << "Error when read max unigram feature id";
    return false;
  }
  if (estimate) {
    maxunifeatid = tmp;
  }

  if (!reader.ReadInt(tmp)) {
    LOG(ERROR) << "Error when read max bigram feature id";
    return false;
  }
  if (estimate) {
    maxbifeatid = tmp;
  }

  if (!reader.ReadInt(tmp)) {
    LOG(ERROR) << "Error when read max label id";
    return false;
  }
  if (estimate) {
    maxlabelid = tmp;
  }

  if (!reader.ReadInt(numsamples)) {
    LOG(ERROR) << "Error when read number of samples";
    return false;
  }

  LOG(INFO) << "Number of Samples      : " << numsamples;
  LOG(INFO) << "Max Unigram Feature Id : " << maxunifeatid;
  LOG(INFO) << "Max Bigram Feature Id  : " << maxbifeatid;
  LOG(INFO) << "Max Label Id           : " << maxlabelid;

  samples.reset(new LccrfSamples());
  labels.reset(new LccrfLabels());

  samples->SetMaxUnigramFeatureId(maxunifeatid);
  samples->SetMaxBigramFeatureId(maxbifeatid);
  labels->SetMaxLabelId(maxlabelid);

  samples->UnigramFeature().resize(numsamples);
  samples->BigramFeature().resize(numsamples);
  labels->Labels().resize(numsamples);

  for (int sampleidx = 0; sampleidx < numsamples; ++sampleidx) {

    reader.ReadSizeT(wordcount);
    currunifeats.resize(wordcount);
    currbifeats.resize(wordcount);
    currlabels.resize(wordcount);
    currunifeatcnts.resize(wordcount);
    currbifeatcnts.resize(wordcount);

    for (size_t pos = 0; pos < wordcount; ++pos) {
      reader.ReadInt(currlabels[pos]);
      reader.ReadInt(currunifeatcnts[pos]);
      int featid;
      for (int idx = 0; idx < currunifeatcnts[pos]; ++idx) {
        reader.ReadInt(featid);
        currunifeats[pos].push_back(featid);
      }

      reader.ReadInt(currbifeatcnts[pos]);
      for (int idx = 0; idx < currbifeatcnts[pos]; ++idx) {
        reader.ReadInt(featid);
        currbifeats[pos].push_back(featid);
      }

      std::sort(currunifeats[pos].begin(), currunifeats[pos].end());
      std::sort(currbifeats[pos].begin(), currbifeats[pos].end());
    }

    samples->UnigramFeature()[sampleidx].reset(
        new DataSamples(wordcount, maxunifeatid + 1));
    samples->BigramFeature()[sampleidx].reset(
        new DataSamples(wordcount, maxbifeatid + 1));
    labels->Labels()[sampleidx].reset(new LabelVector(wordcount));

    samples->UnigramFeature()[sampleidx]->reserve(currunifeatcnts);
    samples->BigramFeature()[sampleidx]->reserve(currbifeatcnts);

    for (size_t pos = 0; pos < wordcount; ++pos) {

      for (int i = 0; i < currunifeatcnts[pos]; ++i) {
        samples->UnigramFeature()[sampleidx]->insert(pos,
                                                     currunifeats[pos][i]) = 1;
      }

      for (int i = 0; i < currbifeatcnts[pos]; ++i) {
        samples->BigramFeature()[sampleidx]->insert(pos, currbifeats[pos][i]) =
            1;
      }

      labels->Labels()[sampleidx]->coeffRef(pos) = currlabels[pos];
    }

    currunifeats.clear();
    currbifeats.clear();
    currunifeatcnts.clear();
    currbifeatcnts.clear();
    currlabels.clear();
  }
  return true;
}

bool build_vocab(const std::string &filepath, size_t cutoffvalue,
                 boost::shared_ptr<Vocabulary> &words,
                 boost::shared_ptr<Vocabulary> &labels) {

  namespace fs = boost::filesystem;
  fs::path basepath(filepath);
  if (!basepath.is_absolute())
    basepath = fs::canonical(basepath);

  std::ofstream query, label;
  std::ifstream src(filepath);

  std::string queryfilepath = (basepath.parent_path() / "query.txt").string();
  std::string labelfilepath = (basepath.parent_path() / "labels.txt").string();
  query.open(queryfilepath);
  label.open(labelfilepath);

  if (!query.is_open()) {
    LOG(ERROR) << "Failed to open query.txt" << std::endl;
    return false;
  }
  if (!label.is_open()) {
    LOG(ERROR) << "Failed to open labels.txt" << std::endl;
    return false;
  }

  if (!src.is_open()) {
    LOG(ERROR) << "Failed to open " << filepath << std::endl;
    return false;
  }

  src.getline(TempLineBuffer, sizeof(TempLineBuffer));
  char *ptr = nullptr;
  std::vector<std::string> segments;
  int linecount = 0;
  while (src.good()) {
    ++linecount;
    segments.clear();
    Util::Split((const unsigned char *)TempLineBuffer,
                std::strlen(TempLineBuffer), segments,
                (const unsigned char *)"\t", false);
    if (segments.size() < 2 || segments[0].empty() || segments[1].empty()) {
      LOG(ERROR) << "Line format error @ " << linecount;
      LOG(ERROR) << TempLineBuffer;
    } else {
      query.write(segments[0].c_str(), segments[0].size());
      query.write("\n", 1);
      label.write(segments[1].c_str(), segments[1].size());
      label.write("\n", 1);
    }
    src.getline(TempLineBuffer, sizeof(TempLineBuffer));
  }

  if (!src.eof()) {
    LOG(ERROR) << "Unexpected error";
    return false;
  }

  src.close();
  query.close();
  label.close();

  words = Vocabulary::BuildVocabForQuery(queryfilepath, cutoffvalue);
  labels = Vocabulary::BuildVocabForLabel(labelfilepath);

  if (!words || !labels) {
    LOG(WARNING)
        << "Failed to build vocabulary for query & labels";
    return false;
  }

  return true;
}

bool parse_nn_sequence_dense(const std::string &textfeats,
                             std::vector<std::string> &buffer,
                             std::vector<float> &feats) {
  if (textfeats.empty())
    return false;
  buffer.clear();
  feats.clear();
  Util::Split(textfeats, buffer, ",", false);
  for (auto &textfeat : buffer) {
    feats.push_back(std::stof(textfeat));
  }
  return true;
}

bool parse_nn_sequence_sparse_binary(const std::string &textfeats,
                                     std::vector<std::string> &buffer,
                                     std::vector<int> &feats) {
  if (textfeats.empty())
    return false;
  buffer.clear();
  feats.clear();
  Util::Split(textfeats, buffer, ",", false);
  if (buffer.empty())
    return false;
  for (auto &textfeat : buffer) {
    if (textfeat.empty())
      return false;
    feats.push_back(std::stoi(textfeat));
  }
  return true;
}

bool parse_nn_sequence_sparse_float(const std::string &textfeats,
                                    std::vector<std::string> &buffer,
                                    std::vector<std::pair<int, float>> &feats) {
  if (textfeats.empty())
    return false;
  buffer.clear();
  Util::Split(textfeats, buffer, ",", false);
  std::vector<std::string> kvbuffer;
  for (auto &kv : buffer) {
    kvbuffer.clear();
    Util::Split(kv, kvbuffer, ":", false);
    if (kvbuffer.size() != 2) {
      return false;
    }
    feats.push_back(
        std::make_pair(std::stoi(kvbuffer[0]), std::stof(kvbuffer[1])));
  }

  return true;
}

bool read_lines(std::ifstream &src, std::vector<std::string> &lines,
                int count) {
  lines.clear();
  std::string line;
  if (!src.good())
    return false;

  std::getline(src, line);
  while (src.good() && (count > 0)) {
    if (!line.empty()) {
      lines.push_back(std::move(line));
      --count;
    }
    std::getline(src, line);
  }

  LOG(INFO) << lines.size() << " lines read";
  if (src.good())
    return true;
  else
    return false;
}

bool read_sentence(std::ifstream &src, std::vector<std::string> &sentence) {
  sentence.clear();
  if (!src.good())
    return false;
  std::string line;
  std::getline(src, line);
  while (src.good() && line.empty())
    std::getline(src, line);

  while (src.good() && !line.empty()) {
    sentence.push_back(std::move(line));
    std::getline(src, line);
  }

  if (src.good())
    return true;
  else
    return false;
}

void estimate_nn_sequence_lines(std::vector<std::string> &lines,
                                NNModel::NNSequenceEstimate &est) {
  if (lines.empty())
    return;
  std::vector<std::string> features, buffer;
  std::vector<int> sparsebinaryfeats;
  std::vector<std::pair<int, float>> sparsefloatfeats;
  std::vector<float> densefeats;
  for (auto &line : lines) {
    features.clear();
    Util::Split((const unsigned char *)line.c_str(), line.size(), features,
                (const unsigned char *)"\t", false);
    if (features.size() > 1) {

      int labelid = std::stoi(features[0]);
      if (labelid > est.labelsize_)
        est.labelsize_ = labelid;

      if (features.size() >= 2 && !features[1].empty() &&
          parse_nn_sequence_sparse_binary(features[1], buffer,
                                          sparsebinaryfeats)) {
        for (auto &n : sparsebinaryfeats)
          if (n > est.sparsebinarysize_)
            est.sparsebinarysize_ = n;
      }

      if (features.size() >= 3 && !features[2].empty() &&
          parse_nn_sequence_sparse_float(features[2], buffer,
                                         sparsefloatfeats)) {
        for (auto &kv : sparsefloatfeats)
          if (kv.first > est.sparsefloatsize_)
            est.sparsefloatsize_ = kv.first;
      }

      if (features.size() >= 4 && !features[3].empty() &&
          parse_nn_sequence_dense(features[3], buffer, densefeats)) {
        if (est.densesize_ == 0)
          est.densesize_ = densefeats.size();
        else if (est.densesize_ != densefeats.size()) {
          LOG(ERROR) << "Line format error " << line;
        }
      }
    }
  }
}

bool estimate_nn_sequence(const std::string &filepath, int &sparsebinary,
                          int &sparsefloat, int &dense, int &label) {
  std::ifstream src(filepath);
  if (!src.is_open()) {
    LOG(ERROR) << "Failed to open " << filepath;
    return false;
  }
  sparsebinary = sparsefloat = dense = label = 0;

  timeutil timer;
  timer.tic();
  NNModel::NNSequenceEstimate est;
  std::mutex updatemut;
  int threads = std::thread::hardware_concurrency() - 2;
  if (threads <= 0)
    threads = 1;
  LOG(INFO) << "Number of Threads : " << threads;
  mltools::ProducerConsumer<std::vector<std::string>> worker(1000, threads);

  std::function<bool(std::vector<std::string> &)> producer =
      [&src](std::vector<std::string> &lines) {
        return read_lines(src, lines, 50000);
      };
  worker.StartProducer(producer);

  std::function<void(std::vector<std::string> &, NNModel::NNSequenceEstimate &)>
      consumer = [&](std::vector<std::string> &lines,
                     NNModel::NNSequenceEstimate &localest) {
        estimate_nn_sequence_lines(lines, localest);
      };

  std::function<void(NNModel::NNSequenceEstimate &)> updater =
      [&updatemut, &est](NNModel::NNSequenceEstimate &localest) {
        std::lock_guard<std::mutex> lk(updatemut);
        est.sparsebinarysize_ =
            std::max(est.sparsebinarysize_, localest.sparsebinarysize_);
        est.sparsefloatsize_ =
            std::max(est.sparsefloatsize_, localest.sparsefloatsize_);
        est.densesize_ = std::max(est.densesize_, localest.densesize_);
        est.labelsize_ = std::max(est.labelsize_, localest.labelsize_);
      };
  worker.StartConsumer<NNModel::NNSequenceEstimate>(consumer, updater);
  worker.BlockProducer();
  worker.BlockConsumer();
  worker.JoinConsumer();
  LOG(INFO) << "Estimate costs " << timer.toc() << " seconds";

  sparsebinary = est.sparsebinarysize_;
  sparsefloat = est.sparsefloatsize_;
  dense = est.densesize_;
  label = est.labelsize_;

  sparsebinary++;
  sparsefloat++;
  label++;
  if (!src.eof()) {
    LOG(ERROR) << "Unexpected EOF";
    return false;
  }
  return true;
}

bool parse_nn_sequence_sentence(
    std::vector<std::string> &sentence,
    boost::shared_ptr<NNModel::SentenceFeature> &feat,
    boost::shared_ptr<NNModel::SentenceLabel> &label,
    const int sparsebinarysize, const int sparsefloatsize,
    const int densesize) {

  if (sentence.empty())
    return false;
  if (!feat)
    feat = boost::make_shared<NNModel::SentenceFeature>();
  if (!label)
    label = boost::make_shared<NNModel::SentenceLabel>();

  std::vector<std::string> buffer;

  std::vector<int> binaryfeat;
  std::vector<std::pair<int, float>> floatfeat;
  std::vector<float> densefeat;

  std::vector<std::vector<int>> binaryfeats;
  std::vector<std::vector<std::pair<int, float>>> floatfeats;

  int seqlen = sentence.size();

  feat->SparseBinaryFeature().resize(seqlen, sparsebinarysize);
  if (sparsefloatsize > 0)
    feat->SparseFeature().resize(seqlen, sparsefloatsize);
  if (densesize > 0)
    feat->DenseFeature().resize(seqlen, densesize);
  label->GetLabels().resize(seqlen);

  std::string binarytext, floattext, densetext;
  binaryfeats.resize(seqlen);
  floatfeats.resize(seqlen);

  for (int idx = 0; idx < seqlen; ++idx) {
    buffer.clear();
    Util::Split(sentence[idx], buffer, "\t", false);
    if (buffer.size() < 2)
      return false;
    binarytext = std::move(buffer[1]);
    if (sparsefloatsize > 0 && buffer.size() >= 3)
      floattext = std::move(buffer[2]);
    if (densesize > 0 && buffer.size() >= 4)
      densetext = std::move(buffer[3]);

    label->SetLabel(idx, std::stoi(buffer[0]));
    binaryfeat.clear();
    if (parse_nn_sequence_sparse_binary(binarytext, buffer, binaryfeat)) {
      binaryfeats[idx] = std::move(binaryfeat);
    } else {
      LOG(FATAL) << "Fundamental feature missing";
      return false;
    }

    floatfeat.clear();
    if (sparsefloatsize > 0 && buffer.size() >= 3 &&
        parse_nn_sequence_sparse_float(floattext, buffer, floatfeat)) {
      floatfeats[idx] = std::move(floatfeat);
    }

    densefeat.clear();
    if (densesize > 0 && buffer.size() >= 4 &&
        parse_nn_sequence_dense(densetext, buffer, densefeat)) {
      if (densefeat.size() != densesize) {
        LOG(WARNING) << "dense feature size do not match";
      }
      for (int j = 0; j < densesize; ++j)
        feat->DenseFeature().coeffRef(idx, j) = densefeat[j];
    }
  }

  std::vector<int> estsize;
  estsize.resize(seqlen);

  for (int idx = 0; idx < seqlen; ++idx)
    estsize[idx] = binaryfeats[idx].size();
  feat->SparseBinaryFeature().reserve(estsize);
  for (int idx = 0; idx < seqlen; ++idx) {
    for (auto &featidx : binaryfeats[idx]) {
      feat->SparseBinaryFeature().coeffRef(idx, featidx) = 1.0;
    }
  }
  feat->SparseBinaryFeature().makeCompressed();

  if (sparsefloatsize > 0) {
    for (int idx = 0; idx < seqlen; ++idx)
      estsize[idx] = floatfeats[idx].size();
    feat->SparseFeature().reserve(estsize);
    for (int idx = 0; idx < seqlen; ++idx) {
      for (auto &kv : floatfeats[idx]) {
        feat->SparseFeature().coeffRef(idx, kv.first) = kv.second;
      }
    }
    feat->SparseFeature().makeCompressed();
  }
  return true;
}

bool load_nn_sequence(const std::string &filepath,
                      boost::shared_ptr<NNModel::NNSequenceFeature> &feats,
                      boost::shared_ptr<NNModel::NNSequenceLabel> &labels) {
  std::ifstream src(filepath);
  if (!src.is_open()) {
    LOG(ERROR) << "Error to open file " << filepath;
    return false;
  }

  timeutil timer;
  timer.tic();
  int spbinarysize = feats->GetSparseBinarySize(),
      spfloatsize = feats->GetSparseFloatSize();
  int densesize = feats->GetDenseSize();
  std::mutex datamut;
  int threads = std::thread::hardware_concurrency() - 2;
  if (threads <= 0)
    threads = 1;
  mltools::ProducerConsumer<std::vector<std::vector<std::string>>> worker(
      1000, threads);
  int maxsentences = 10000;

  std::function<bool(std::vector<std::vector<std::string>> &)> producer =
      [&src, maxsentences](std::vector<std::vector<std::string>> &sentence) {
        bool succ = true;
        std::vector<std::string> buffer;
        sentence.clear();
        for (int i = 0; i < maxsentences && succ; ++i) {
          succ = read_sentence(src, buffer);
          sentence.push_back(std::move(buffer));
        }
        return succ;
      };

  worker.StartProducer(producer);

  std::function<void(std::vector<std::vector<std::string>> &,
                     NNModel::NNSequenceParse &)>
      processor = [&](std::vector<std::vector<std::string>> &sentences,
                      NNModel::NNSequenceParse &res) {
        boost::shared_ptr<NNModel::SentenceFeature> feat;
        boost::shared_ptr<NNModel::SentenceLabel> label;
        res.feats_.clear();
        res.labels_.clear();
        for (auto &sentence : sentences) {
          if (parse_nn_sequence_sentence(sentence, feat, label, spbinarysize,
                                         spfloatsize, densesize)) {
            res.feats_.push_back(std::move(feat));
            res.labels_.push_back(std::move(label));
          }
        }
      };

  std::function<void(NNModel::NNSequenceParse &)> updater =
      [&datamut, &feats, &labels](NNModel::NNSequenceParse &res) {
        std::lock_guard<std::mutex> lk(datamut);
        int size = res.feats_.size();
        for (int i = 0; i < size; ++i) {
          if (!res.feats_[i] || !res.labels_[i])
            continue;
          feats->AppendSequenceFeature(res.feats_[i]);
          labels->AppendSequenceLabel(res.labels_[i]);
        }
      };

  worker.StartConsumer<NNModel::NNSequenceParse>(processor, updater);
  worker.BlockProducer();
  worker.BlockConsumer();
  worker.JoinConsumer();

  LOG(INFO) << "Loading costs " << timer.toc() << " seconds";
  if (!src.eof()) {
    LOG(ERROR) << "Unexpected EOF";
    return false;
  }

  return true;
}

template <> bool DataLoader<kLibSVM, DataSamples, LabelVector>::LoadData() {
  if (filepath_.empty()) {
    valid_ = false;
  } else {
    valid_ = load_libsvm_data(filepath_, features_, labels_, !specifyfeatdim_,
                              maxfeatid_ + 1);
    maxfeatid_ = features_->cols() - 1;
  }
  return valid_;
}

template <> bool DataLoader<kLCCRF, LccrfSamples, LccrfLabels>::LoadData() {
  if (filepath_.empty()) {
    valid_ = false;
  } else {
    valid_ =
        load_lccrf_data_bin(filepath_, features_, labels_, !specifyfeatdim_,
                            maxunifeatid_, maxbifeatid_, maxlabelid_);

    maxunifeatid_ = features_->GetMaxUnigramFeatureId();
    maxbifeatid_ = features_->GetMaxBigramFeatureId();
    maxlabelid_ = labels_->GetMaxLabelId();
  }
  return valid_;
}

template <>
bool DataLoader<kNNSequence, NNModel::NNSequenceFeature,
                NNModel::NNSequenceLabel>::LoadData() {
  namespace fs = boost::filesystem;
  fs::path path(filepath_);
  if (!fs::exists(path) || !fs::is_regular_file(path)) {
    LOG(ERROR) << "file path " << filepath_ << " bad";
    valid_ = false;
  } else {
    if (features_.get() == nullptr)
      features_ = boost::make_shared<NNModel::NNSequenceFeature>();
    if (labels_.get() == nullptr)
      labels_ = boost::make_shared<NNModel::NNSequenceLabel>();

    if (!specifyfeatdim_) {
      int spbinarysize = 0, spfloatsize = 0, densesize = 0, labelsize = 0;
      if (estimate_nn_sequence(filepath_, spbinarysize, spfloatsize, densesize,
                               labelsize)) {
        features_->SetSparseBinarySize(spbinarysize);
        features_->SetSparseFloatSize(spfloatsize);
        features_->SetDenseSize(densesize);
        labels_->SetLabelSize(labelsize);
      } else {
        LOG(ERROR) << "Parse nn sequence format error";
        return false;
      }
    }

    valid_ = load_nn_sequence(filepath_, features_, labels_);
  }
  return valid_;
}

template <>
bool DataLoader<kNNQuery, NNModel::NNQueryFeature,
                NNModel::NNQueryLabel>::LoadData() {
  namespace fs = boost::filesystem;
  fs::path path(filepath_);
  if (filepath_.empty() || !fs::exists(path) || !fs::is_regular_file(path)) {
    valid_ = false;
    LOG(INFO)
        << "either empty path or invaida path " << filepath_ << std::endl;
    return false;
  } else {

    if (features_.get() == nullptr)
      features_ = boost::make_shared<NNModel::NNQueryFeature>();

    if (labels_.get() == nullptr)
      labels_ = boost::make_shared<NNModel::NNQueryLabel>();

    if (!specifyfeatdim_) {
      boost::shared_ptr<Vocabulary> words, labels;
      if (!build_vocab(filepath_, cutoff_, words, labels)) {
        std::abort();
      }
      features_->SetVocabulary(words);
      labels_->SetVocabulary(labels);
    }

    NNModel::NNQueryFeaturizer featurizer(features_->GetVocabulary(),
                                          labels_->GetVocabulary());
    valid_ = featurizer.Featurize(features_, labels_, filepath_);
    return valid_;
  }
}

template <>
void DataLoader<kLibSVM, DataSamples, LabelVector>::SetModelMetaInfo(
    const boost::shared_ptr<DataLoader<kLibSVM, DataSamples, LabelVector>>
        &infosrc) {
  specifyfeatdim_ = true;
  maxfeatid_ = infosrc->maxfeatid_;
}

template <>
void DataLoader<kLCCRF, LccrfSamples, LccrfLabels>::SetModelMetaInfo(
    const boost::shared_ptr<DataLoader<kLCCRF, LccrfSamples, LccrfLabels>>
        &infosrc) {
  specifyfeatdim_ = true;
  maxunifeatid_ = infosrc->maxunifeatid_;
  maxbifeatid_ = infosrc->maxbifeatid_;
  maxlabelid_ = infosrc->maxlabelid_;
}

template <>
void DataLoader<kNNQuery, NNModel::NNQueryFeature, NNModel::NNQueryLabel>::
    SetModelMetaInfo(
        const boost::shared_ptr<DataLoader<kNNQuery, NNModel::NNQueryFeature,
                                           NNModel::NNQueryLabel>> &infosrc) {
  specifyfeatdim_ = true;
  if (features_.get() == nullptr) {
    features_.reset(new NNModel::NNQueryFeature());
  }

  if (labels_.get() == nullptr) {
    labels_.reset(new NNModel::NNQueryLabel());
  }

  features_->SetVocabulary(infosrc->features_->GetVocabulary());
  labels_->SetVocabulary(infosrc->labels_->GetVocabulary());
}

template <>
void DataLoader<kNNSequence, NNModel::NNSequenceFeature,
                NNModel::NNSequenceLabel>::
    SetModelMetaInfo(const boost::shared_ptr<
                     DataLoader<kNNSequence, NNModel::NNSequenceFeature,
                                NNModel::NNSequenceLabel>> &infosrc) {
  specifyfeatdim_ = true;

  if (features_.get() == nullptr)
    features_.reset(new NNModel::NNSequenceFeature());
  if (labels_.get() == nullptr)
    labels_.reset(new NNModel::NNSequenceLabel());

  features_->SetSparseBinarySize(infosrc->features_->GetSparseBinarySize());
  features_->SetSparseFloatSize(infosrc->features_->GetSparseFloatSize());
  features_->SetDenseSize(infosrc->features_->GetDenseSize());
  labels_->SetLabelSize(infosrc->labels_->GetLabelSize());
}

template class DataLoader<kLibSVM, DataSamples, LabelVector>;
template class DataLoader<kLCCRF, LccrfSamples, LccrfLabels>;
template class DataLoader<kNNQuery, NNModel::NNQueryFeature,
                          NNModel::NNQueryLabel>;
template class DataLoader<kNNSequence, NNModel::NNSequenceFeature,
                          NNModel::NNSequenceLabel>;
