/*
 * =====================================================================================
 *
 *       Filename:  slot_reader.cpp
 *
 *    Description:  implement coordinate fashion reading
 *
 *        Version:  1.0
 *        Created:  07/08/2018 13:50:51
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "data/slot_reader.h"
#include "data/text_parser.h"
#include "util/filelinereader.h"
#include "util/recordio.h"
#include "util/threadpool.h"

namespace mltools {

void SlotReader::init(const DataConfig &data, const DataConfig &cache) {
  ASSERT_TRUE(cache.file_size() == 1);
  data_ = data;
  cache_ = cache.file(0);
}

std::string SlotReader::cacheName(const DataConfig &data, int slotId) const {
  CHECK_GE(data.file_size(), 1);
  return cache_ + getFilename(data.file(0)) + "_slot_" + std::to_string(slotId);
}

size_t SlotReader::nnzEle(int slotId) const {
  size_t nnz = 0;
  for (int i = 0; i < info_.slot_size(); ++i) {
    if (info_.slot(i).id() == slotId) {
      nnz = info_.slot(i).nnz_ele();
    }
  }
  return nnz;
}

DArray<uint64> SlotReader::index(int slotId) {
  size_t nnz = nnzEle(slotId);
  if (nnz == 0) {
    return DArray<uint64>();
  }
  auto idx = indexCache_[slotId];
  if (idx.size() == nnz) {
    return idx;
  }
  idx.clear();
  idx.reserve(nnz);
  for (int i = 0; i < data_.file_size(); ++i) {
    std::string cacheFilepath =
        cacheName(ithFile(data_, i), slotId) + ".colidx";
    DArray<char> comp;
    if (!comp.readFromFile(cacheFilepath)) {
      LOG(INFO) << "Failed to read from " << cacheFilepath;
      continue;
    }
    LOG(INFO) << "raw index size " << comp.size();
    DArray<uint64> raw;
    raw.uncompressFrom(comp);
    idx.append(raw);
  }
  CHECK_EQ(idx.size(), nnz);
  indexCache_[slotId] = idx;
  return idx;
}

DArray<size_t> SlotReader::offset(int slotId) {
  if (offsetCache_[slotId].size() == info_.num_ex() + 1) {
    return offsetCache_[slotId];
  }

  DArray<size_t> os(1);
  os[0] = 0;
  int totalExamples = 0;
  for (int i = 0; i < data_.file_size(); ++i) {
    std::string cachepath = cacheName(ithFile(data_, i), slotId) + ".rowsiz";
    DArray<char> comp;
    DArray<uint16> uncomp;
    if (comp.readFromFile(cachepath) && !comp.empty()) {
      LOG(INFO) << "raw offset size " << comp.size();
      uncomp.uncompressFrom(comp);
    } else {
      LOG(INFO) << "raw offset all 0 ";
      uncomp.resize(numEx_[i], 0);
    }
    totalExamples += numEx_[i];
    CHECK_EQ(uncomp.size(), numEx_[i]);
    size_t prevSize = os.size();
    os.resize(prevSize + uncomp.size());
    for (int i = 0; i < uncomp.size(); ++i) {
      os[prevSize + i] = os[prevSize + i - 1] + uncomp[i];
    }
  }
  CHECK_EQ(totalExamples, info_.num_ex());
  CHECK_EQ(os.size(), info_.num_ex() + 1);
  offsetCache_[slotId] = os;
  return os;
}

bool SlotReader::readOneFile(InfoParser &metaParser, const DataConfig &data,
                             int ithFile) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    LOG(INFO) << "loading data file [" << data.file(0) << "]; loaded ["
              << loadedFileCount_ << "/" << data_.file_size() << "]";
  }
  std::string infoCachepath = cache_ + getFilename(data.file(0)) + ".info";
  ExampleInfo info;
  if (readFileToProto(infoCachepath, &info)) {
    std::lock_guard<std::mutex> lk(mu_);
    LOG(INFO) << "Read meta info " << info.DebugString();
    info_ = mergeExampleInfo(info_, info);
    numEx_[ithFile] = info.num_ex();
    loadedFileCount_++;
    return true;
  }

  Example ex;
  metaParser.clear();
  size_t exnum = 0;
  size_t failedToParseCnt = 0, failedToAddCnt = 0;
  VSlot slots[static_cast<int>(FeatureConstants::kSlotIdMax)];
  auto storeData = [&]() {
    if (!metaParser.add(ex)) {
      failedToAddCnt++;
      return;
    }
    CHECK(ex.slot_size() > 0);
    for (int i = 0; i < ex.slot_size(); ++i) {
      auto &slot = ex.slot(i);
      CHECK_LT(slot.id(), static_cast<int>(FeatureConstants::kSlotIdMax));
      auto &slotMeta = slots[slot.id()];
      for (int j = 0; j < slot.key_size(); ++j) {
        slotMeta.idx_.push_back(slot.key(j));
      }
      for (int j = 0; j < slot.val_size(); ++j) {
        slotMeta.val_.push_back(slot.val(j));
      }
      while (slotMeta.cnt_.size() < exnum) {
        slotMeta.cnt_.push_back(0);
      }
      slotMeta.cnt_.push_back(std::max(slot.key_size(), slot.val_size()));
    }
    ++exnum;
  };

  if (data.format() == DataConfig::TEXT) {
    FileLineReader src(data);
    ExampleParser textParser;
    textParser.init(data.text(), data.ignore_feature_group());
    std::function<void(char *)> handle = [&](char *line) {
      if (!textParser.toProto(line, &ex)) {
        failedToParseCnt++;
        return;
      }
      storeData();
    };
    src.setLineCallback(handle);
    LOG(INFO) << "load text data";
    src.reload();
    CHECK(src.loadedSuccessfully());
  } else if (data.format() == DataConfig::PROTO) {
    RecordReader reader;
    while (reader.readProtoMessage(&ex)) {
      storeData();
    }
  } else {
    LOG(ERROR) << "Unsupported data format " << data.DebugString();
    return false;
  }

  LOG(INFO) << "There are " << failedToParseCnt << " lines failed to parse";
  LOG(INFO) << "There are " << failedToAddCnt << " examples failed to add";

  if (!dirExists(getPath(infoCachepath))) {
    LOG(INFO) << "Create directory " << getPath(infoCachepath);
    dirCreate(getPath(infoCachepath));
  }

  info = metaParser.info();
  LOG(INFO) << "info of file " << data.file(0) << ":" << info.DebugString();
  CHECK_EQ(info.num_ex(), exnum);
  writeProtoToASCIIFileOrDie(info, infoCachepath);
  for (int i = 0; i < static_cast<int>(FeatureConstants::kSlotIdMax); ++i) {
    auto &slotMeta = slots[i];
    if (slotMeta.cnt_.empty() && slotMeta.val_.empty()) {
      continue;
    }
    while (slotMeta.cnt_.size() < exnum) {
      slotMeta.cnt_.push_back(0);
    }
    CHECK_EQ(slotMeta.cnt_.size(), exnum);
    auto slotCacheName = cacheName(data, i);
    LOG(INFO) << "Write data to " << slotCacheName;
    if (!dirExists(getPath(slotCacheName))) {
      LOG(INFO) << "Creating directory " << getPath(slotCacheName);
      dirCreate(getPath(slotCacheName));
    }
    slotMeta.writeToFile(cacheName(data, i));
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    info_ = mergeExampleInfo(info_, info);
    loadedFileCount_++;
    numEx_[ithFile] = exnum;
    VLOG(1) << "loaded data file [" << data.file(0) << "]; loaded ["
            << loadedFileCount_ << "/" << data_.file_size() << "]";
  }
  return true;
}

int SlotReader::read(ExampleInfo *info) {
  CHECK_GT(FLAGS_num_threads, 0);
  {
    std::lock_guard<std::mutex> lk(mu_);
    loadedFileCount_ = 0;
    numEx_.resize(data_.file_size(), 0);
  }
  std::vector<InfoParser> infoArrays;
  {
    ThreadPool pool(FLAGS_num_threads);
    infoArrays.resize(data_.file_size());
    for (int i = 0; i < data_.file_size(); ++i) {
      auto ithData = ithFile(data_, i);
      auto *ithInfoParser = &infoArrays[i];
      pool.add([this, ithData, i, ithInfoParser]() {
        readOneFile(*ithInfoParser, ithData, i);
      });
    }
    pool.startWorkers();
  }

  if (info != nullptr) {
    *info = info_;
  }
  for (int i = 0; i < info_.slot_size(); ++i) {
    slotsInfo_[info_.slot(i).id()] = info_.slot(i);
  }
  return info_.num_ex();
}

} // namespace mltools
