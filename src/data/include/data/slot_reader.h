/*
 * =====================================================================================
 *
 *       Filename:  slot_reader.h
 *
 *    Description:  read training data in block coordinate fashion
 *
 *        Version:  1.0
 *        Created:  07/08/2018 13:50:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "data/common.h"
#include "data/info_parser.h"
#include "proto/example.pb.h"
#include "util/dynamic_array_impl.h"
#include <string>

namespace mltools {
class SlotReader {
public:
  SlotReader() {}
  SlotReader(const DataConfig &data, const DataConfig &cache) {
    init(data, cache);
  }

  void init(const DataConfig &data, const DataConfig &cache);

  /**
   * @brief main inferface for slot style reader
   *
   * For slot-style reader, will read training data in block-coordinate style.
   */
  int read(ExampleInfo *info = nullptr);

  template <typename V> MatrixInfo info(int slotId) const {
    return readMatrixInfo(info_, slotId, sizeof(uint64), sizeof(V));
  }

  DArray<size_t> offset(int slotId);
  DArray<uint64> index(int slotId);
  template <typename V> DArray<V> value(int slotId) const;

  void clear(int slotId) {
    offsetCache_.erase(slotId);
    indexCache_.erase(slotId);
  }

private:
  struct VSlot {
    DArray<float> val_;
    DArray<uint64> idx_;
    DArray<uint16> cnt_;
    bool writeToFile(const std::string &prefix) {
      return cnt_.compressTo().writeToFile(prefix + ".rowsiz") &&
             idx_.compressTo().writeToFile(prefix + ".colidx") &&
             val_.compressTo().writeToFile(prefix + ".value");
    }
  };

  std::string cacheName(const DataConfig &data, int slotId) const;
  size_t nnzEle(int slotId) const;
  bool readOneFile(InfoParser &metaParser, const DataConfig &data, int ithFile);

  std::string cache_;
  DataConfig data_;
  ExampleInfo info_;
  std::mutex mu_;
  size_t loadedFileCount_ = 0;
  std::vector<uint32> numEx_;
  std::unordered_map<int, DArray<size_t>> offsetCache_;
  std::unordered_map<int, DArray<uint64>> indexCache_;
  std::unordered_map<int, SlotInfo> slotsInfo_;
};

template <typename V> DArray<V> SlotReader::value(int slotId) const {
  DArray<V> val;
  if (nnzEle(slotId) == 0) {
    return val;
  }
  for (int i = 0; i < data_.file_size(); ++i) {
    std::string cacheFilepath = cacheName(ithFile(data_, i), slotId) + ".value";
    DArray<char> raw;
    CHECK(raw.readFromFile(cacheFilepath));
    LOG(INFO) << "Raw size " << raw.size();
    DArray<V> full;
    full.uncompressFrom(raw);
    auto prevSize = val.size();
    int currSize = prevSize + full.size();
    val.resize(currSize);
    std::memcpy(val.data() + prevSize, full.data(), sizeof(V) * full.size());
  }
  CHECK_EQ(val.size(), nnzEle(slotId)) << " Read corrupted data";
  return val;
}
} // namespace mltools
