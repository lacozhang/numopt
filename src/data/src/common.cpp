/*
 * =====================================================================================
 *
 *       Filename:  common.cpp
 *
 *    Description:  implement the interface in common.h
 *
 *        Version:  1.0
 *        Created:  07/08/2018 14:08:57
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "data/common.h"
#include "util/file.h"
#include <regex>

namespace mltools {
DEFINE_string(input, "stdin", "stdin or a filename");
DEFINE_string(output, "stdout", "stdout or a filename");
DEFINE_string(format, "none", "proto, pserver, libsvm, vw, adfea, or others");
DEFINE_uint64(hash_kernel, 0, "hash kernel size");
DECLARE_bool(verbose);

DataConfig ithFile(const DataConfig &conf, int i, std::string suffix) {
  CHECK_GE(i, 0);
  CHECK_LT(i, conf.file_size());
  auto f = conf;
  f.clear_file();
  f.add_file(conf.file(i) + suffix);
  return f;
}

DataConfig appendFiles(const DataConfig &confA, const DataConfig &confB) {
  DataConfig ret = confA;
  for (int i = 0; i < confB.file_size(); ++i) {
    ret.add_file(confB.file(i));
  }
  return ret;
}

MatrixInfo readMatrixInfo(const ExampleInfo &info, int slotId, int sizeOfIdx,
                          int sizeOfVal) {
  MatrixInfo minfo;
  int slotIdx = 0;
  while (slotIdx < info.slot_size()) {
    if (info.slot(slotIdx).id() == slotId) {
      break;
    }
    ++slotIdx;
  }
  if (slotIdx >= info.slot_size()) {
    return minfo;
  }
  auto slot = info.slot(slotIdx);
  if (slot.format() == SlotInfo::DENSE) {
    minfo.set_type(MatrixInfo::DENSE);
  } else if (slot.format() == SlotInfo::SPARSE) {
    minfo.set_type(MatrixInfo::SPARSE);
  } else if (slot.format() == SlotInfo::SPARSE_BINARY) {
    minfo.set_type(MatrixInfo::SPARSE_BINARY);
  }
  minfo.set_row_major(true);
  minfo.set_grp_id(slotId);
  minfo.mutable_row()->set_begin(0);
  minfo.mutable_row()->set_end(info.num_ex());
  minfo.mutable_col()->set_begin(slot.min_key());
  minfo.mutable_col()->set_end(slot.max_key());
  minfo.set_nnz(slot.nnz_ele());
  minfo.set_sizeof_idx(sizeOfIdx);
  minfo.set_sizeof_val(sizeOfVal);
  return minfo;
}

ExampleInfo mergeExampleInfo(const ExampleInfo &infoA,
                             const ExampleInfo &infoB) {
  std::map<int, SlotInfo> slots;
  for (int i = 0; i < infoA.slot_size(); ++i) {
    slots[infoA.slot(i).id()] = infoA.slot(i);
  }

  for (int i = 0; i < infoB.slot_size(); ++i) {
    auto sid = infoB.slot(i).id();
    if (!slots.count(sid)) {
      slots[sid] = infoB.slot(i);
    } else {
      SlotInfo &slotA = slots[sid];
      CHECK_EQ(slotA.format(), infoB.slot(i).format());
      SlotInfo c = slotA;
      c.set_min_key(std::min(slotA.min_key(), infoB.slot(i).min_key()));
      c.set_max_key(std::max(slotA.max_key(), infoB.slot(i).max_key()));
      c.set_nnz_ex(slotA.nnz_ex() + infoB.slot(i).nnz_ex());
      c.set_nnz_ele(slotA.nnz_ele() + infoB.slot(i).nnz_ele());
      slots[sid] = c;
    }
  }

  ExampleInfo ret;
  ret.set_num_ex(infoA.num_ex() + infoB.num_ex());
  for (const auto &slot : slots) {
    *(ret.add_slot()) = slot.second;
  }
  return ret;
}

DataConfig searchFiles(const DataConfig &conf) {
  int num = conf.file_size();
  CHECK_GE(num, 1) << "No files in config " << conf.DebugString();
  std::vector<std::string> matchFiles;

  for (int i = 0; i < num; ++i) {
    std::regex pattern;
    bool valid = false;
    try {
      pattern = std::regex(getFilename(conf.file(i)));
      valid = true;
    } catch (std::regex_error &e) {
      VLOG(1) << "invalid regex pattern from filename "
              << getFilename(conf.file(i));
    }

    std::string baseName = getPath(conf.file(i));
    std::vector<std::string> filenames = {};
    if (valid) {
      filenames = readFilenamesInDir(baseName);
    }
    for (const auto &filename : filenames) {
      if (std::regex_match(getFilename(filename), pattern)) {
        auto l = conf.format() == DataConfig::TEXT ? filename
                                                   : removeExtension(filename);
        matchFiles.push_back(baseName + "/" + getFilename(l));
      }
    }
  }
  std::sort(matchFiles.begin(), matchFiles.end());
  auto it = std::unique(matchFiles.begin(), matchFiles.end());
  matchFiles.resize(std::distance(matchFiles.begin(), it));
  DataConfig ret = conf;
  ret.clear_file();
  for (auto &file : matchFiles) {
    ret.add_file(file);
  }
  return ret;
}

std::vector<DataConfig> divideFiles(const DataConfig &conf, int num) {
  std::vector<DataConfig> ret;
  for (int i = 0; i < num; ++i) {
    DataConfig df = conf;
    df.clear_file();
    for (int j = 0; j < conf.file_size(); ++j) {
      if (j % num == i) {
        df.add_file(conf.file(j));
      }
      auto limits = conf.maximum_files_per_worker();
      if (limits >= 0 && limits <= df.file_size()) {
        LOG(WARNING) << "num is too small, overload the worker"
                     << conf.DebugString();
        break;
      }
    }
    ret.emplace_back(df);
  }
  return ret;
}

DataConfig shuffleFiles(const DataConfig &data) {
  DataConfig ret = data;
  ret.clear_file();
  std::vector<int> idxes(data.file_size());
  for (int i = 0; i < data.file_size(); ++i) {
    idxes[i] = i;
  }
  std::random_shuffle(idxes.begin(), idxes.end());
  for (int i = 0; i < idxes.size(); ++i) {
    ret.add_file(data.file(idxes[i]));
  }
  return ret;
}
} // namespace mltools
