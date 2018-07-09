/*
 * =====================================================================================
 *
 *       Filename:  common.h
 *
 *    Description:  common utility for DataConfig
 *
 *        Version:  1.0
 *        Created:  07/08/2018 14:06:50
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "proto/dataconfig.pb.h"
#include "proto/example.pb.h"
#include "proto/matrix.pb.h"
#include "util/common.h"

#ifndef __DATA_COMMON_H__
#define __DATA_COMMON_H__

namespace mltools {

DECLARE_string(input);
DECLARE_string(output);
DECLARE_string(format);

enum class FeatureConstants { kSlotIdMax = 4096 };

/// @brief search all the files match regex "conf"
DataConfig searchFiles(const DataConfig &conf);

/// @brief split files in "conf" into "num" parts
std::vector<DataConfig> divideFiles(const DataConfig &conf, int num);

/// @brief find the i-th file, append file name with suffix and keep other
/// untouched
DataConfig ithFile(const DataConfig &conf, int i, std::string suffix = "");

/// @brief merge files
DataConfig appendFiles(const DataConfig &confA, const DataConfig &confB);

/// @brief merge examples info from two different training data
ExampleInfo mergeExampleInfo(const ExampleInfo &infoA,
                             const ExampleInfo &infoB);

MatrixInfo readMatrixInfo(const ExampleInfo &info, int slotId, int sizeOfIdx,
                          int sizeOfVal);

DataConfig shuffleFiles(const DataConfig &data);
} // namespace mltools

#endif // __DATA_COMMON_H__
