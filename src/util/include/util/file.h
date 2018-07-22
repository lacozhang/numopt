/*
 * =====================================================================================
 *
 *       Filename:  file.h
 *
 *    Description:  read & write proto buffer into file
 *
 *        Version:  1.0
 *        Created:  07/07/2018 15:21:18
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#ifndef __FILE_H__
#define __FILE_H__
#include "proto/dataconfig.pb.h"
#include "util/common.h"
#include "util/integral_types.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <cstdlib>
#include <glog/logging.h>
#include <string>
#include <zlib.h>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/tokenizer.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

namespace mltools {

class File {
public:
  /// @brief wrapper of fopen in c lib.
  static File *open(const std::string &filePath, const char *const flags);
  /// @brief Program will exit if failed to open
  static File *openOrDie(const std::string &filePath, const char *const flags);
  /// @brief Read file specified in DataConfig
  static File *open(const DataConfig &config, const char *const flags);
  /// @brief exit when failed to open
  static File *openOrDie(const DataConfig &config, const char *const flags);
  /// @brief return the size of a file
  static size_t size(const std::string &filepath);
  /// @brief delete a file
  static bool remove(const std::string &filepath) {
    return std::remove(filepath.c_str());
  }
  /// @brief get file existed or not
  static bool exists(const char *filepath) {
    return access(filepath, F_OK) == 0;
  }
  static bool gzfile(const std::string &filepath) {
    return (filepath.size() > 3) &&
           (std::string(filepath.end() - 3, filepath.end()) == ".gz");
  }

  virtual ~File();

  /// @brief read at most "size" bytes to pre-allocated buffer "buff"
  size_t read(void *const buff, size_t size);
  void readOrDie(void *const buff, size_t size) {
    ASSERT_EQ(read(buff, size), size);
  }

  /// @brief Read a line with at most "maxLength" bytes from a file
  char *readLine(char *const output, uint64 maxLength);

  /**
   * @brief Read the whole file content into a string, with maximum length of
   * "maxLength"
   *
   * @return The number of bytes already read.
   */
  int64 readToString(std::string *const line, uint64 maxLength);

  /**
   * @brief write "size" bytes to file
   *
   * @param buff the source buffer
   * @param size The number of bytes need to write out
   *
   * @return The number of bytes written successfully
   */
  size_t write(const void *const buff, size_t size);

  /// @brief Exit when failed to write
  void writeOrDie(const void *const buff, size_t size) {
    ASSERT_EQ(write(buff, size), size) << "Failed to write all the bytes";
  }

  /// @brief write a line to file
  size_t writeString(const std::string &line);

  /// @brief flush the buffer
  bool flush();

  /// @brief close the opened file
  bool close();

  /// @brief size of file
  size_t size() const;

  /// @brief seek a position, start from head
  bool seek(size_t pos);

  std::string filename() const { return name_; }

  /// @brief check file is open or not
  bool isOpen() const { return (f_ != NULL) || (gz_ != NULL); }

private:
  File(FILE *fdesc, const std::string &filePath) : f_(fdesc), name_(filePath) {
    gz_ = NULL;
    isgz_ = false;
  }

  File(gzFile gz, const std::string &filePath) : gz_(gz), name_(filePath) {
    isgz_ = true;
  }

  FILE *f_ = NULL;
  gzFile gz_ = NULL;
  bool isgz_ = false;
  const std::string name_;
};

bool readFileToString(const std::string &filepath, std::string *output);
bool writeStringToFile(const std::string &data, const std::string &filepath);

typedef google::protobuf::Message GProto;
bool readFileToProto(const DataConfig &name, GProto *proto);
void readFileToProtoOrDie(const DataConfig &name, GProto *proto);

bool readFileToProto(const std::string &name, GProto *proto);
void readFileToProtoOrDie(const std::string &name, GProto *proto);

bool writeProtoToASCIIFile(const GProto &proto, const std::string &filepath);
void writeProtoToASCIIFileOrDie(const GProto &proto,
                                const std::string &filepath);

bool writeProtoToFile(const GProto &proto, const std::string &filepath);
void writeProtoToFileOrDie(const GProto &proto, const std::string &filepath);

std::string hadoopFS(const HDFSConfig &conf);

bool dirExists(const std::string &dir);
bool dirCreate(const std::string &dir);

std::vector<std::string> readFilenamesInDir(const std::string &dir);
std::vector<std::string> readFilenamesInDir(const DataConfig &dir);

std::string removeExtension(const std::string &filepath);
std::string getPath(const std::string &fullpath);
std::string getFilename(const std::string &filepath);
} // namespace mltools

#endif // __FILE_H__
