/*
 * =====================================================================================
 *
 *       Filename:  recordio.h
 *
 *    Description:  Read training data in binary format
 *
 *        Version:  1.0
 *        Created:  07/08/2018 18:39:28
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
#include "util/file.h"
#include <memory>
#include <string>

#ifndef __RECORD_IO_H__
#define __RECORD_IO_H__

namespace {
static const int kBinaryMagic = 0x3ed7230a;
}

namespace mltools {

class RecordWriter {
public:
  RecordWriter() { f_ = nullptr; }
  explicit RecordWriter(File *file) : f_(file) {}
  bool close() { return f_ && f_->close(); }

  bool valid() const;

  template <class P> bool writeProtoMessage(const P &proto) {
    if (f_ == nullptr) {
      return false;
    }
    std::string content;
    proto.SerializeToString(&content);
    size_t size = content.size();
    if (f_->write(&kBinaryMagic, sizeof(kBinaryMagic)) !=
        sizeof(kBinaryMagic)) {
      return false;
    }
    if (f_->write(&size, sizeof(size)) != sizeof(size)) {
      return false;
    }
    if (f_->write(content.c_str(), size) != size) {
      return false;
    }
    return true;
  }

private:
  File *f_ = nullptr;
};

class RecordReader {
public:
  RecordReader() { f_ = nullptr; }
  explicit RecordReader(File *f) : f_(f) {}
  bool close() { return f_ && f_->close(); }

  bool valid() const;

  template <class P> bool readProtoMessage(P *proto) {
    if (!proto || !f_) {
      return false;
    }
    size_t size = 0;
    int magicNumber = 0;
    if (f_->read(&magicNumber, sizeof(magicNumber)) != sizeof(magicNumber)) {
      return false;
    }
    if (magicNumber != kBinaryMagic) {
      return false;
    }
    if (f_->read(&size, sizeof(size)) != sizeof(size)) {
      return false;
    }
    std::unique_ptr<char[]> buffer(new char[size + 1]);
    if (f_->read(buffer.get(), size) != size) {
      return false;
    }
    proto->ParseFromArray(buffer.get(), size);
    return true;
  }

private:
  File *f_ = nullptr;
};
} // namespace mltools

#endif // __RECORD_IO_H__
