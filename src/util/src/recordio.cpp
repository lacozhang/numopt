/*
 * =====================================================================================
 *
 *       Filename:  recordio.cpp
 *
 *    Description:  implementation of binary data operation
 *
 *        Version:  1.0
 *        Created:  07/08/2018 18:39:53
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#include "util/recordio.h"

namespace mltools {
bool RecordReader::valid() const { return f_ && f_->isOpen(); }

bool RecordWriter::valid() const { return f_ && f_->isOpen(); }
} // namespace mltools
