#include "proto/matrix.pb.h"
#include "util/dynamic_array.h"
#include <algorithm>
#include <random>

namespace mltools {

template <typename V> void DArray<V>::resize(size_t n) {
  if (capacity_ >= n) {
    size_ = n;
    return;
  }
  V *data = new V[n + 5];
  memcpy(data, data_, size_ * sizeof(V));
  reset(data, n);
}

template <typename V>
void DArray<V>::reset(V *data, size_t size, bool deletable) {
  capacity_ = size;
  size_ = size;
  data_ = data;
  if (deletable) {
    ptr_.reset(reinterpret_cast<char *>(data_), [](char *p) { delete[] p; });
  } else {
    ptr_.reset(reinterpret_cast<char *>(data_), [](char *p) {});
  }
}

template <typename V> void DArray<V>::setValue(V val) {
  if (val == 0) {
    setZero();
  } else {
    for (int i = 0; i < size_; ++i) {
      data_[i] = val;
    }
  }
}

template <typename V> template <typename W> DArray<V>::DArray(DArray<W> &arr) {
  *this = arr;
}

template <typename V>
template <typename W>
void DArray<V>::operator=(const DArray<W> &arr) {
  size_ = (arr.size() * sizeof(W)) / sizeof(V);
  capacity_ = (arr.capacity() * sizeof(W)) / sizeof(V);
  data_ = reinterpret_cast<V *>(arr.data());
  ptr_ = arr.pointer();
}

template <typename V> void DArray<V>::copyFrom(const V *src, size_t size) {
  resize(size);
  memcpy(data_, src, size * sizeof(V));
}

template <typename V> void DArray<V>::copyFrom(const DArray<V> &arr) {
  copyFrom(arr.data(), arr.size());
}

template <typename V>
template <typename RndAccessIt>
void DArray<V>::copyFrom(RndAccessIt beginIt, RndAccessIt endIt) {
  size_t srcSize = std::distance(beginIt, endIt);
  V *data = new V[srcSize + 5];
  reset(data, srcSize);
  for (size_t i = 0; i < srcSize; ++i) {
    data_[i] = *(beginIt + i);
  }
}

template <typename V>
template <typename W>
DArray<V>::DArray(const std::initializer_list<W> &list) {
  copyFrom(list.begin(), list.end());
}

template <typename V>
template <typename W>
void DArray<V>::operator=(const std::initializer_list<W> &list) {
  copyFrom(list.begin(), list.end());
}

template <typename V>
DArray<V> DArray<V>::segment(const Range<size_t> &range) const {
  EXPECT_TRUE(range.valid());
  EXPECT_LE(range.end(), size());
  DArray<V> result = *this;
  result.data_ += range.begin();
  result.size_ = result.capacity_ = range.size();
  return result;
}

template <typename V>
DArray<V> DArray<V>::setIntersection(const DArray<V> &other) const {
  DArray<V> result(std::min(other.size(), size()) + 1);
  V *last = std::set_intersection(begin(), end(), other.begin(), other.end(),
                                  result.begin());
  result.size_ = last - result.begin();
  return result;
}

template <typename V>
DArray<V> DArray<V>::setUnion(const DArray<V> &other) const {
  DArray<V> result(other.size() + size());
  V *last = std::set_union(begin(), end(), other.begin(), other.end(),
                           result.begin());
  result.size_ = last - result.begin();
  return result;
}

template <typename V> SizeR DArray<V>::findRange(const Range<V> &bound) const {
  if (empty()) {
    return SizeR(0, 0);
  }
  EXPECT_TRUE(bound.valid());
  auto lb = std::lower_bound(begin(), end(), bound.begin());
  auto ub = std::lower_bound(begin(), end(), bound.end());
  return SizeR(lb - begin(), ub - begin());
}

template <typename V>
template <typename W>
bool DArray<V>::operator==(const DArray<W> &rhs) const {
  if ((size() * sizeof(V)) != (rhs.size() * sizeof(W))) {
    return false;
  }
  if (size() == 0) {
    return true;
  }
  return (std::memcmp(data(), rhs.data(), size() * sizeof(V)) == 0);
}

template <typename V> void DArray<V>::reserve(size_t n) {
  if (capacity_ >= n) {
    return;
  }
  auto prevSize = size_;
  resize(n);
  size_ = prevSize;
}

template <typename V> void DArray<V>::append(const DArray<V> &tail) {
  if (tail.empty()) {
    return;
  }
  auto prevSize = size();
  resize(prevSize + tail.size());
  std::memcpy(data() + prevSize, tail.data(), tail.size() * sizeof(V));
}

template <typename V> void DArray<V>::push_back(const V &val) {
  if (size_ == capacity_) {
    reserve(size_ * 2);
  }
  data_[size_++] = val;
}

template <typename V> void DArray<V>::setValue(const ParamInitConfig &cf) {
  typedef ParamInitConfig Type;
  if (cf.type() == Type::ZERO) {
    setZero();
  } else if (cf.type() == Type::CONSTANT) {
    setValue(static_cast<V>(cf.constant()));
  } else if (cf.type() == Type::GAUSSIAN) {
    std::default_random_engine generator;
    std::normal_distribution<V> distribution(static_cast<V>(cf.mean()),
                                             static_cast<V>(cf.std()));
    for (size_t i = 0; i < size_; ++i) {
      data_[i] = distribution(generator);
    }
  } else if (cf.type() == Type::FILE) {
    EXPECT_TRUE(false);
  }
}

template <typename V> size_t DArray<V>::nnz() const {
  size_t nnzCount = 0;
  for (size_t i = 0; i < size_; ++i) {
    nnzCount += data_[i] == 0 ? 1 : 0;
  }
  return nnzCount;
}

template <typename V>
std::shared_ptr<Matrix<V>> DArray<V>::SMatrix(size_t rows, size_t cols) {
  MatrixInfo info;
  info.set_type(MatrixInfo::DENSE);
  info.set_row_major(false);
  SizeR(0, size_).to(info.mutable_row());
  SizeR(0, 1).to(info.mutable_col());
  info.set_nnz(size_);
  info.set_sizeof_val(sizeof(V));
  return std::shared_ptr<Matrix<V>>(new DenseMatrix(info, *this));
}

} // namespace mltools
