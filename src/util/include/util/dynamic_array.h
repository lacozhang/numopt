#pragma once

#include "proto/dataconfig.pb.h"
#include "proto/param.pb.h"
#include "util/common.h"
#include "util/file.h"
#include "util/range.h"
#include "gtest/gtest.h"
#include <Eigen/Core>
#include <atomic>
#include <cstring>
#include <initializer_list>
#include <string>

namespace mltools {

template <typename V> class Matrix;

/**
 * A wrapper for dynamically allocated array, managed by std::shared_ptr.
 * With the goal to maintain the semantic consistence with raw pointers.
 * The managment of dynamically array is done via std::shared_ptr
 */
template <typename V> class DArray {
public:
  DArray() { defaultInit(); }
  ~DArray() {}

  /**
   * @brief Create an array with length n.
   *
   * The values of created array is not initialized, can be done via setValue or
   * setZero.
   */
  DArray(size_t n) {
    defaultInit();
    resize(n);
  }
  DArray(size_t n, V val) {
    defaultInit();
    resize(n, val);
  }

  /**
   * @brief Zero-copy constructor
   *
   * This constructor can only be applied on pointer that's not managed yet!!!
   *
   * @param data
   * @param size
   * @param deletable
   */
  DArray(V *data, size_t size, bool deletable = true) {
    reset(data, size, deletable);
  }

  /**
   * @brief Zero-copy constructor
   *
   * Just invoke the assign operator to finish the job with only pointer copy
   */
  template <typename W> explicit DArray(const DArray<W> &arr);

  /**
   * @brief Copy from initializer list
   *
   * @param list
   */
  template <typename W> DArray(const std::initializer_list<W> &list);

  /**
   * @brief Copy from initializer list
   *
   * @param list
   */
  template <typename W> void operator=(const std::initializer_list<W> &list);

  /**
   * @brief Zero-copy assign operator
   *
   * copy the pointer & shared_ptr, no new array allocated, should be fast
   */
  template <typename W> void operator=(const DArray<W> &arr);

  /// @brief Compare with different array
  template <typename W> bool operator==(const DArray<W> &rhs) const;

  /// @brief Compare operator
  template <typename W> bool operator!=(const DArray<W> &rhs) const;

  void clear() { reset(nullptr, 0, false); }

  /**
   * @brief Deep copy c-type array with specified size
   *
   * @param src
   * @param size
   */
  void copyFrom(const V *src, size_t size);

  /**
   * @brief Deep copy DArray
   *
   * @param arr
   */
  void copyFrom(const DArray<V> &arr);

  /**
   * @brief Deep copy data specified with iterator
   *
   * @param beginIt
   * @param endIt
   */
  template <typename RndAccessIt>
  void copyFrom(const RndAccessIt beginIt, const RndAccessIt endIt);

  /**
   * @brief Resize the array to n elements
   *
   * if capacity_ >= n, then only size_ = n happens. Otherwise, append (n -
   * capacity_) elements to the array without proper value initialization.
   *
   * @param n
   */
  void resize(size_t n);

  /**
   * @brief Slice a segment, zero-copy
   *
   * Return a segment of current array, wrapped within a DArray
   *
   * @param range
   * @return DArray [range.begin(), range.end())
   */
  DArray<V> segment(const Range<size_t> &range) const;

  /**
   * @brief The intersection of two sorted DArray
   *
   * Return an array storing the intersection of two sorted DArray
   * An example:
   \code{cpp}
   DArray<int> a{1,2,3,4}, b{3,4}, c{3,4};
   CHECK_EQ(a.setIntersection(b), c);
   \endcode
   *
   * @param other
   * @return *this \f$\cap\f$ other
   */
  DArray<V> setIntersection(const DArray<V> &other) const;

  /**
   * @brief The union of two sorted array
   *
   * Example below
   \code{cpp}
   DArray<int> a{1,2,3}, b{3,4}, c{1,2,3,4};
   CHECK_EQ(a.setUnion(b), c);
   \endcode
   *
   * @param other
   * @return *this \f$\cup\f$ other
   */
  DArray<V> setUnion(const DArray<V> &other) const;

  /**
   * @brief Find the index range with element values bounded
   *
   \code{cpp}
   DArray a{1,3,5,7,9};
   CHECK_EQ(a.findRange(Range<int>(2, 7)), SizeR(1,3));
   \endcode
   *
   * @param bound
   * @return SizeR
   */
  SizeR findRange(const Range<V> &bound) const;

  /**
   * @brief Return the consumed memory
   */
  size_t memSize() const { return capacity_ * sizeof(V); }

  /**
   * @brief allocate memory but not used
   */
  void reserve(size_t n);

  /**
   * @brief Resize the array to n element & reset all the values to val
   *
   * same logic as resize(size_t n) but with extra setValue call to initiaize
   * the new array
   *
   * @param n
   * @param val
   */
  void resize(size_t n, V val) {
    resize(n);
    setValue(val);
  }

  /**
   * @brief Reset the ptr_ to manage raw pointer data
   *
   * Release previous raw pointer and get new array managed. If deletable=true,
   * then ptr_ will able to delete the array.
   *
   * @param data
   * @param size
   * @param deletable
   */
  void reset(V *data, size_t size, bool deletable = true);

  /**
   * @brief Set each element in array to value
   *
   * @param val
   */
  void setValue(V val);

  /**
   * @brief Set value according to config
   */
  void setValue(const ParamInitConfig &cf);

  /**
   * @brief Return the element value range
   */
  Range<V> range() const {
    return empty() ? Range<V>(0, 0) : Range<V>(front(), back() + 1);
  }

  /// @brief Return #non-zeros in array
  size_t nnz() const;

  /**
   * @brief Set all the byte to value 0
   */
  void setZero() { memset(data_, 0, size_ * sizeof(V)); }

  /**
   * \brief Return the size of current array
   */
  size_t size() const { return size_; }

  /**
   * @brief empty or not
   */
  bool empty() const { return size() == 0; }

  /**
   * @brief compare data array values
   */
  bool valueCompare(const void *lhs, const void *rhs, size_t size) const;

  /**
   * @brief Return the capacity of current array
   */
  size_t capacity() const { return capacity_; }

  /**
   * @brief Return const pointer to data array
   */
  V *data() const { return data_; }

  /**
   * @brief Return the shared pointer object
   */
  const std::shared_ptr<void> &pointer() const { return ptr_; }
  std::shared_ptr<void> &pointer() { return ptr_; }

  V *begin() { return data(); }
  const V *begin() const { return data(); }
  V *end() { return data() + size(); }
  const V *end() const { return data() + size(); }

  V back() const {
    assert(!empty());
    return data_[size_ - 1];
  }

  V front() const {
    assert(!empty());
    return data_[0];
  }

  V &operator[](const int i) { return data_[i]; }
  const V &operator[](const int i) const { return data_[i]; }

  void append(const DArray<V> &tail);
  void push_back(const V &val);
  void pop_back() {
    if (size_ > 0) {
      --size_;
    }
  }

  /// @brief return an Eigen3 vector, zero-copy
  typedef Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>> EVecMap;
  EVecMap eigenVector() const { return EVecMap(data(), size()); }
  EVecMap vec() const { return EVecMap(data(), size()); }

  /// @brief return an Eigen3 array, zero-copy
  typedef Eigen::Map<Eigen::Array<V, Eigen::Dynamic, 1>> EArrayMap;
  EArrayMap eigenArray() const { return EArrayMap(data(), size()); }
  EArrayMap array() const { return EArrayMap(data(), size()); }

  /// @brief return an Eigen3 Matrix
  typedef Eigen::Map<Eigen::Array<V, Eigen::Dynamic, Eigen::Dynamic>> EMatMap;
  EMatMap eigenMatrix(size_t cols) const {
    ASSERT_TRUE(size() % cols == 0);
    ASSERT_TRUE(cols > 0);
    return EMatMap(data(), size() / cols, cols);
  }
  EMatMap mat(size_t cols) const {
    ASSERT_TRUE(size() % cols == 0);
    ASSERT_TRUE(cols > 0);
    return EMatMap(data(), size() / cols, cols);
  }

  double sum() const { return empty() ? 0 : array().sum(); }
  double mean() const { return empty() ? 0 : sum() / size_; }
  double std() const {
    return size_ <= 1 ? 0
                      : (array() - mean()).matrix().norm() /
                            static_cast<double>(size_);
  }

  /**
   * @brief return the data in mltools::DenseMatrix class as column vector
   */
  std::shared_ptr<Matrix<V>> denseMatrix(size_t rows = -1, size_t cols = -1);

  /**
   * @brief Return the compressed array
   */
  DArray<char> compressTo() const;

  /// @brief uncompress the data in src of size srcSize, but this array need to
  /// be large enough.
  void uncompressFrom(const char *src, size_t srcSize);
  void uncompressFrom(const DArray<char> &src) {
    uncompressFrom(src.data(), src.size());
  }

  /// @brief read the specified segment from binary file *fileName*
  bool readFromFile(SizeR range, const std::string &fileName);
  bool readFromFile(const std::string &fileName) {
    return readFromFile(SizeR::all(), fileName);
  }
  bool readFromFile(SizeR range, DataConfig &config);

  /// @brief Write all the values to file in binary mode
  bool writeToFile(const std::string &fileName) const {
    return writeToFile(SizeR(0, size_), fileName);
  }

  /// @brief Write all the values within segment range to file in binary mode
  bool writeToFile(SizeR range, const std::string &fileName) const;

private:
  void defaultInit() {
    size_ = capacity_ = 0;
    data_ = nullptr;
    ptr_ = std::shared_ptr<void>(nullptr);
  }

  size_t size_;
  size_t capacity_;
  V *data_;
  std::shared_ptr<void> ptr_;
};

template <typename V>
std::ostream &operator<<(std::ostream &os, const DArray<V> &val) {
  os << dbgstr(val.data(), 10);
  return os;
}

} // namespace mltools
