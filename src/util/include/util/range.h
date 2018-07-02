#pragma once
#include "proto/range.pb.h"
#include <limits>
#include <numeric>
#include <string>

namespace mltools {

template <typename T> class Range;
typedef Range<size_t> SizeR;

// Represent a range of form [a, b)
template <class T> class Range {
public:
  Range() : begin_(0), end_(0) {}

  template <typename V> Range(const Range<V> &other) {
    set(other.begin(), other.end());
  }

  template <typename U, typename V> Range(U begin, V end) { set(begin, end); }

  Range(const PbRange &pb) { copyFrom(pb); }

  template <typename U> void operator=(const Range<U> &rhs) {
    set(rhs.begin(), rhs.end());
  }

  void copyFrom(const PbRange &pb) { set(pb.begin(), pb.end()); }
  void to(PbRange *pb) const {
    pb->set_begin(begin_);
    pb->set_end(end_);
  }

  template <typename U, typename V> void set(U begin, V end) {
    begin_ = static_cast<T>(begin);
    end_ = static_cast<T>(end);
  }

  T begin() const { return begin_; }
  T &begin() { return begin_; }

  T end() const { return end_; }
  T &end() { return end_; }

  size_t size() const { return static_cast<size_t>(end_ - begin_); }
  bool valid() const { return end_ >= begin_; }
  bool empty() const { return end_ <= begin_; }

  bool operator==(const Range &rhs) const {
    return (begin_ == rhs.begin_) && (end_ == rhs.end_);
  }

  Range operator+(const T v) const { return Range(begin_ + v, end_ + v); }
  Range operator-(const T v) const { return Range(begin_ - v, end_ - v); }
  Range operator*(const T v) const { return Range(begin_ * v, end_ * v); }

  template <typename V> bool contains(const V &v) const {
    return (begin_ <= static_cast<T>(v)) && (static_cast<T>(v) < end_);
  }

  bool inLeft(const Range &other) const {
    return (begin_ <= other.begin_) ||
           (begin_ == other.begin_ && end_ <= other.end_);
  }

  bool inRight(const Range &other) const { return !inLeft(other); }

  // project into interval
  template <typename V> V project(const V &v) const {
    return static_cast<V>(std::max(begin_, std::min(end_, static_cast<T>(v))));
  }

  Range setIntersection(const Range &dest) const {
    return Range(std::max(begin_, dest.begin_), std::min(end_, dest.end_));
  }

  Range setUnion(const Range &dest) const {
    return Range(std::min(begin_, dest.begin_), std::max(end_, dest.end_));
  }

  Range evenDivide(size_t n, size_t i) const;

  std::string toString() const {
    return ("[" + std::to_string(begin_) + "," + std::to_string(end_) + ")");
  }

  static Range all() { return Range(0, std::numeric_limits<T>::max()); }

private:
  T begin_;
  T end_;
};

template <typename T> Range<T> Range<T>::evenDivide(size_t n, size_t i) const {
  auto itv =
      static_cast<long double>(end_ - begin_) / static_cast<long double>(n);
  return Range(static_cast<T>(begin_ + itv * i),
               static_cast<T>(begin_ + itv * (i + 1)));
}

template <typename T>
std::ostream &operator<<(std::ostream &os, Range<T> &obj) {
  return (os << obj.toString());
}
} // namespace mltools

namespace std {

// support for using range object as key for std::map or std::unordered_map
template <typename T> struct hash<mltools::Range<T>> {
  std::size_t operator()(mltools::Range<T> const &obj) const {
    return static_cast<size_t>(obj.begin() ^ obj.end() << 1);
  }
};

template <typename T> struct hash<std::pair<int, mltools::Range<T>>> {
  size_t operator()(std::pair<int, mltools::Range<T>> const &obj) const {
    return (obj.first ^ obj.second.begin() ^ obj.second.end());
  }
};

} // namespace std
