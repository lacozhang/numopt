#include "util/dynamic_array_impl.h"
#include "gtest/gtest.h"

using namespace mltools;

TEST(DArray, SetIntersection) {
  DArray<int> a{1, 2, 3, 4}, b{4, 5, 6}, c{4};
  EXPECT_EQ(a.setIntersection(b), c);
  EXPECT_NE(c.setIntersection(b), a);
}

TEST(DArray, setUnion) {
  DArray<int> a{1,2,3}, b{3,4,5}, c{1,2,3,4,5}, d{1,2,3,5,6};
  EXPECT_EQ(a.setUnion(b), c);
  EXPECT_NE(a.setUnion(c), b);
  EXPECT_NE(a.setUnion(b), d);
}

TEST(DArray, findRange) {
  DArray<int> a{1,2,3,4,5};
  SizeR res(0, 4), empty(5,5);
  EXPECT_EQ(a.findRange(SizeR(1,5)), res);
  EXPECT_EQ(a.findRange(SizeR(20, 30)), empty);
  EXPECT_NE(a.findRange(SizeR(20, 30)), res);
}

TEST(DArray, segment) {
  DArray<float> a{1,2,3,4,5,7,8};
  DArray<float> res{1, 2};
  EXPECT_EQ(a.segment(SizeR(0, 2)), res);
  CHECK_EQ(a.segment(SizeR(0, 4)).size(), 4);
  CHECK_EQ(a.segment(SizeR(0, 4)).data(), a.data());
  
  EXPECT_EQ(a.segment(SizeR(2, 4)).size(), 2);
  EXPECT_EQ(a.segment(SizeR(2, 4)).data(), a.data() + 2);
}

TEST(DArray, compress) {
  DArray<double> v(200);
  for(int i=0; i<200; ++i) {
    v[i] = i+0.5;
  }
  DArray<char> compressed = v.compressTo();
  DArray<double> rv;
  rv.uncompressFrom(compressed);
  
  CHECK_EQ(rv, v);
}

TEST(DArray, fileop) {
  const char *path = mktemp("/tmp/temp.XXXXXX");
  std::string fullpath(path, std::strlen(path));
  path = mktemp("/tmp/temp.XXXXXX");
  std::string partpath(path, std::strlen(path));
  DArray<double> v(200);
  for(int i=0; i<200; ++i) {
    v[i] = i+0.5;
  }
  v.writeToFile(fullpath);
  v.writeToFile(SizeR(10, 20), partpath);
  
  DArray<double> test;
  test.readFromFile(fullpath);
  CHECK_EQ(test, v);
  
  test.readFromFile(SizeR(10, 20), fullpath);
  CHECK_EQ(test, v.segment(SizeR(10, 20)));
  
  test.readFromFile(SizeR(100, 200), fullpath);
  CHECK_EQ(test, v.segment(SizeR(100, 200)));
  
  test.readFromFile(partpath);
  CHECK_EQ(test, v.segment(SizeR(10, 20)));
  
  test.readFromFile(SizeR(0, 5), partpath);
  CHECK_EQ(test, v.segment(SizeR(10, 15)));
  
  File::remove(fullpath);
  File::remove(partpath);
}


